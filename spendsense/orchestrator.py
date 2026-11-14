"""Orchestrator: ties the agents together for the web app.

Ingest (on upload):
    Parser Agent -> Categorizer Agent -> ChromaDB transaction index -> rule alerts

Chat (each message), built with LCEL:

    RunnableWithMessageHistory(                      <- conversation memory per session
        RunnablePassthrough.assign(intent=router)    <- intent detection (few-shot, structured)
        | RunnableBranch(
            (no statement loaded,      no_data reply),
            (transaction_lookup /
             spending_analysis,        Analysis Agent: ReAct + pandas tools + vector search),
            (savings_guidance,         Advisor Agent: alerts + Agentic RAG + reflection),
            default:                   general chain (zero-shot),
        )
    )

Every reply comes back with a trace (intent, route, tool calls, retrievals, critique, timings)
for the "agent reasoning" panel in the UI.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from spendsense import config
from spendsense.agents.advisor_agent import AdvisorAgent
from spendsense.agents.analysis_agent import AnalysisAgent
from spendsense.agents.categorizer_agent import apply_user_correction, categorize
from spendsense.agents.parser_agent import parse_statement
from spendsense.chains import text_chain
from spendsense.llm import get_fast_llm
from spendsense.memory import TrimmedChatHistory
from spendsense.rag import retrieval
from spendsense.router import IntentRouter
from spendsense.tools.alerts import detect_alerts

ANALYSIS_INTENTS = {"transaction_lookup", "spending_analysis"}


@dataclass
class Session:
    id: str
    df: pd.DataFrame | None = None
    filename: str | None = None
    ingest_summary: dict = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    history: TrimmedChatHistory = field(default_factory=TrimmedChatHistory)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    analysis: AnalysisAgent | None = None
    advisor: AdvisorAgent | None = None

    @property
    def has_data(self) -> bool:
        return self.df is not None and not self.df.empty


class Orchestrator:
    def __init__(self, chroma_client=None, router: IntentRouter | None = None, analysis_factory=None,
                 advisor_factory=None, general_chain=None, categorize_with_llm: bool = True):
        self.client = chroma_client
        self.router = router or IntentRouter()
        self.analysis_factory = analysis_factory or (lambda s: AnalysisAgent(lambda: s.df, chroma_client=self.client))
        self.advisor_factory = advisor_factory or (lambda s: AdvisorAgent(lambda: s.df, chroma_client=self.client))
        self.general_chain = general_chain or text_chain("general", llm=get_fast_llm())
        self.categorize_with_llm = categorize_with_llm
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.chat_chain = self._build_chat_chain()

    # ================================================================ sessions
    def session(self, session_id: str | None = None) -> Session:
        with self._lock:
            sid = session_id or uuid.uuid4().hex[:12]
            if sid not in self.sessions:
                self.sessions[sid] = Session(id=sid)
            return self.sessions[sid]

    def reset(self, session_id: str) -> Session:
        with self._lock:
            self.sessions[session_id] = Session(id=session_id)
            return self.sessions[session_id]

    # ================================================================ ingest
    def ingest(self, session_id: str, data: bytes, filename: str) -> dict:
        s = self.session(session_id)
        t: dict[str, float] = {}

        t0 = time.time()
        parsed = parse_statement(data, filename)                       # Parser Agent
        t["parse"] = round(time.time() - t0, 2)

        t0 = time.time()
        store = None
        if self.client is not None:
            from spendsense.rag.vectorstore import get_store
            store = get_store(config.EXAMPLES_COLLECTION, self.client)
        cat = categorize(parsed.df, store=store, use_llm=self.categorize_with_llm)   # Categorizer Agent
        t["categorize"] = round(time.time() - t0, 2)

        t0 = time.time()
        indexed = retrieval.index_transactions(cat.df, client=self.client)          # ChromaDB
        t["index"] = round(time.time() - t0, 2)

        s.df, s.filename = cat.df, filename
        s.alerts = detect_alerts(cat.df)                                            # rule alerts
        s.analysis, s.advisor = self.analysis_factory(s), self.advisor_factory(s)
        s.history.clear()
        s.ingest_summary = {"filename": filename, "parse": parsed.summary(), "categorize": cat.summary(),
                            "categorize_trace": cat.trace, "reflection": cat.reflection,
                            "indexed": indexed, "alerts": len(s.alerts), "timings": t}
        return s.ingest_summary

    def load_sample(self, session_id: str, name: str = "bank_statement.csv") -> dict:
        path = config.SAMPLES_DIR / Path(name).name      # Path(...).name blocks '../' tricks
        if not path.exists():
            raise FileNotFoundError(f"No sample called {name}")
        return self.ingest(session_id, path.read_bytes(), path.name)

    def set_category(self, session_id: str, txn_id: str, category: str, apply_to_same_payee: bool = True) -> int:
        s = self.session(session_id)
        if not s.has_data:
            raise ValueError("No statement loaded")
        store = None
        if self.client is not None:
            from spendsense.rag.vectorstore import get_store
            store = get_store(config.EXAMPLES_COLLECTION, self.client)
        before = s.df.copy()
        s.df = apply_user_correction(s.df, txn_id, category, store=store, apply_to_same_payee=apply_to_same_payee)
        changed = s.df.index[(before["category"] != s.df["category"]) | (s.df["txn_id"] == txn_id)]
        for _, row in s.df.loc[changed].iterrows():                                 # re-embed changed rows
            retrieval.update_transaction_category(row, client=self.client)
        s.alerts = detect_alerts(s.df)
        return len(changed)

    # ================================================================ chat (LCEL)
    def _route(self, x: dict) -> dict:
        intent, how = self.router.classify(x["question"], x.get("history", []))
        return {**x, "intent": intent, "route_info": how}

    def _no_data(self, x: dict) -> dict:
        return {"answer": "Please upload a bank or UPI statement (CSV or PDF) first, or click "
                          "**Load sample statement** to try it with demo data.", "agent": "none", "details": {}}

    def _analysis(self, x: dict) -> dict:
        s: Session = x["session"]
        intent = x["intent"]
        ents = {k: v for k, v in intent.model_dump().items() if k not in ("intent", "reasoning") and v is not None}
        res = s.analysis.run(x["question"], history=x.get("history", []), intent=intent.intent, entities=ents)
        return {"answer": res.answer, "agent": "Analysis Agent", "details": res.to_dict()}

    def _advisor(self, x: dict) -> dict:
        s: Session = x["session"]
        ents = {k: v for k, v in x["intent"].model_dump().items() if k not in ("intent", "reasoning") and v}
        res = s.advisor.run(x["question"], history=x.get("history", []), entities=ents)
        return {"answer": res.answer, "agent": "Advisor Agent", "details": res.to_dict()}

    def _general(self, x: dict) -> dict:
        s: Session = x["session"]
        answer = self.general_chain.invoke({"question": x["question"], "history": x.get("history", []),
                                            "has_data": f"yes, {s.filename}" if s.has_data else "no"})
        return {"answer": answer.strip(), "agent": "General responder", "details": {}}

    def _build_chat_chain(self):
        branch = RunnableBranch(
            (lambda x: not x["session"].has_data and x["intent"].intent != "general", RunnableLambda(self._no_data)),
            (lambda x: x["intent"].intent in ANALYSIS_INTENTS, RunnableLambda(self._analysis)),
            (lambda x: x["intent"].intent == "savings_guidance", RunnableLambda(self._advisor)),
            RunnableLambda(self._general),
        )

        # route -> branch -> merge the agent's result with the routing info for the trace
        core = (RunnableLambda(self._route)
                | RunnablePassthrough.assign(result=branch)
                | RunnableLambda(lambda x: {**x["result"], "intent": x["intent"], "route_info": x["route_info"]}))
        return RunnableWithMessageHistory(
            core,
            lambda session_id: self.session(session_id).history,
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="answer",
        )

    def chat(self, session_id: str, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Empty question")
        s = self.session(session_id)
        t0 = time.time()
        out = self.chat_chain.invoke({"question": question[:2000], "session": s},
                                     config={"configurable": {"session_id": s.id}})
        intent = out["intent"]
        return {
            "answer": out["answer"],
            "intent": intent.intent,
            "entities": {k: v for k, v in intent.model_dump().items() if k not in ("intent", "reasoning") and v is not None},
            "intent_reasoning": intent.reasoning,
            "routing": out.get("route_info", {}),
            "agent": out["agent"],
            "details": out["details"],
            "seconds": round(time.time() - t0, 2),
            "history_length": len(s.history.messages),
            "date": date.today().isoformat(),
        }
