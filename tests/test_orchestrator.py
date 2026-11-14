"""Router fallback, memory and the orchestrator's LCEL routing with fake agents (no API calls)."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from spendsense.memory import MAX_MESSAGES, TrimmedChatHistory
from spendsense.orchestrator import Orchestrator
from spendsense.rag.vectorstore import ephemeral_client
from spendsense.router import IntentRouter, keyword_intent
from spendsense.schemas import Intent

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.mark.parametrize("q,expected", [
    ("hi", "general"), ("What can you do?", "general"),
    ("How can I save more money?", "savings_guidance"), ("Am I overspending on food?", "savings_guidance"),
    ("Show my payments to Rahul", "transaction_lookup"), ("Find that pharmacy payment", "transaction_lookup"),
    ("How much did I spend in September?", "spending_analysis"), ("Compare July and August", "spending_analysis"),
])
def test_keyword_fallback(q, expected):
    assert keyword_intent(q).intent == expected


def test_router_falls_back_when_llm_fails():
    class Broken:
        def with_structured_output(self, *a, **k):
            return RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("API down")))
    intent, how = IntentRouter(llm=Broken()).classify("Show my payments to Priya")
    assert intent.intent == "transaction_lookup" and how["method"].startswith("keyword fallback")


def test_trimmed_history():
    h = TrimmedChatHistory()
    for i in range(20):
        h.add_messages([HumanMessage(f"q{i}"), AIMessage("x" * 5000)])
    assert len(h.messages) == MAX_MESSAGES and h.messages[-2].content == "q19"
    assert len(h.messages[-1].content) < 2000


# ------------------------------------------------------------------ orchestrator with fakes
class FakeRouter:
    def __init__(self):
        self.seen_history = []

    def classify(self, question, history=None, today=None):
        self.seen_history.append(list(history or []))
        return keyword_intent(question), {"method": "fake"}


class FakeAgentResult:
    def __init__(self, answer):
        self.answer = answer

    def to_dict(self):
        return {"answer": self.answer}


class FakeAgent:
    def __init__(self, name):
        self.name, self.calls = name, []

    def run(self, question, history=None, **kw):
        self.calls.append({"question": question, "history": list(history or []), **kw})
        return FakeAgentResult(f"{self.name} answered: {question}")


@pytest.fixture
def orch():
    analysis, advisor = FakeAgent("analysis"), FakeAgent("advisor")
    o = Orchestrator(chroma_client=ephemeral_client(), router=FakeRouter(),
                     analysis_factory=lambda s: analysis, advisor_factory=lambda s: advisor,
                     general_chain=RunnableLambda(lambda x: f"general: has_data={x['has_data']}"),
                     categorize_with_llm=False)
    o._fakes = (analysis, advisor)
    return o


def test_no_data_guard_and_general_without_data(orch):
    sid = orch.session().id
    r = orch.chat(sid, "How much did I spend in September?")
    assert r["agent"] == "none" and "upload" in r["answer"].lower()
    g = orch.chat(sid, "hi")
    assert g["agent"] == "General responder" and "has_data=no" in g["answer"]


def test_ingest_then_routes_to_each_agent(orch):
    sid = orch.session().id
    summary = orch.load_sample(sid)
    assert summary["parse"]["transactions"] == 336 and summary["indexed"] == 336
    analysis, advisor = orch._fakes
    assert orch.chat(sid, "How much did I spend in September?")["agent"] == "Analysis Agent"
    assert orch.chat(sid, "Show my payments to Rahul")["intent"] == "transaction_lookup"
    assert orch.chat(sid, "How can I save more money?")["agent"] == "Advisor Agent"
    assert orch.chat(sid, "hello")["answer"].startswith("general: has_data=yes")
    assert len(analysis.calls) == 2 and len(advisor.calls) == 1


def test_memory_is_passed_to_router_and_agents(orch):
    sid = orch.session().id
    orch.load_sample(sid)
    orch.chat(sid, "How much did I spend on Swiggy in September?")
    r = orch.chat(sid, "How much did I spend on Swiggy in October?")
    analysis, _ = orch._fakes
    second = analysis.calls[-1]["history"]
    assert [m.type for m in second] == ["human", "ai"]
    assert "September" in second[0].content
    assert r["history_length"] == 4
    assert orch.router.seen_history[-1] == second


def test_sessions_are_isolated(orch):
    a, b = orch.session().id, orch.session().id
    orch.load_sample(a)
    assert orch.chat(b, "How much did I spend?")["agent"] == "none"


def test_user_correction_updates_df_and_alerts(orch):
    sid = orch.session().id
    orch.load_sample(sid)
    s = orch.session(sid)
    tid = s.df.loc[s.df["payee"] == "CULT FIT", "txn_id"].iloc[0]
    n = orch.set_category(sid, tid, "Entertainment")
    assert n == (s.df["payee"] == "CULT FIT").sum()
    assert set(s.df.loc[s.df["payee"] == "CULT FIT", "category"]) == {"Entertainment"}


def test_sample_name_cannot_escape_samples_dir(orch):
    with pytest.raises(FileNotFoundError):
        orch.load_sample(orch.session().id, "../../.env")
