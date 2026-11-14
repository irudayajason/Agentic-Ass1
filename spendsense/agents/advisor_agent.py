"""Advisor Agent: overspending alerts, savings advice and plain-language insights.

Pipeline for a question (each step is an LCEL chain, split across two models so a single
answer stays inside the free Groq tier's tokens-per-minute allowance):

  1. Alerts        rule engine finds problems and measures them in rupees     (pandas, no LLM)
  2. Plan          decide IF the knowledge base is needed, and write queries   (fast model)   <- Agentic RAG
  3. Retrieve      similarity search in ChromaDB finance_kb, weak hits dropped (vector DB)
  4. Draft         answer from figures + alerts + retrieved guidance, cited    (main model)   <- CoT, RAG
  5. Critique      grade the draft: grounded? cited? actionable?               (fast model)   <- Reflection
  6. Revise        rewrite to fix the issues, only if the critique asks for it (main model)   <- Reflection
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.messages import BaseMessage

from spendsense.chains import structured_chain, text_chain
from spendsense.llm import get_fast_llm, get_llm
from spendsense.schemas import AdviceCritique, RetrievalPlan
from spendsense.tools.alerts import alerts_text, detect_alerts, financial_profile, profile_text

PASS_SCORE = 3            # critique score at or above this (and grounded) -> keep the draft as is
KB_HITS_PER_QUERY = 3


@dataclass
class AdvisorResult:
    answer: str
    alerts: list[dict]
    plan: dict
    retrieved: list[dict] = field(default_factory=list)
    draft: str = ""
    critique: dict = field(default_factory=dict)
    revised: bool = False
    grounding: dict = field(default_factory=dict)   # programmatic number check on draft / revision
    timings: dict = field(default_factory=dict)

    @property
    def used_retrieval(self) -> bool:
        return bool(self.retrieved)

    @property
    def sources(self) -> list[str]:
        return sorted({r["source"] for r in self.retrieved})

    def to_dict(self) -> dict:
        return {"answer": self.answer, "plan": self.plan, "retrieved": self.retrieved,
                "used_retrieval": self.used_retrieval, "sources": self.sources, "draft": self.draft,
                "critique": self.critique, "revised": self.revised, "grounding": self.grounding,
                "timings": self.timings,
                "alerts_used": [a["title"] for a in self.alerts[:10]]}


_MONEY = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)")
_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
_ANY_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def ungrounded_numbers(answer: str, reference: str) -> list[str]:
    """Rupee amounts and percentages in the answer that don't match any number in the reference text.

    A programmatic grounding check: it doesn't trust the LLM's own critique. Small rounding
    differences (₹0.5, 0.05%) and amounts the answer writes as '₹13 k' are tolerated.
    """
    ref = [_num(n) for n in _ANY_NUM.findall(reference)]
    bad = []
    for m in _MONEY.finditer(answer):
        v = _num(m.group(1))
        tail = answer[m.end():m.end() + 2].strip().lower()
        if tail.startswith("k"):           # '₹13 k' style rounding: allow any reference within 1,000
            ok = any(abs(r - v * 1000) <= 1000 for r in ref)
        else:
            ok = any(abs(r - v) <= 0.5 for r in ref)
        if not ok:
            bad.append(m.group(0))
    for m in _PCT.finditer(answer):
        v = float(m.group(1))
        if not any(abs(r - v) <= 0.05 for r in ref):
            bad.append(m.group(0))
    return list(dict.fromkeys(bad))


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "No guidance retrieved (the plan decided it was not needed, or nothing relevant was found)."
    return "\n\n".join(f"[source: {h['source']}] ({h['title']} - {h['section']}, relevance {h['score']})\n"
                       f"{h['text'].split(chr(10), 1)[-1]}" for h in hits)


class AdvisorAgent:
    def __init__(self, get_df, chroma_client=None, main_llm=None, fast_llm=None, search_fn=None):
        self.get_df = get_df
        main, fast = main_llm or get_llm(), fast_llm or get_fast_llm()
        self.plan_chain = structured_chain("advisor_retrieval_plan", RetrievalPlan, llm=fast, fallback_llm=main)
        self.draft_chain = text_chain("advisor", llm=main)
        self.critique_chain = structured_chain("advisor_critique", AdviceCritique, llm=fast, fallback_llm=main)
        self.revise_chain = text_chain("advisor_revise", llm=main)
        self.summary_chain = text_chain("insight_summary", llm=fast)
        if search_fn is None:
            from spendsense.rag.retrieval import search_kb
            search_fn = lambda q, k: search_kb(q, k=k, client=chroma_client)  # noqa: E731
        self.search = search_fn

    # ------------------------------------------------------------------ question answering
    def run(self, question: str, history: list[BaseMessage] | None = None, entities: dict | None = None) -> AdvisorResult:
        t = {}
        t0 = time.time()
        df = self.get_df()
        alerts = detect_alerts(df)
        if entities and entities.get("category"):   # bring alerts about the asked-about category to the top
            cat = entities["category"].lower()
            alerts = sorted(alerts, key=lambda a: (a.get("category") or "").lower() != cat)
        figures = profile_text(financial_profile(df))
        a_text = alerts_text(alerts)
        t["alerts"] = round(time.time() - t0, 2)

        # 2. plan: does this need the knowledge base?
        t0 = time.time()
        plan: RetrievalPlan = self.plan_chain.invoke({"question": question, "alerts": alerts_text(alerts, 5)})
        t["plan"] = round(time.time() - t0, 2)

        # 3. retrieve (only if the plan says so)
        hits: list[dict] = []
        if plan.need_retrieval:
            t0 = time.time()
            seen = set()
            for q in plan.queries[:3]:
                for h in self.search(q, KB_HITS_PER_QUERY):
                    key = (h["source"], h["section"])
                    if key not in seen:
                        seen.add(key)
                        hits.append({**h, "query": q})
            hits = sorted(hits, key=lambda h: -h["score"])[:5]
            t["retrieve"] = round(time.time() - t0, 2)
        context = format_context(hits)

        # 4. draft
        t0 = time.time()
        inputs = {"figures": figures, "alerts": a_text, "context": context, "question": question,
                  "history": history or []}
        draft = self.draft_chain.invoke(inputs).strip()
        t["draft"] = round(time.time() - t0, 2)

        # 5. critique (reflection)
        t0 = time.time()
        try:
            crit: AdviceCritique | None = self.critique_chain.invoke(
                {"figures": figures, "context": context, "question": question, "draft": draft})
        except Exception:  # noqa: BLE001 - both models failed; keep the draft rather than fail the answer
            crit = None
        t["critique"] = round(time.time() - t0, 2)

        # programmatic grounding check: every ₹ amount / % must exist in what the model was given
        reference = "\n".join([figures, a_text, context, question])
        draft_bad = ungrounded_numbers(draft, reference)
        grounding = {"draft_ungrounded": draft_bad}

        # 6. revise only when needed
        answer, revised = draft, False
        if crit is None:
            return AdvisorResult(answer=answer, alerts=alerts, plan=plan.model_dump(), retrieved=hits, draft=draft,
                                 critique={"skipped": True, "reason": "critique step failed on both models"},
                                 revised=False, grounding=grounding, timings=t)
        # Revise when the draft is weak, the critic says it isn't grounded, or the number check found
        # invented figures. Minor nitpicks on a grounded draft don't justify another call.
        if crit.score < PASS_SCORE or not crit.grounded_in_numbers or len(draft_bad) > 1:
            issues = list(crit.issues) + [f"{x} does not appear in the user's figures; remove it or use a figure "
                                          f"from the list" for x in draft_bad[:6]]
            t0 = time.time()
            candidate = self.revise_chain.invoke({"figures": figures, "context": context, "question": question,
                                                  "draft": draft, "issues": "\n".join(f"- {i}" for i in issues)
                                                  or "- Improve overall quality"}).strip()
            t["revise"] = round(time.time() - t0, 2)
            cand_bad = ungrounded_numbers(candidate, reference)
            grounding["revision_ungrounded"] = cand_bad
            if len(cand_bad) <= len(draft_bad):
                answer, revised = candidate, True
            else:   # the rewrite invented more numbers than the draft: keep the draft
                grounding["revision_rejected"] = True

        return AdvisorResult(answer=answer, alerts=alerts, plan=plan.model_dump(), retrieved=hits, draft=draft,
                             critique=crit.model_dump(), revised=revised, grounding=grounding, timings=t)

    # ------------------------------------------------------------------ insights tab
    def insights(self) -> dict:
        """Alerts, the 50/30/20 split, subscriptions and a short plain-language summary."""
        df = self.get_df()
        profile = financial_profile(df)
        alerts = detect_alerts(df)
        stats = profile_text(profile) + "\n\nAlerts:\n" + alerts_text(alerts, 8)
        try:
            summary = self.summary_chain.invoke({"stats": stats}).strip()
        except Exception as exc:  # noqa: BLE001 - alerts and figures don't need the LLM
            quota = "rate" in str(exc).lower() or "429" in str(exc)
            summary = ("_The written summary needs the LLM, which is unavailable right now"
                       + (" (the free Groq quota is used up; it refills within the hour)" if quota else "")
                       + ". The alerts and figures below are computed without it._")
        return {"summary": summary, "alerts": alerts, "budget_split": profile["budget_split"],
                "subscriptions": profile["subscriptions"], "months": profile["months"]}


def quick_alerts(df: pd.DataFrame) -> list[dict]:
    """Alerts only, no LLM (used right after upload so the dashboard can show them instantly)."""
    return detect_alerts(df)
