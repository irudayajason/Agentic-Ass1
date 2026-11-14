"""Intent detection.

Primary: the few-shot `router` prompt on the fast model with structured output (`Intent`).
Fallback: keyword rules, used if the LLM call fails, so routing never blocks a reply.
The orchestrator uses the result to pick a branch in an LCEL `RunnableBranch`.
"""

from __future__ import annotations

import re
import time
from datetime import date

from langchain_core.messages import BaseMessage

from spendsense.chains import structured_chain
from spendsense.llm import get_llm, get_router_llm
from spendsense.schemas import Intent

HISTORY_FOR_ROUTER = 4   # last few messages are enough to resolve "and last month?"

_GENERAL = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b|what can you do|"
                      r"how (do|can) i (use|upload)|\bhelp\b|who are you", re.I)
_SAVINGS = re.compile(r"\bsav(e|ing|ings)\b|overspend|budget|cut (back|down)|reduce|advice|tips?\b|"
                      r"too much|afford|should i|habit|50/30/20|emergency fund", re.I)
_LOOKUP = re.compile(r"\b(show|list|find|which transactions?|what was that|payments? to|paid to|"
                     r"sent to|received from|transactions? (on|from|to))\b", re.I)


def keyword_intent(question: str) -> Intent:
    q = question.strip()
    if _GENERAL.search(q) and len(q.split()) <= 8:
        name = "general"
    elif _SAVINGS.search(q):
        name = "savings_guidance"
    elif _LOOKUP.search(q):
        name = "transaction_lookup"
    elif re.search(r"\b(spend|spent|cost|total|how much|month|category|trend|compare|most|top)\b", q, re.I):
        name = "spending_analysis"
    else:
        name = "general"
    return Intent(reasoning="Keyword fallback", intent=name)


class IntentRouter:
    def __init__(self, llm=None):
        self.chain = structured_chain("router", Intent, llm=llm or get_router_llm(),
                                      fallback_llm=None if llm else get_llm())

    def classify(self, question: str, history: list[BaseMessage] | None = None,
                 today: str | None = None) -> tuple[Intent, dict]:
        """Return the intent and how it was decided (for the trace)."""
        t0 = time.time()
        try:
            intent = self.chain.invoke({"question": question, "history": (history or [])[-HISTORY_FOR_ROUTER:],
                                        "today": today or date.today().isoformat()})
            method = "llm (few-shot, structured output)"
        except Exception as exc:  # noqa: BLE001 - never fail a reply because routing failed
            intent = keyword_intent(question)
            method = f"keyword fallback ({type(exc).__name__})"
        return intent, {"method": method, "seconds": round(time.time() - t0, 2)}
