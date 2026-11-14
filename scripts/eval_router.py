"""Router accuracy on a labelled question set: LLM router vs the keyword fallback.

Run:  .venv/bin/python scripts/eval_router.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from spendsense.router import IntentRouter, keyword_intent  # noqa: E402

L, A, S, G = "transaction_lookup", "spending_analysis", "savings_guidance", "general"
CASES = [
    ("Show me all my Uber rides in August", L),
    ("List the transactions above ₹5,000", L),
    ("What did I pay Ramesh Kumar?", L),
    ("Find the payment I made at the pharmacy", L),
    ("Was there any refund from Amazon?", L),
    ("When did my salary come in October?", L),
    ("Which payments went to my friends?", L),
    ("How much did I spend in total?", A),
    ("What are my top 5 spending categories?", A),
    ("How much do I spend on food per month on average?", A),
    ("Compare my September and October spending", A),
    ("Which merchant do I spend the most at?", A),
    ("What share of my money goes to rent?", A),
    ("Did my transport costs go up?", A),
    ("How much did I earn over these months?", A),
    ("Am I overspending?", S),
    ("How can I cut down my food delivery bills?", S),
    ("Is my savings rate healthy?", S),
    ("Should I cancel some subscriptions?", S),
    ("Give me a budget plan for next month", S),
    ("How do I build an emergency fund from my salary?", S),
    ("What's one habit I should change to save money?", S),
    ("hello", G),
    ("What can you do?", G),
    ("How do I upload my statement?", G),
    ("Thanks, that was helpful!", G),
    ("What's the weather in Chennai?", G),
    ("Who made you?", G),
]
FOLLOW_UPS = [   # (previous question, previous answer, follow-up, expected)
    ("How much did I spend on Swiggy in September?", "You spent ₹511 on Swiggy in September.", "And in October?", A),
    ("Am I overspending on shopping?", "Yes, shopping rose 715% in September.", "What should I do about it?", S),
]

router = IntentRouter()
llm_ok = kw_ok = 0
wrong = []
t0 = time.time()
for q, exp in CASES:
    got, _ = router.classify(q)
    kw = keyword_intent(q).intent
    llm_ok += got.intent == exp
    kw_ok += kw == exp
    if got.intent != exp:
        wrong.append((q, exp, got.intent))
fu_ok = 0
for prev_q, prev_a, q, exp in FOLLOW_UPS:
    got, _ = router.classify(q, [HumanMessage(prev_q), AIMessage(prev_a)])
    fu_ok += got.intent == exp
    if got.intent != exp:
        wrong.append((f"{q} (after: {prev_q})", exp, got.intent))
    else:
        print(f"follow-up ok: {q!r} -> {got.intent}  entities={ {k: v for k, v in got.model_dump().items() if k not in ('intent','reasoning') and v} }")

n = len(CASES)
print(f"\nLLM router:        {llm_ok}/{n} = {llm_ok / n:.0%}   follow-ups {fu_ok}/{len(FOLLOW_UPS)}")
print(f"Keyword fallback:  {kw_ok}/{n} = {kw_ok / n:.0%}")
print(f"({time.time() - t0:.1f}s, {(time.time() - t0) / (n + len(FOLLOW_UPS)):.2f}s per question)")
for q, exp, got in wrong:
    print(f"  wrong: {q!r} expected {exp}, got {got}")
