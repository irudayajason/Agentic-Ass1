"""Phase 6 demo: ask the ReAct Analysis Agent real questions and check its numbers against pandas.

Run:  .venv/bin/python scripts/demo_phase6.py          (add -v to print every tool observation)
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from spendsense.agents.analysis_agent import AnalysisAgent  # noqa: E402
from spendsense.agents.categorizer_agent import categorize  # noqa: E402
from spendsense.agents.parser_agent import parse_file  # noqa: E402
from spendsense.rag import retrieval  # noqa: E402

VERBOSE = "-v" in sys.argv
df = categorize(parse_file(ROOT / "data/samples/bank_statement.csv").df).df
retrieval.index_transactions(df)
D = df[df["type"] == "DEBIT"]
month = lambda m: D[D["date"].dt.strftime("%Y-%m") == m]  # noqa: E731

by_month = D.groupby(D["date"].dt.strftime("%Y-%m"))["amount"].sum()
cases = [
    ("How much did I spend on Swiggy and Zomato in September?",
     month("2026-09").query("payee in ['SWIGGY', 'ZOMATO']")["amount"].sum()),
    ("Which month did I spend the most, and how much more was it than my average month?",
     by_month.max() - by_month.mean()),
    ("Compare my shopping spend in August vs September",
     month("2026-09").query("category == 'Shopping'")["amount"].sum()),
    ("What percentage of my total spending went to rent?",
     D.query("category == 'Rent'")["amount"].sum() / D["amount"].sum() * 100),
    ("What was that medical store payment in August?",
     month("2026-08").query("category == 'Health'")["amount"].iloc[0]),
    ("How much have I sent to Priya in total?",
     D[D["payee"].str.contains("PRIYA")]["amount"].sum()),
]


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)]


def show(q, expected, res, secs):
    found = any(abs(n - expected) <= max(1.0, abs(expected) * 0.005) for n in numbers_in(res.answer))
    print(f"\nQ: {q}")
    for s in res.steps:
        t = (s.thought or "").replace("\n", " ")
        print(f"   [{s.step}] {s.tool}({s.args})" + (f"\n       thought: {t[:150]}" if t else ""))
        if VERBOSE:
            print(f"       observation: {s.observation[:300]}")
    print(f"   retrieval used: {res.used_retrieval}   |   {secs:.1f}s")
    print("   A: " + res.answer.replace("\n", "\n      "))
    print(f"   expected key number ≈ {expected:,.2f} -> {'FOUND in answer ✓' if found else 'NOT found ✗'}")
    return found


agent = AnalysisAgent(lambda: df)
score = 0
for q, exp in cases:
    t0 = time.time()
    res = agent.run(q)
    score += show(q, exp, res, time.time() - t0)

# follow-up using conversation memory
print("\n--- Follow-up with conversation memory ---")
first = agent.run(cases[0][0])
history = [HumanMessage(cases[0][0]), AIMessage(first.answer)]
t0 = time.time()
follow = agent.run("And in October?", history=history)
score += show("And in October?  (after the Swiggy/Zomato question)",
              month("2026-10").query("payee in ['SWIGGY', 'ZOMATO']")["amount"].sum(), follow, time.time() - t0)

print(f"\n{score}/{len(cases) + 1} answers contain the correct key number")
