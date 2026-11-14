"""Phase 5 demo: build the persistent ChromaDB (knowledge base + categorised sample statement)
and run similarity searches with and without metadata filters.

Run:  .venv/bin/python scripts/demo_phase5.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendsense.agents.categorizer_agent import categorize  # noqa: E402
from spendsense.agents.parser_agent import parse_file  # noqa: E402
from spendsense.rag import retrieval  # noqa: E402

t0 = time.time()
kb = retrieval.ensure_kb_indexed()
print(f"Knowledge base: {kb['notes']} notes -> {kb['chunks']} chunks "
      f"({'re-indexed' if kb['indexed'] else 'already up to date'}) in {time.time() - t0:.1f}s")

t0 = time.time()
df = categorize(parse_file(ROOT / "data/samples/bank_statement.csv").df).df
n = retrieval.index_transactions(df)
print(f"Transactions: {n} categorised and indexed in {time.time() - t0:.1f}s")
print(f"Collections: {retrieval.collection_stats()}")

print("\n=== Knowledge base search (relevance cut-off "
      f"{retrieval.KB_MIN_RELEVANCE}) ===")
for q in ["How big should my emergency fund be?",
          "I spend too much on Swiggy and Zomato",
          "Is it OK that my EMIs are 35% of my salary?",
          "How do I stop overspending in the Diwali sale?",
          "Who won the cricket match yesterday?"]:
    hits = retrieval.search_kb(q, k=3, min_relevance=0)
    print(f"\n  Q: {q}")
    for h in hits:
        keep = "kept   " if h["score"] >= retrieval.KB_MIN_RELEVANCE else "dropped"
        print(f"     {h['score']:.3f} {keep} {h['source']:<34} § {h['section']}")

print("\n=== Transaction search ===")
cases = [
    ("that medical store payment", {}),
    ("payments to my landlord", {}),
    ("food delivery orders", {"month": "2026-09", "min_amount": 500}),
    ("online shopping", {"type": "DEBIT", "date_from": "2026-09-20", "date_to": "2026-09-30"}),
    ("money my friends sent me", {"type": "CREDIT"}),
]
for q, filters in cases:
    print(f"\n  Q: {q!r}  filters={filters or 'none'}")
    for h in retrieval.search_transactions(q, k=4, **filters):
        print(f"     {h['score']:.3f}  {h['date']}  {h['payee']:<24} {h['type']:<6} ₹{h['amount']:>9,.2f}  {h['category']}")
