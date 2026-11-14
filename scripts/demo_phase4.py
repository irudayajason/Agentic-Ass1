"""Phase 4 demo: categorise the sample statement with the real LLM and score it against ground truth.

Run:  .venv/bin/python scripts/demo_phase4.py
Uses a throwaway in-memory ChromaDB so repeated runs start from the same state.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from spendsense.agents.categorizer_agent import categorize  # noqa: E402
from spendsense.agents.parser_agent import parse_file  # noqa: E402
from spendsense.rag.vectorstore import ephemeral_client, get_store  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
df = parse_file(SAMPLES / "bank_statement.csv").df
truth = pd.read_csv(SAMPLES / "ground_truth.csv")

store = get_store("demo_examples", client=ephemeral_client())
t0 = time.time()
res = categorize(df, store=store)
elapsed = time.time() - t0
out = res.df.assign(truth=truth["category"].values)
out["ok"] = out["category"] == out["truth"]

s = res.summary()
print(f"Categorised {s['transactions']} transactions in {elapsed:.1f}s with {s['llm_calls']} LLM calls")
print(f"  by source: {s['by_source']}")
print(f"  payee groups sent to LLM: {s['groups_sent_to_llm']}  |  reflection reviewed {s['reflection_reviewed']}, "
      f"changed {s['reflection_changed']}")

print("\nAccuracy vs ground truth:")
for src, g in out.groupby("cat_source"):
    print(f"  {src:<16} {g['ok'].mean() * 100:6.1f}%  ({g['ok'].sum()}/{len(g)})")
print(f"  {'OVERALL':<16} {out['ok'].mean() * 100:6.1f}%  ({out['ok'].sum()}/{len(out)})")

print("\nLLM decisions per payee group:")
for t in res.trace:
    print(f"  {t['gid']:<4} {t['payee']:<26} {t['type']:<6} x{t['count']:<3} -> {t['category']:<20} "
          f"conf={t['confidence']:.2f} [{t['source']}]")
print(f"\n  e.g. examples retrieved for {res.trace[0]['payee']}: {res.trace[0]['examples']}")
print(f"  e.g. reasoning: {res.trace[0]['reasoning']}")

if res.reflection:
    print("\nReflection pass:")
    for r in res.reflection:
        print(f"  {r['payee']:<26} {r['original']} -> {r['final']}  ({'changed' if r['changed'] else 'kept'}): {r['critique']}")

wrong = out[~out["ok"]].groupby(["payee", "category", "truth"]).size()
if len(wrong):
    print("\nRemaining errors (payee, predicted, truth, count):")
    print(wrong.to_string())

# ------------------------------------------------------------------ ambiguous cases
print("\n=== Ambiguous transactions (to show the reflection pass) ===")
from spendsense.agents.parser_agent import extract_details  # noqa: E402

odd = [
    ("UPI/DR/1/S KRISHNAN/SBIN/skrish@oksbi/Advance", 15000.0, "DEBIT"),
    ("UPI/DR/2/MAHALAKSHMI ENTERPRISES/HDFC/mlent@ybl/Payment", 2340.0, "DEBIT"),
    ("UPI/CR/3/GOOGLE PAY/UTIB/gpay-rewards@okaxis/Cashback", 51.0, "CREDIT"),
    ("IMPS/P2A/4/NEW INDIA ASSURANCE/Policy 99812", 8620.0, "DEBIT"),
]
rows = [{"txn_id": f"X{i}", "date": pd.Timestamp("2026-11-02"), "description": d, "type": t, "amount": a,
         "balance": None, "ref": "", **extract_details(d)} for i, (d, a, t) in enumerate(odd, 1)]
amb = categorize(pd.DataFrame(rows), store=store)
for t in amb.trace:
    print(f"  {t['payee']:<26} -> {t['category']:<20} conf={t['confidence']:.2f} [{t['source']}]")
for r in amb.reflection:
    print(f"  self-check {r['payee']}: {r['original']} -> {r['final']} ({'changed' if r['changed'] else 'kept'}): {r['critique']}")
