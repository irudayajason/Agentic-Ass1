"""Phase 3 demo: parse all four sample statements; the last one needs the LLM fallback.

Run:  .venv/bin/python scripts/demo_phase3.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from spendsense.agents.parser_agent import parse_file  # noqa: E402

pd.set_option("display.width", 160, "display.max_colwidth", 48)
SAMPLES = ROOT / "data" / "samples"

for name in ["bank_statement.csv", "upi_statement.csv", "bank_statement.pdf", "unusual_layout_oct.csv"]:
    res = parse_file(SAMPLES / name)
    s = res.summary()
    bc = s["balance_check"]
    print(f"\n=== {name} ===")
    print(f"  method={s['method']}  transactions={s['transactions']}  rows_read={s['rows_read']}  dropped={s['dropped']}")
    print(f"  dates {s['date_range'][0]} -> {s['date_range'][1]}   debit ₹{s['total_debit']:,.2f}   credit ₹{s['total_credit']:,.2f}")
    print(f"  modes {s['modes']}")
    print(f"  balance check: " + (f"{bc['mismatches']} mismatches in {bc['rows_checked']} rows" if bc["checked"] else bc["reason"]))
    if s["method"] == "llm_mapping":
        print(f"  LLM column mapping: { {k: v for k, v in s['mapping'].items() if v is not None} }")
    for w in s["warnings"]:
        print(f"  warning: {w}")
    print(res.df[["txn_id", "date", "payee", "mode", "type", "amount", "note"]].head(4).to_string(index=False))
