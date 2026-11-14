"""Phase 7 demo: alerts, then three Advisor questions, then the Insights summary, all live on Groq.

Run:  .venv/bin/python scripts/demo_phase7.py
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendsense.agents.advisor_agent import AdvisorAgent  # noqa: E402
from spendsense.agents.categorizer_agent import categorize  # noqa: E402
from spendsense.agents.parser_agent import parse_file  # noqa: E402
from spendsense.rag import retrieval  # noqa: E402
from spendsense.tools.alerts import alerts_text, detect_alerts, financial_profile, profile_text  # noqa: E402

retrieval.ensure_kb_indexed()
df = categorize(parse_file(ROOT / "data/samples/bank_statement.csv").df).df
agent = AdvisorAgent(lambda: df)
figures = profile_text(financial_profile(df))

print("=== Alerts (rule engine, no LLM) ===")
print(alerts_text(detect_alerts(df), 20))

questions = [
    ("Am I overspending? Where should I cut back first?", {}),
    ("How can I save ₹5,000 more every month?", {"amount": 5000}),
    ("What was my savings rate in July?", {}),
]
for q, ents in questions:
    t0 = time.time()
    r = agent.run(q, entities=ents)
    amounts = re.findall(r"₹[\d,]+", r.answer)
    in_figures = [a for a in amounts if a in figures or a in alerts_text(r.alerts, 30)]
    print(f"\n{'=' * 100}\nQ: {q}   ({time.time() - t0:.1f}s, steps {r.timings})")
    print(f"  1. plan: need_retrieval={r.plan['need_retrieval']}  queries={r.plan['queries']}  reason: {r.plan['reason']}")
    print(f"  2. retrieved: " + (", ".join(f"{h['source']} § {h['section']} ({h['score']})" for h in r.retrieved) or "nothing"))
    c = r.critique
    print(f"  3. critique: score={c['score']} grounded={c['grounded_in_numbers']} cites={c['cites_sources']} "
          f"actionable={c['actionable']} issues={c['issues']}")
    print(f"  4. revised: {r.revised}")
    print(f"  ₹ amounts in answer: {len(amounts)}, of which {len(in_figures)} appear verbatim in the figures/alerts "
          f"(the rest are estimates the advisor worked out)")
    print("  ANSWER:\n    " + r.answer.replace("\n", "\n    "))

print(f"\n{'=' * 100}\n=== Insights tab summary ===")
t0 = time.time()
ins = agent.insights()
print(f"({time.time() - t0:.1f}s)\n" + ins["summary"])
