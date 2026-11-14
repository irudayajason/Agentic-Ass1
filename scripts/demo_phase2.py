"""Phase 2 demo: the prompt catalogue plus three live LCEL chains on Groq.

Run:  .venv/bin/python scripts/demo_phase2.py
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendsense.chains import structured_chain, text_chain  # noqa: E402
from spendsense.llm import get_router_llm  # noqa: E402
from spendsense.prompts import prompt_catalog  # noqa: E402
from spendsense.schemas import ColumnMapping, Intent  # noqa: E402

TODAY = date.today().isoformat()

print("=== Prompt catalogue ===")
for p in prompt_catalog():
    print(f"  {p['agent']:<30} {p['title']:<28} {', '.join(p['techniques'])}")

print("\n=== 1. Router chain: few-shot + structured output (gpt-oss-20b) ===")
router = structured_chain("router", Intent, llm=get_router_llm())
questions = [
    "How much did I spend at Swiggy?",
    "Show me the payments I made to Priya",
    "I think I'm spending way too much, help me cut back",
    "What was that pharmacy payment in August?",
    "Which month did I spend the most?",
    "hello, how do I use this?",
]
for q in questions:
    r = router.invoke({"question": q, "today": TODAY})
    ents = {k: v for k, v in r.model_dump().items() if k not in ("intent", "reasoning") and v is not None}
    print(f"  {q!r:<56} -> {r.intent:<19} {ents}")

print("\n=== 2. Parser mapping chain: few-shot + structured output on the sample bank CSV ===")
lines = (ROOT / "data/samples/bank_statement.csv").read_text().splitlines()[:9]
sample = "\n".join(f"line {i}: {l}" for i, l in enumerate(lines))
mapping = structured_chain("parser_mapping", ColumnMapping).invoke({"sample_lines": sample})
print("  " + "\n  ".join(f"{k} = {v!r}" for k, v in mapping.model_dump().items() if v is not None))

print("\n=== 3. General chain: zero-shot text ===")
print("  " + text_chain("general", llm=get_router_llm()).invoke({"question": "What can you do?", "has_data": "no"}))
