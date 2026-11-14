"""Phase 8 demo: a full conversation through the orchestrator (router -> agents, with memory).

Run:  .venv/bin/python scripts/demo_phase8.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendsense.orchestrator import Orchestrator  # noqa: E402
from spendsense.rag import retrieval  # noqa: E402

retrieval.ensure_kb_indexed()
orch = Orchestrator()
sid = orch.session().id

print(">>> before uploading anything")
for q in ["Hi! What can you do?", "How much did I spend on food?"]:
    r = orch.chat(sid, q)
    print(f"  Q: {q}\n     [{r['intent']} -> {r['agent']}, {r['seconds']}s] {r['answer'][:220]}")

t0 = time.time()
s = orch.load_sample(sid)
print(f"\n>>> loaded sample in {time.time() - t0:.1f}s: {s['parse']['transactions']} txns, "
      f"categorised {s['categorize']['by_source']}, {s['alerts']} alerts, timings {s['timings']}")

conversation = [
    "Where did most of my money go in September?",
    "And how does that compare with August?",
    "Find that payment I made at a medical store",
    "OK, so how do I avoid overspending like that again?",
    "Thanks!",
]
for q in conversation:
    r = orch.chat(sid, q)
    d = r["details"]
    extra = ""
    if r["agent"] == "Analysis Agent":
        extra = f"tools={d['tools_used']} retrieval={d['used_retrieval']} models={d['models']}"
    elif r["agent"] == "Advisor Agent":
        extra = (f"kb_sources={d['sources']} critique={d['critique'].get('score')} revised={d['revised']} "
                 f"ungrounded={d['grounding'].get('draft_ungrounded')}")
    print(f"\n  Q: {q}\n     intent={r['intent']} entities={r['entities']} via {r['routing'].get('method')}"
          f"\n     -> {r['agent']} ({r['seconds']}s) {extra}\n     A: " + r["answer"].replace("\n", "\n        "))
print(f"\nhistory kept: {orch.session(sid).history.messages.__len__()} messages")
