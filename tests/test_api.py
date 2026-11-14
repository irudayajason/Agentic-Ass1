"""API tests with FastAPI's TestClient and an orchestrator using fake agents (no LLM calls)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from server import create_app
from spendsense.orchestrator import Orchestrator
from spendsense.rag import retrieval
from spendsense.rag.vectorstore import ephemeral_client
from tests.test_orchestrator import FakeAgent, FakeRouter

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


class FakeAdvisor(FakeAgent):
    def insights(self):
        return {"summary": "fake summary", "alerts": []}


@pytest.fixture(scope="module")
def client():
    c = ephemeral_client()
    retrieval.ensure_kb_indexed(client=c)
    orch = Orchestrator(chroma_client=c, router=FakeRouter(), analysis_factory=lambda s: FakeAgent("analysis"),
                        advisor_factory=lambda s: FakeAdvisor("advisor"),
                        general_chain=RunnableLambda(lambda x: "hello!"), categorize_with_llm=False)
    # route the server's retrieval calls to the same in-memory DB
    import server
    orig_kb, orig_tx = retrieval.search_kb, retrieval.search_transactions
    server.retrieval.search_kb = lambda q, k=4, min_relevance=0.55: orig_kb(q, k=k, min_relevance=min_relevance, client=c)
    server.retrieval.search_transactions = lambda q, k=8, **kw: orig_tx(q, k=k, client=c, **kw)
    with TestClient(create_app(orch)) as tc:
        yield tc
    server.retrieval.search_kb, server.retrieval.search_transactions = orig_kb, orig_tx


H = {"X-Session-Id": "test-session-1"}


def test_health_and_no_data(client):
    r = client.get("/api/health", headers=H).json()
    assert r["ok"] and r["session_id"] == "test-session-1" and not r["has_data"]
    assert "GROQ" not in str(r) and "gsk_" not in str(r)            # key never leaves the server
    assert client.get("/api/dashboard", headers=H).status_code == 409


def test_bad_session_id_rejected(client):
    assert client.get("/api/health", headers={"X-Session-Id": "../../etc"}).status_code == 400


def test_upload_rejects_bad_files(client):
    r = client.post("/api/upload", headers=H, files={"file": ("x.xlsx", b"abc", "application/octet-stream")})
    assert r.status_code == 422
    r = client.post("/api/upload", headers=H, files={"file": ("x.csv", b"just,some\nrandom,text\n", "text/csv")})
    assert r.status_code in (422, 500) or r.json()


def test_upload_csv_and_dashboard(client):
    data = (SAMPLES / "bank_statement.csv").read_bytes()
    r = client.post("/api/upload", headers=H, files={"file": ("bank_statement.csv", data, "text/csv")})
    assert r.status_code == 200 and r.json()["parse"]["transactions"] == 336
    d = client.get("/api/dashboard", headers=H).json()
    assert d["kpis"]["transactions"] == 336 and len(d["monthly"]) == 4
    assert abs(sum(c["total"] for c in d["categories"]) - d["kpis"]["total_spent"]) < 1


def test_transactions_filters_and_edit(client):
    r = client.get("/api/transactions", headers=H, params={"q": "swiggy", "type": "DEBIT"}).json()
    assert r["total"] > 0 and all("SWIGGY" in row["payee"] for row in r["rows"])
    page2 = client.get("/api/transactions", headers=H, params={"page": 2, "page_size": 10}).json()
    assert page2["page"] == 2 and len(page2["rows"]) == 10
    tid = client.get("/api/transactions", headers=H, params={"q": "cult fit"}).json()["rows"][0]["txn_id"]
    ok = client.patch(f"/api/transactions/{tid}", headers=H, json={"category": "Health"})
    assert ok.status_code == 200 and ok.json()["changed"] >= 1
    assert client.patch(f"/api/transactions/{tid}", headers=H, json={"category": "Crypto"}).status_code == 422
    assert client.patch("/api/transactions/NOPE", headers=H, json={"category": "Health"}).status_code == 404
    csv = client.get("/api/transactions.csv", params={"session_id": "test-session-1"})
    assert csv.status_code == 200 and csv.text.startswith("txn_id,date")


def test_chat_routes(client):
    r = client.post("/api/chat", headers=H, json={"message": "How much did I spend in September?"}).json()
    assert r["agent"] == "Analysis Agent" and r["intent"] == "spending_analysis"
    assert client.post("/api/chat", headers=H, json={"message": ""}).status_code == 422
    hist = client.get("/api/history", headers=H).json()["messages"]
    assert hist[-2]["role"] == "user" and hist[-1]["role"] == "assistant"


def test_insights_cached(client):
    assert client.get("/api/insights", headers=H).json()["summary"] == "fake summary"


def test_prompts_and_search(client):
    p = client.get("/api/prompts").json()
    assert len(p["prompts"]) == 11 and len(p["criteria"]) == 7
    s = client.post("/api/search", headers=H, json={"query": "emergency fund", "collection": "finance_kb"}).json()
    assert s["results"][0]["source"] == "emergency_fund.md"
    t = client.post("/api/search", headers=H, json={"query": "medicines", "collection": "transactions"}).json()
    assert t["results"]


def _statement_without_october_income() -> bytes:
    """The sample bank CSV with every October credit removed: a month with spending but no income."""
    import csv, io
    rows = list(csv.reader(io.StringIO((SAMPLES / "bank_statement.csv").read_text())))
    keep = [r for r in rows if not (len(r) == 7 and r[0].endswith("/10/2026") and r[5])]
    out = io.StringIO()
    csv.writer(out).writerows(keep)
    return out.getvalue().encode()


def test_month_without_income_does_not_break_dashboard_or_insights(client):
    """Regression: a month with no income made savings rate NaN and every page returned 500."""
    h = {"X-Session-Id": "test-no-income"}
    r = client.post("/api/upload", headers=h, files={"file": ("no_oct_income.csv", _statement_without_october_income(), "text/csv")})
    assert r.status_code == 200
    d = client.get("/api/dashboard", headers=h)
    assert d.status_code == 200
    oct_row = next(m for m in d.json()["monthly"] if m["month"] == "2026-10")
    assert oct_row["income"] == 0 and oct_row["savings_rate_pct"] is None
    assert client.get("/api/insights", headers=h).status_code == 200
    assert client.get("/api/transactions", headers=h).status_code == 200


def test_profile_text_handles_month_without_income():
    import pandas as pd
    from spendsense.agents.parser_agent import parse_statement
    from spendsense.tools.alerts import detect_alerts, financial_profile, profile_text
    df = parse_statement(_statement_without_october_income(), "x.csv", allow_llm=False).df
    df["category"] = "Others"
    p = financial_profile(df)
    assert next(m for m in p["months"] if m["month"] == "2026-10")["savings_rate_pct"] is None
    text = profile_text(p)
    import re
    assert not re.search(r"\bnan\b", text, re.I) and "no income this month" in text
    assert any(a["id"] == "negative_month" and a["month"] == "2026-10" for a in detect_alerts(df))


def test_json_safe_response():
    from server import SafeJSONResponse
    body = SafeJSONResponse({"a": float("nan"), "b": [1.5, float("inf")], "c": {"d": -float("inf")}}).body
    assert body == b'{"a":null,"b":[1.5,null],"c":{"d":null}}'


def test_sample_and_reset(client):
    h = {"X-Session-Id": "test-session-2"}
    assert client.post("/api/sample", headers=h, params={"name": "../.env"}).status_code == 404
    assert client.post("/api/sample", headers=h).json()["parse"]["transactions"] == 336
    client.post("/api/reset", headers=h)
    assert not client.get("/api/health", headers=h).json()["has_data"]
