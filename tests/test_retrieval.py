"""Retrieval tests on an in-memory ChromaDB (local embeddings, no LLM)."""

from pathlib import Path

import pytest

from spendsense.agents.categorizer_agent import categorize
from spendsense.agents.parser_agent import parse_file
from spendsense.rag import retrieval
from spendsense.rag.vectorstore import ephemeral_client

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def client():
    c = ephemeral_client()
    retrieval.ensure_kb_indexed(client=c)
    df = parse_file(SAMPLES / "bank_statement.csv", allow_llm=False).df
    df = categorize(df, use_llm=False).df   # rule labels are enough here
    retrieval.index_transactions(df, client=c)
    return c


def test_kb_chunks_have_titles_and_sections():
    docs = retrieval.load_kb_documents()
    assert len({d.metadata["source"] for d in docs}) == 20
    assert all(d.metadata["section"] and " - " in d.page_content.split("\n")[0] for d in docs)


@pytest.mark.parametrize("query,expected", [
    ("how many months of expenses should I keep aside for emergencies", "emergency_fund.md"),
    ("I order Swiggy too often, how do I cut down", "food_delivery_savings.md"),
    ("I spend too much on Swiggy and Zomato", "food_delivery_savings.md"),
    ("Is it OK that my EMIs are 35% of my salary?", "emi_debt_management.md"),
    ("How do I stop overspending in the Diwali sale?", "shopping_impulse_control.md"),
    ("netflix spotify autopay I forgot about", "subscriptions_audit.md"),
    ("too many tiny chai and parking payments", "small_upi_payments.md"),
    ("how should I split my salary between needs and wants", "budgeting_50_30_20.md"),
    ("someone asked for my PIN to receive a refund", "upi_safety.md"),
])
def test_kb_search_finds_right_note(client, query, expected):
    hits = retrieval.search_kb(query, client=client)
    assert hits and hits[0]["source"] == expected, [h["source"] for h in hits]


@pytest.mark.parametrize("query", ["Who won the cricket match yesterday?", "what is the capital of France",
                                   "write me a poem about cats"])
def test_kb_search_returns_nothing_for_off_topic(client, query):
    assert retrieval.search_kb(query, client=client) == []


def test_txn_search_by_meaning(client):
    hits = retrieval.search_transactions("medical store medicines", k=5, client=client)
    assert {h["payee"] for h in hits} & {"APOLLO PHARMACY", "MEDPLUS"}


def test_txn_search_with_filters(client):
    hits = retrieval.search_transactions("food order", k=20, type="DEBIT", month="2026-09",
                                         min_amount=300, client=client)
    assert hits and all(h["month"] == "2026-09" and h["amount"] >= 300 and h["type"] == "DEBIT" for h in hits)


def test_txn_search_date_range(client):
    hits = retrieval.search_transactions("payment", k=50, date_from="2026-08-10", date_to="2026-08-12", client=client)
    assert hits and all("2026-08-10" <= h["date"] <= "2026-08-12" for h in hits)


def test_collection_stats(client):
    stats = retrieval.collection_stats(client=client)
    assert stats["finance_kb"] > 40 and stats["transactions"] == 336
