"""Categorizer tests. Rules are checked against ground truth; the LLM tiers use fakes (no API calls)."""

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from spendsense.agents.categorizer_agent import (
    apply_user_correction, categorize, rule_category,
)
from spendsense.agents.parser_agent import parse_file
from spendsense.rag.vectorstore import ephemeral_client, get_store
from spendsense.schemas import CategoryAssignment, CategoryBatch, ReflectionBatch, ReflectionItem

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def parsed() -> pd.DataFrame:
    return parse_file(SAMPLES / "bank_statement.csv", allow_llm=False).df


@pytest.fixture(scope="module")
def truth() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "ground_truth.csv")


@pytest.fixture
def store():
    return get_store(f"test_{uuid4().hex[:8]}", client=ephemeral_client())


def test_rules_are_precise_on_sample(parsed, truth):
    """Whatever the rules label must be right; they are allowed to leave things for the LLM."""
    labelled = [(rule_category(r)[0], t) for (_, r), t in zip(parsed.iterrows(), truth["category"])]
    hits = [(c, t) for c, t in labelled if c]
    # ~57% on the sample: the rest are local shops and people, deliberately left to the LLM
    assert len(hits) / len(labelled) > 0.5, "rules should cover the well-known merchants"
    wrong = [(c, t) for c, t in hits if c != t]
    assert not wrong, wrong


@pytest.mark.parametrize("payee,note,desc,ttype,mode,expected", [
    ("SWIGGY", "", "UPI/DR/1/SWIGGY/x/y@ybl/Food", "DEBIT", "UPI", "Food & Dining"),
    ("SURAKSHA FINANCE LTD", "BIKE LOAN EMI", "ACH/DR/1/SURAKSHA FINANCE LTD/BIKE LOAN EMI", "DEBIT", "ACH", "EMI & Loans"),
    ("ACME", "SALARY OCT", "NEFT/CR/1/ACME/SALARY OCT", "CREDIT", "NEFT", "Salary & Income"),
    ("AMAZON", "Refund", "UPI/CR/1/AMAZON/x/a@apl/Refund", "CREDIT", "UPI", "Shopping"),
    ("ATM CASH", "", "ATM/CASH WDL/1/X", "DEBIT", "ATM", "Others"),
    ("RAHUL SHARMA", "Dinner split", "UPI/CR/1/RAHUL SHARMA/x/r@ok/Dinner split", "CREDIT", "UPI", None),
    ("SRI MURUGAN TEA STALL", "Tea", "UPI/DR/1/SRI MURUGAN TEA STALL/x/m@ok/Tea", "DEBIT", "UPI", None),
])
def test_rule_examples(payee, note, desc, ttype, mode, expected):
    row = {"payee": payee, "note": note, "description": desc, "type": ttype, "mode": mode}
    assert rule_category(row)[0] == expected


def test_llm_tier_groups_by_payee_and_reflects(parsed, store):
    seen_inputs = []

    def fake_batch(inputs):
        seen_inputs.extend(inputs)
        out = []
        for inp in inputs:
            assigns = []
            for line in inp["transactions"].splitlines():
                gid = line.split(" | ")[0]
                person = "RAHUL" in line or "PRIYA" in line or "KARTHIK" in line or "SNEHA" in line or "VIKRAM" in line
                conf = 0.4 if "RAMESH KUMAR" in line else 0.9
                assigns.append(CategoryAssignment(txn_id=gid, reasoning="fake",
                                                  category="Transfers to People" if person else "Others",
                                                  confidence=conf))
            out.append(CategoryBatch(assignments=assigns))
        return out

    def fake_reflect(inp):
        gid = inp["items"].split(" | ")[0]
        return ReflectionBatch(items=[ReflectionItem(txn_id=gid, original_category="Others",
                                                     critique="Note says RENT", final_category="Rent", changed=True)])

    res = categorize(parsed, store=store, llm_batch_fn=fake_batch, reflect_fn=fake_reflect)
    df = res.df
    assert df["category"].notna().all()
    # one LLM line per (payee, type) group, far fewer than the raw rows
    sent_lines = sum(len(i["transactions"].splitlines()) for i in seen_inputs)
    assert sent_lines == res.groups_sent_to_llm < (df["cat_source"] != "rule").sum()
    # retrieved examples were inserted into the prompt
    assert all("->" in i["examples"] for i in seen_inputs)
    # reflection changed the low-confidence RAMESH KUMAR group to Rent
    rent = df[df["payee"] == "RAMESH KUMAR"]
    assert set(rent["category"]) == {"Rent"} and set(rent["cat_source"]) == {"llm+reflection"}
    assert res.summary()["reflection_changed"] == 1


def test_upload_survives_llm_quota_exhausted(parsed, store):
    """Regression: when every Groq model was rate-limited the whole upload failed with a 500."""
    def quota_gone(inputs):
        raise RuntimeError("Error code: 429 - rate_limit_exceeded: tokens per day (TPD)")

    def reflect_never(_):
        raise AssertionError("reflection must be skipped when the LLM is down")

    res = categorize(parsed, store=store, llm_batch_fn=quota_gone, reflect_fn=reflect_never)
    df = res.df
    assert df["category"].notna().all()
    assert res.llm_error == "LLM quota used up (Groq rate limit)"
    assert set(df["cat_source"]) <= {"rule", "vector", "default"}
    assert (df["cat_source"] == "rule").sum() > 150                 # rules still did their part
    assert res.summary()["llm_error"]
    # seed examples like "MANI TEA STALL -> Others" should let similar local shops borrow a category
    vec = df[df["cat_source"] == "vector"]
    assert all(r.startswith("LLM quota used up") for r in vec["cat_reason"])


def test_one_failed_batch_keeps_the_others(parsed, store):
    def half(inputs):
        out = [RuntimeError("429 rate_limit")]
        for inp in inputs[1:]:
            out.append(CategoryBatch(assignments=[
                CategoryAssignment(txn_id=l.split(" | ")[0], reasoning="ok", category="Others", confidence=0.9)
                for l in inp["transactions"].splitlines()]))
        return out

    res = categorize(parsed, store=store, llm_batch_fn=half, reflect_fn=lambda _: ReflectionBatch(items=[]))
    srcs = set(res.df["cat_source"])
    assert "llm" in srcs and srcs & {"vector", "default"}


def test_user_correction_updates_payee_and_store(parsed, store):
    df = categorize(parsed, store=store, use_llm=False).df
    tid = df.loc[df["payee"] == "CULT FIT", "txn_id"].iloc[0]
    df2 = apply_user_correction(df, tid, "Health", store=store)
    assert set(df2.loc[df2["payee"] == "CULT FIT", "category"]) == {"Health"}
    hit = store.similarity_search("CULT FIT gym", k=1)[0]
    assert hit.metadata["category"] == "Health" and hit.metadata["source"] == "user"


def test_user_correction_rejects_unknown_category(parsed, store):
    df = categorize(parsed, store=store, use_llm=False).df
    with pytest.raises(ValueError):
        apply_user_correction(df, df["txn_id"].iloc[0], "Crypto", store=store)
