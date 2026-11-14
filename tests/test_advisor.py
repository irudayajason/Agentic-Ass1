"""Alert engine on the sample statement, and the Advisor pipeline with fake models (no API calls)."""

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from spendsense.agents.advisor_agent import AdvisorAgent
from spendsense.agents.parser_agent import parse_file
from spendsense.formatting import inr
from spendsense.schemas import AdviceCritique, RetrievalPlan
from spendsense.tools.alerts import budget_split, detect_alerts, detect_subscriptions, financial_profile, profile_text

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = parse_file(SAMPLES / "bank_statement.csv", allow_llm=False).df
    d["category"] = pd.read_csv(SAMPLES / "ground_truth.csv")["category"].values
    return d


@pytest.mark.parametrize("x,dec,out", [(123456.5, 0, "₹1,23,456"), (1234567.891, 2, "₹12,34,567.89"),
                                       (999, 0, "₹999"), (-29515, 0, "-₹29,515"), (0, 0, "₹0")])
def test_inr(x, dec, out):
    assert inr(x, dec) == out


# ------------------------------------------------------------------ alerts
def test_planted_september_problems_are_found(df):
    alerts = detect_alerts(df)
    ids = [(a["id"], a.get("month"), a.get("category")) for a in alerts]
    assert ("negative_month", "2026-09", None) in ids
    assert ("category_spike", "2026-09", "Shopping") in ids
    assert ("food_orders", "2026-09", "Food & Dining") in ids
    assert any(a["id"] == "small_upi" and a["month"] == "2026-09" for a in alerts)
    assert alerts[0]["severity"] == "high"          # sorted most severe first
    assert alerts[-1]["severity"] == "positive"


def test_positive_habits(df):
    ids = {a["id"] for a in detect_alerts(df)}
    assert {"positive_investing", "positive_emi"} <= ids


def test_subscriptions_flags_new_one(df):
    subs = {s["payee"]: s for s in detect_subscriptions(df)}
    assert {"NETFLIX", "SPOTIFY INDIA", "GOOGLE YOUTUBE PREMIUM"} <= subs.keys()
    assert subs["GOOGLE YOUTUBE PREMIUM"]["new"] and not subs["NETFLIX"]["new"]


def test_budget_split_adds_up(df):
    b = budget_split(df)
    assert b["needs"] + b["wants"] + b["saved"] == pytest.approx(b["income"], abs=1)
    assert b["needs_pct"] + b["wants_pct"] + b["savings_pct"] == pytest.approx(100, abs=0.05)


def test_profile_text_is_compact_and_indian_formatted(df):
    text = profile_text(financial_profile(df))
    assert len(text) / 4 < 800                        # rough token count, keeps prompts cheap
    assert "₹1,11,177" in text and "₹111,177" not in text


def test_savings_levers_use_own_best_month(df):
    from spendsense.tools.alerts import savings_levers
    lv = {l["category"]: l for l in savings_levers(df)}
    shop = df[(df["category"] == "Shopping") & (df["type"] == "DEBIT")].groupby(df["date"].dt.month)["amount"].sum()
    assert lv["Shopping"]["monthly_saving_if_like_best_month"] == pytest.approx(shop.mean() - shop.min(), abs=0.01)
    assert list(lv)[0] == "Shopping"     # biggest lever first
    assert "Rent" not in lv              # only 'wants' categories


def test_advisor_keeps_draft_if_critique_fails(df):
    class Broken(FakeModel):
        def with_structured_output(self, schema, method=None):
            if schema is AdviceCritique:
                return RunnableLambda(lambda _: (_ for _ in ()).throw(ValueError("json_validate_failed")))
            return super().with_structured_output(schema, method)
    plan = [RetrievalPlan(need_retrieval=False, reason="-")]
    fast, main = Broken(structured={RetrievalPlan: plan}), Broken(texts=["Draft answer."])
    agent, _ = _agent(df, main, fast, [])
    res = agent.run("Am I overspending?")
    assert res.answer == "Draft answer." and res.critique.get("skipped") and not res.revised


def test_ungrounded_numbers():
    from spendsense.agents.advisor_agent import ungrounded_numbers
    ref = "Shopping saves ₹8,174/month. Savings rate 25.0%. Wants ₹30,741."
    assert ungrounded_numbers("Save ₹8,174 and keep 25%.", ref) == []
    assert ungrounded_numbers("Wants are about ₹31 k.", ref) == []
    assert ungrounded_numbers("Your rate is -8.4% and you'd save ₹1,250.", ref) == ["₹1,250", "-8.4%"]


def test_revision_that_invents_numbers_is_rejected(df):
    fast = FakeModel(structured={
        RetrievalPlan: [RetrievalPlan(need_retrieval=False, reason="-")],
        AdviceCritique: [AdviceCritique(grounded_in_numbers=False, cites_sources=True, actionable=True,
                                        issues=["unclear"], score=2)],
    })
    main = FakeModel(texts=["September spending was ₹1,11,177.", "Save ₹1,234 and ₹5,678 at 2.6%."])
    agent, _ = _agent(df, main, fast, [])
    res = agent.run("Am I overspending?")
    assert res.answer == "September spending was ₹1,11,177." and not res.revised
    assert res.grounding["revision_rejected"] and res.grounding["draft_ungrounded"] == []


def test_insights_work_without_llm(df):
    class Down(FakeModel):
        def _reply(self, prompt_value):
            raise RuntimeError("429 rate_limit_exceeded")
    agent, _ = _agent(df, Down(), Down(), [])
    ins = agent.insights()
    assert "quota is used up" in ins["summary"] and ins["alerts"] and ins["months"]


def test_no_alert_for_small_changes(df):
    for a in detect_alerts(df):
        if a["id"] == "category_spike":
            assert a["metrics"]["current"] - a["metrics"]["previous"] >= 2000


# ------------------------------------------------------------------ advisor pipeline with fakes
class FakeModel(RunnableLambda):
    """Text replies come from `texts`; structured replies come from `structured[schema]`."""

    def __init__(self, texts=(), structured=None):
        self.texts, self.structured, self.calls = list(texts), structured or {}, []
        super().__init__(self._reply)

    def _reply(self, prompt_value):
        self.calls.append(prompt_value.to_messages())
        return AIMessage(self.texts.pop(0))

    def with_structured_output(self, schema, method=None):
        def respond(prompt_value):
            self.calls.append(prompt_value.to_messages())
            return self.structured[schema].pop(0)
        return RunnableLambda(respond)


def _agent(df, main, fast, hits):
    searched = []
    def search(q, k):
        searched.append(q)
        return hits
    return AdvisorAgent(lambda: df, main_llm=main, fast_llm=fast, search_fn=search), searched


KB_HIT = {"source": "food_delivery_savings.md", "title": "Cutting Food Delivery", "section": "Practical ways",
          "text": "Food delivery - Practical ways\nSet a monthly cap.", "score": 0.7}


def test_advisor_retrieves_critiques_and_revises(df):
    fast = FakeModel(structured={
        RetrievalPlan: [RetrievalPlan(need_retrieval=True, reason="needs tips", queries=["cut food delivery"])],
        AdviceCritique: [AdviceCritique(grounded_in_numbers=True, cites_sources=False, actionable=False,
                                        issues=["No rupee amounts"], score=2)],
    })
    main = FakeModel(texts=["Draft: order less.", "Final: cap shopping to save ₹8,174 [source: food_delivery_savings.md]"])
    agent, searched = _agent(df, main, fast, [KB_HIT])
    res = agent.run("How do I spend less on food?", entities={"category": "Food & Dining"})
    assert searched == ["cut food delivery"]
    assert res.used_retrieval and res.sources == ["food_delivery_savings.md"]
    assert res.revised and res.answer.startswith("Final:") and res.draft == "Draft: order less."
    # the draft prompt contained the figures, the alerts and the retrieved guidance
    draft_prompt = main.calls[0][-1].content
    assert "₹1,11,177" in draft_prompt and "Shopping up" in draft_prompt and "[source: food_delivery_savings.md]" in draft_prompt
    # the alert about the asked-about category was moved to the top
    assert res.alerts[0]["category"] == "Food & Dining"


def test_advisor_skips_retrieval_and_revision_when_not_needed(df):
    fast = FakeModel(structured={
        RetrievalPlan: [RetrievalPlan(need_retrieval=False, reason="only needs the user's numbers")],
        AdviceCritique: [AdviceCritique(grounded_in_numbers=True, cites_sources=True, actionable=True, score=5)],
    })
    main = FakeModel(texts=["Your savings rate in July was 25.0%."])
    agent, searched = _agent(df, main, fast, [KB_HIT])
    res = agent.run("What was my savings rate in July?")
    assert searched == [] and not res.used_retrieval
    assert not res.revised and res.answer == "Your savings rate in July was 25.0%."
    assert "No guidance retrieved" in main.calls[0][-1].content
