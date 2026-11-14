"""Tools and the ReAct loop, offline. Tool numbers are checked against direct pandas on ground truth."""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from spendsense.agents.analysis_agent import AnalysisAgent
from spendsense.agents.parser_agent import parse_file
from spendsense.tools import pandas_tools as pt
from spendsense.tools.calculator import CalcError, calculate

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = parse_file(SAMPLES / "bank_statement.csv", allow_llm=False).df
    d["category"] = pd.read_csv(SAMPLES / "ground_truth.csv")["category"].values
    return d


# ------------------------------------------------------------------ calculator
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14), ("(4235.5 - 3120) / 3120 * 100", 35.7532), ("₹1,23,456 / 12", 10288.0),
    ("18% * 4350", 783.0), ("round(1234.5678, 2)", 1234.57), ("max(1, 9, 3)", 9), ("sum([1, 2, 3])", 6),
])
def test_calculator(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "1/0", "2**1000", "a + 1"])
def test_calculator_rejects_unsafe(expr):
    with pytest.raises(CalcError):
        calculate(expr)


# ------------------------------------------------------------------ analyses
def test_resolve_month(df):
    assert pt.resolve_month(df, "September") == "2026-09"
    assert pt.resolve_month(df, "sep 2026") == "2026-09"
    assert pt.resolve_month(df, "2026-10") == "2026-10"
    assert pt.resolve_month(df, "09/2026") == "2026-09"
    with pytest.raises(ValueError):
        pt.resolve_month(df, "someday")


def test_overview_matches_pandas(df):
    o = pt.statement_overview(df)
    sep = next(m for m in o["months"] if m["month"] == "2026-09")
    s = df[df["date"].dt.month == 9]
    assert sep["spent"] == pytest.approx(s.loc[s["type"] == "DEBIT", "amount"].sum())
    assert o["transactions"] == len(df)


def test_category_totals_sum_to_total(df):
    r = pt.category_totals(df, month="2026-09")
    assert sum(c["total"] for c in r["categories"]) == pytest.approx(r["grand_total"])
    food = next(c for c in r["categories"] if c["category"] == "Food & Dining")
    exp = df[(df["date"].dt.month == 9) & (df["category"] == "Food & Dining") & (df["type"] == "DEBIT")]["amount"].sum()
    assert food["total"] == pytest.approx(exp)


def test_monthly_trend_and_change(df):
    r = pt.monthly_trend(df, category="Shopping")
    assert [m["month"] for m in r["months"]] == ["2026-07", "2026-08", "2026-09", "2026-10"]
    sep = r["months"][2]
    assert sep["change_pct_vs_prev"] > 100   # the planted festival-sale spike


def test_filter_merchant_and_text(df):
    r = pt.filter_transactions(df, merchant="swiggy", type="DEBIT")
    assert r["matched"] == ((df["payee"] == "SWIGGY") & (df["type"] == "DEBIT")).sum()
    rent = pt.filter_transactions(df, text="rent")
    assert rent["matched"] == 4 and rent["total_debit"] == 88_000


def test_compare_periods(df):
    r = pt.compare_periods(df, "August", "September")
    assert r["period_a"] == "2026-08" and r["difference_b_minus_a"] == pytest.approx(r["total_b"] - r["total_a"])
    assert r["biggest_changes"][0]["diff"] > 0


def test_tools_return_errors_as_text(df):
    tools = {t.name: t for t in pt.build_tools(lambda: df)}
    out = json.loads(tools["get_category_totals"].invoke({"month": "someday"}))
    assert "error" in out
    assert json.loads(tools["calculator"].invoke({"expression": "1/0"}))["error"]


# ------------------------------------------------------------------ ReAct loop with a scripted model
class ScriptedLLM:
    """Stands in for ChatGroq: returns pre-written AIMessages in order."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def bind_tools(self, tools):
        def respond(prompt_value):
            self.seen.append(prompt_value.to_messages())
            return self.replies.pop(0)
        return RunnableLambda(respond)


def test_react_loop_runs_tools_and_answers(df):
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "get_category_totals", "args": {"month": "2026-09"}, "id": "c1"}],
                  additional_kwargs={"reasoning_content": "Need September category totals."}),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "5 + 5"}, "id": "c2"}]),
        AIMessage(content="You spent the most on Rent.", additional_kwargs={"reasoning_content": "Done."}),
    ])
    agent = AnalysisAgent(lambda: df, llm=llm)
    res = agent.run("Where did my money go in September?")
    assert res.answer == "You spent the most on Rent."
    assert res.tools_used == ["get_category_totals", "calculator"]
    assert res.steps[0].thought == "Need September category totals."
    assert not res.used_retrieval
    # the observation from step 1 was fed back to the model on step 2 (scratchpad)
    assert any(m.type == "tool" and "grand_total" in m.content for m in llm.seen[1])


def test_empty_final_answer_is_retried(df):
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "filter_transactions", "args": {"sort": "amount", "limit": 1}, "id": "c1"}]),
        AIMessage(content=""),                                   # model ends with no text
        AIMessage(content="Your largest transaction was ₹78,500."),
    ])
    res = AnalysisAgent(lambda: df, llm=llm).run("Which transaction is the highest amount?")
    assert res.answer == "Your largest transaction was ₹78,500." and res.empty_answer_retried
    assert "did not write an answer" in llm.seen[-1][-1].content


def test_never_returns_an_empty_answer(df):
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "get_statement_overview", "args": {}, "id": "c1"}]),
        AIMessage(content=""), AIMessage(content=""),
    ])
    res = AnalysisAgent(lambda: df, llm=llm).run("?")
    assert res.answer and "How I got this answer" in res.answer


def test_largest_transactions_tool(df):
    r = pt.filter_transactions(df, sort="amount", type="DEBIT", limit=1)
    assert r["transactions"][0]["amount"] == df.loc[df["type"] == "DEBIT", "amount"].max()


def test_react_loop_stops_after_max_steps(df):
    loop = [AIMessage(content="", tool_calls=[{"name": "get_statement_overview", "args": {}, "id": f"c{i}"}])
            for i in range(3)]
    llm = ScriptedLLM(loop + [AIMessage(content="Partial answer.")])
    res = AnalysisAgent(lambda: df, llm=llm, max_steps=3).run("?")
    assert res.stopped_early and res.answer == "Partial answer." and len(res.steps) == 3
