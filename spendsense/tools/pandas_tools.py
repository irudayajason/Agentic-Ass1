"""Data tools for the Analysis Agent.

Each analysis is a plain function over the categorised transactions DataFrame (easy to test),
and `build_tools()` wraps them as LangChain tools bound to the loaded statement. The LLM never
adds up numbers itself: every total, count and percentage it reports comes from here.
"""

from __future__ import annotations

import json
import re
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from spendsense.tools.calculator import CalcError, calculate

TxnType = Literal["DEBIT", "CREDIT"]
MAX_ROWS = 25


# =================================================================== helpers
def _r(x) -> float:
    return round(float(x), 2)


def resolve_month(df: pd.DataFrame, month: str | None) -> str | None:
    """'2026-09', 'September', 'sep 2026', '09/2026' -> '2026-09' (latest matching month in the data)."""
    if not month:
        return None
    m = month.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}", m):
        return m
    if mm := re.fullmatch(r"(\d{1,2})[/-](\d{4})", m):
        return f"{mm.group(2)}-{int(mm.group(1)):02d}"
    available = sorted(df["date"].dt.strftime("%Y-%m").unique())
    for fmt in ("%B %Y", "%b %Y", "%B", "%b"):
        try:
            ts = pd.to_datetime(m.title(), format=fmt)
        except (ValueError, TypeError):
            continue
        if "%Y" in fmt:
            return ts.strftime("%Y-%m")
        matches = [a for a in available if a.endswith(f"-{ts.month:02d}")]
        return matches[-1] if matches else f"{df['date'].max().year}-{ts.month:02d}"
    raise ValueError(f"Could not understand the month '{month}'. Use YYYY-MM, e.g. 2026-09.")


def _slice(df: pd.DataFrame, month=None, date_from=None, date_to=None, type=None, category=None,
           merchant=None) -> pd.DataFrame:
    out = df
    if month:
        out = out[out["date"].dt.strftime("%Y-%m") == resolve_month(df, month)]
    if date_from:
        out = out[out["date"] >= pd.Timestamp(date_from)]
    if date_to:
        out = out[out["date"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)]
    if type:
        out = out[out["type"] == type.upper()]
    if category:
        out = out[out["category"].str.lower() == category.lower()]
    if merchant:
        out = out[out["payee"].str.contains(re.escape(merchant), case=False, na=False)]
    return out


def _rows(df: pd.DataFrame, limit: int = MAX_ROWS) -> list[dict]:
    cols = ["txn_id", "date", "payee", "type", "amount", "category", "mode", "note"]
    return [{**r, "date": r["date"].strftime("%Y-%m-%d"), "amount": _r(r["amount"])}
            for r in df[cols].head(limit).to_dict("records")]


# =================================================================== analyses
def statement_overview(df: pd.DataFrame) -> dict:
    by_month = []
    for month, g in df.groupby(df["date"].dt.strftime("%Y-%m")):
        inc, spent = g.loc[g["type"] == "CREDIT", "amount"].sum(), g.loc[g["type"] == "DEBIT", "amount"].sum()
        by_month.append({"month": month, "income": _r(inc), "spent": _r(spent), "net": _r(inc - spent),
                         "savings_rate_pct": _r((inc - spent) / inc * 100) if inc else None, "transactions": len(g)})
    debit, credit = df.loc[df["type"] == "DEBIT", "amount"].sum(), df.loc[df["type"] == "CREDIT", "amount"].sum()
    return {"date_from": df["date"].min().strftime("%Y-%m-%d"), "date_to": df["date"].max().strftime("%Y-%m-%d"),
            "transactions": len(df), "total_spent": _r(debit), "total_income": _r(credit), "net": _r(credit - debit),
            "months": by_month, "categories": sorted(df["category"].dropna().unique().tolist())}


def category_totals(df: pd.DataFrame, month=None, date_from=None, date_to=None, type="DEBIT") -> dict:
    s = _slice(df, month, date_from, date_to, type)
    total = s["amount"].sum()
    g = s.groupby("category")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    return {"filters": {"month": resolve_month(df, month), "date_from": date_from, "date_to": date_to, "type": type},
            "grand_total": _r(total), "transactions": len(s),
            "categories": [{"category": c, "total": _r(r["sum"]), "count": int(r["count"]),
                            "share_pct": _r(r["sum"] / total * 100) if total else 0} for c, r in g.iterrows()]}


def monthly_trend(df: pd.DataFrame, category=None, merchant=None, type="DEBIT") -> dict:
    s = _slice(df, type=type, category=category, merchant=merchant)
    months = sorted(df["date"].dt.strftime("%Y-%m").unique())
    g = s.groupby(s["date"].dt.strftime("%Y-%m"))["amount"].agg(["sum", "count"])
    out, prev = [], None
    for m in months:
        tot = float(g.loc[m, "sum"]) if m in g.index else 0.0
        cnt = int(g.loc[m, "count"]) if m in g.index else 0
        out.append({"month": m, "total": _r(tot), "count": cnt,
                    "change_pct_vs_prev": _r((tot - prev) / prev * 100) if prev else None})
        prev = tot
    return {"filters": {"category": category, "merchant": merchant, "type": type}, "months": out,
            "average_per_month": _r(sum(o["total"] for o in out) / len(out)) if out else 0}


def top_merchants(df: pd.DataFrame, n=10, month=None, category=None, type="DEBIT") -> dict:
    n = max(1, min(int(n), 30))
    s = _slice(df, month, type=type, category=category)
    g = s.groupby("payee")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(n)
    return {"filters": {"month": resolve_month(df, month), "category": category, "type": type},
            "merchants": [{"payee": p, "total": _r(r["sum"]), "count": int(r["count"])} for p, r in g.iterrows()]}


def filter_transactions(df: pd.DataFrame, merchant=None, category=None, month=None, date_from=None, date_to=None,
                        min_amount=None, max_amount=None, type=None, text=None, sort="date", limit=MAX_ROWS) -> dict:
    limit = max(1, min(int(limit), MAX_ROWS))
    s = _slice(df, month, date_from, date_to, type, category, merchant)
    if min_amount is not None:
        s = s[s["amount"] >= float(min_amount)]
    if max_amount is not None:
        s = s[s["amount"] <= float(max_amount)]
    if text:
        hay = (s["description"] + " " + s["note"].fillna("")).str.lower()
        s = s[hay.str.contains(re.escape(text.lower()), na=False)]
    s = s.sort_values("amount", ascending=False) if sort == "amount" else s.sort_values("date")
    debit, credit = s.loc[s["type"] == "DEBIT", "amount"].sum(), s.loc[s["type"] == "CREDIT", "amount"].sum()
    return {"matched": len(s), "total_debit": _r(debit), "total_credit": _r(credit),
            "showing": min(len(s), limit), "transactions": _rows(s, limit)}


def compare_periods(df: pd.DataFrame, period_a: str, period_b: str, category=None, merchant=None,
                    type="DEBIT") -> dict:
    a_m, b_m = resolve_month(df, period_a), resolve_month(df, period_b)
    a = _slice(df, a_m, type=type, category=category, merchant=merchant)
    b = _slice(df, b_m, type=type, category=category, merchant=merchant)
    ta, tb = a["amount"].sum(), b["amount"].sum()
    out = {"period_a": a_m, "period_b": b_m, "filters": {"category": category, "merchant": merchant, "type": type},
           "total_a": _r(ta), "total_b": _r(tb), "count_a": len(a), "count_b": len(b),
           "difference_b_minus_a": _r(tb - ta), "change_pct": _r((tb - ta) / ta * 100) if ta else None}
    if not category and not merchant:
        ca, cb = a.groupby("category")["amount"].sum(), b.groupby("category")["amount"].sum()
        cats = sorted(set(ca.index) | set(cb.index))
        diffs = [{"category": c, "a": _r(ca.get(c, 0)), "b": _r(cb.get(c, 0)), "diff": _r(cb.get(c, 0) - ca.get(c, 0))}
                 for c in cats]
        out["biggest_changes"] = sorted(diffs, key=lambda d: -abs(d["diff"]))[:6]
    return out


# =================================================================== LangChain tool wrappers
class NoArgs(BaseModel):
    pass


class _Period(BaseModel):
    month: Optional[str] = Field(None, description="Month as YYYY-MM, e.g. 2026-09")
    date_from: Optional[str] = Field(None, description="Start date YYYY-MM-DD (inclusive)")
    date_to: Optional[str] = Field(None, description="End date YYYY-MM-DD (inclusive)")


class CategoryTotalsArgs(_Period):
    type: TxnType = Field("DEBIT", description="DEBIT = spending, CREDIT = money received")


class TrendArgs(BaseModel):
    category: Optional[str] = Field(None, description="Exact category name to limit to")
    merchant: Optional[str] = Field(None, description="Part of the payee name, e.g. 'swiggy'")
    type: TxnType = "DEBIT"


class TopMerchantsArgs(BaseModel):
    n: int = Field(10, description="How many payees (max 30)")
    month: Optional[str] = Field(None, description="Month as YYYY-MM")
    category: Optional[str] = None
    type: TxnType = "DEBIT"


class FilterArgs(_Period):
    merchant: Optional[str] = Field(None, description="Part of the payee name, case-insensitive")
    category: Optional[str] = Field(None, description="Exact category name")
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    type: Optional[TxnType] = Field(None, description="Leave empty for both debits and credits")
    text: Optional[str] = Field(None, description="Word that must appear in the description or UPI note")
    sort: Literal["date", "amount"] = "date"
    limit: int = Field(15, description=f"Rows to return (max {MAX_ROWS}); totals always cover all matches")


class CompareArgs(BaseModel):
    period_a: str = Field(description="Earlier month, YYYY-MM")
    period_b: str = Field(description="Later month, YYYY-MM")
    category: Optional[str] = None
    merchant: Optional[str] = None
    type: TxnType = "DEBIT"


class CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic only, e.g. '(4235.5 - 3120) / 3120 * 100'")


class SearchArgs(BaseModel):
    query: str = Field(description="What the transaction was about, in plain words")
    k: int = Field(8, description="Number of results (max 20)")
    month: Optional[str] = Field(None, description="Month as YYYY-MM")
    type: Optional[TxnType] = None
    category: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _safe(fn):
    """Return errors to the agent as text so it can correct its call instead of crashing."""
    def wrapper(**kwargs):
        try:
            return _dump(fn(**kwargs))
        except (ValueError, KeyError, CalcError) as exc:
            return _dump({"error": str(exc)})
    return wrapper


def build_tools(get_df, chroma_client=None) -> list[StructuredTool]:
    """Tools bound to the loaded statement. `get_df` returns the current categorised DataFrame."""

    def _calc(expression: str) -> dict:
        return {"expression": expression, "result": calculate(expression)}

    def _search(query, k=8, month=None, type=None, category=None, min_amount=None, max_amount=None) -> dict:
        from spendsense.rag.retrieval import search_transactions
        hits = search_transactions(query, k=max(1, min(int(k), 20)), type=type, category=category, month=resolve_month(get_df(), month),
                                   min_amount=min_amount, max_amount=max_amount, client=chroma_client)
        return {"results": [{k_: h[k_] for k_ in ("txn_id", "date", "payee", "type", "amount", "category", "score")}
                            for h in hits]}

    specs = [
        ("get_statement_overview", "Date range, totals and income/spending/savings rate for every month. "
         "Call this first when you need the big picture or the available months.",
         lambda: statement_overview(get_df()), None),
        ("get_category_totals", "Total spent (or received) per category, with counts and % share, "
         "for the whole statement, a month or a date range.",
         lambda **a: category_totals(get_df(), **a), CategoryTotalsArgs),
        ("get_monthly_trend", "Month-by-month totals and % change, optionally for one category or merchant.",
         lambda **a: monthly_trend(get_df(), **a), TrendArgs),
        ("get_top_merchants", "People and merchants ranked by total amount, optionally for a month or category. "
         "type='DEBIT' = who you paid most, type='CREDIT' = who paid you most.",
         lambda **a: top_merchants(get_df(), **a), TopMerchantsArgs),
        ("filter_transactions", "List transactions matching exact filters (merchant, category, month, dates, "
         "sort='amount' lists the largest first, "
         "amount range, type, a word in the description). Returns matched count and totals too.",
         lambda **a: filter_transactions(get_df(), **a), FilterArgs),
        ("compare_periods", "Compare spending between two months, overall or for one category/merchant, "
         "including the categories that changed most.",
         lambda **a: compare_periods(get_df(), **a), CompareArgs),
        ("calculator", "Exact arithmetic on numbers from other tools: sums, differences, averages, percentages.",
         _calc, CalcArgs),
        ("semantic_txn_search", "Search transactions by MEANING using the vector database (ChromaDB). Use only "
         "when the user describes a transaction vaguely and exact filters would miss it, e.g. "
         "'that medical store payment' or 'the gadget I bought'.",
         _search, SearchArgs),
    ]
    return [StructuredTool.from_function(func=_safe(fn), name=name, description=desc, args_schema=schema or NoArgs)
            for name, desc, fn, schema in specs]
