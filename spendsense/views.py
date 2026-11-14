"""JSON views for the web UI: dashboard figures, the transactions table and the prompt catalogue.

Pure pandas, no LLM, so the dashboard loads instantly after an upload.
"""

from __future__ import annotations

import math

import pandas as pd

from spendsense import config
from spendsense.tools.alerts import budget_split, detect_subscriptions, monthly_table, savings_levers


def _r(x) -> float:
    return round(float(x), 2)


def dashboard(df: pd.DataFrame, alerts: list[dict]) -> dict:
    debit = df[df["type"] == "DEBIT"]
    credit = df[df["type"] == "CREDIT"]
    mt = monthly_table(df)
    spent, income = debit["amount"].sum(), credit["amount"].sum()
    invested = debit[debit["category"].isin(config.SAVINGS_CATEGORIES)]["amount"].sum()
    cats = debit.groupby("category")["amount"].sum().sort_values(ascending=False)
    merch = debit.groupby("payee")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(10)
    weekly = debit.set_index("date")["amount"].resample("W-MON", label="left", closed="left").sum()
    modes = debit.groupby("mode")["amount"].sum().sort_values(ascending=False)
    return {
        "period": {"from": df["date"].min().strftime("%Y-%m-%d"), "to": df["date"].max().strftime("%Y-%m-%d"),
                   "months": len(mt)},
        "kpis": {
            "total_spent": _r(spent), "total_income": _r(income), "net": _r(income - spent),
            "invested": _r(invested),
            "savings_rate_pct": _r((income - spent + invested) / income * 100) if income else 0,
            "transactions": int(len(df)), "avg_monthly_spent": _r(mt["spent"].mean()),
            "alerts_high": sum(1 for a in alerts if a["severity"] == "high"),
        },
        "categories": [{"category": c, "total": _r(v), "share_pct": _r(v / spent * 100) if spent else 0}
                       for c, v in cats.items()],
        "monthly": [{"month": m, "income": _r(r["income"]), "spent": _r(r["spent"]),
                     "savings_rate_pct": None if math.isnan(r["savings_rate_pct"]) else _r(r["savings_rate_pct"])}
                    for m, r in mt.iterrows()],
        "merchants": [{"payee": p, "total": _r(r["sum"]), "count": int(r["count"])} for p, r in merch.iterrows()],
        "weekly": [{"week": d.strftime("%Y-%m-%d"), "spent": _r(v)} for d, v in weekly.items()],
        "modes": [{"mode": m, "total": _r(v)} for m, v in modes.items()],
        "budget_split": budget_split(df),
        "levers": savings_levers(df),
        "subscriptions": detect_subscriptions(df),
        "top_alerts": alerts[:4],
    }


def transactions_page(df: pd.DataFrame, q: str = "", category: str = "", month: str = "", type: str = "",
                      sort: str = "date", desc: bool = False, page: int = 1, page_size: int = 25) -> dict:
    s = df
    if q:
        hay = (s["payee"] + " " + s["description"] + " " + s["note"].fillna("")).str.lower()
        s = s[hay.str.contains(q.lower(), regex=False)]
    if category:
        s = s[s["category"] == category]
    if month:
        s = s[s["date"].dt.strftime("%Y-%m") == month]
    if type:
        s = s[s["type"] == type.upper()]
    if sort in {"date", "amount", "payee", "category"}:
        s = s.sort_values([sort, "txn_id"], ascending=not desc, kind="stable")
    page_size = max(5, min(int(page_size), 100))
    pages = max(1, math.ceil(len(s) / page_size))
    page = max(1, min(int(page), pages))
    chunk = s.iloc[(page - 1) * page_size: page * page_size]
    rows = [{"txn_id": r.txn_id, "date": r.date.strftime("%Y-%m-%d"), "payee": r.payee, "description": r.description,
             "mode": r.mode, "type": r.type, "amount": _r(r.amount), "category": r.category,
             "cat_source": r.cat_source, "cat_confidence": None if pd.isna(r.cat_confidence) else _r(r.cat_confidence),
             "cat_reason": r.cat_reason, "note": r.note or ""} for r in chunk.itertuples()]
    return {"rows": rows, "total": int(len(s)), "page": page, "pages": pages, "page_size": page_size,
            "sum_debit": _r(s.loc[s["type"] == "DEBIT", "amount"].sum()),
            "sum_credit": _r(s.loc[s["type"] == "CREDIT", "amount"].sum()),
            "months": sorted(df["date"].dt.strftime("%Y-%m").unique().tolist()),
            "categories": config.CATEGORIES}


def transactions_csv(df: pd.DataFrame) -> str:
    cols = ["txn_id", "date", "payee", "description", "mode", "type", "amount", "balance", "category", "cat_source"]
    out = df[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False)


ACCEPTANCE_CRITERIA = [
    {"n": 1, "name": "LLM integration", "where": "Groq API: gpt-oss-120b for agents, gpt-oss-20b for routing, "
     "planning and critique, each falling back to the other when rate-limited (spendsense/llm.py)."},
    {"n": 2, "name": "Interactive GUI", "where": "This web app: upload, dashboard, chat with reasoning trace, "
     "editable transactions table, insights."},
    {"n": 3, "name": "Prompt engineering", "where": "11 labelled system/user prompts below: few-shot, dynamic "
     "few-shot, Chain-of-Thought, ReAct, reflection, zero-shot, structured output."},
    {"n": 4, "name": "LangChain core", "where": "ChatPromptTemplate + FewShotChatMessagePromptTemplate; LCEL "
     "prompt | llm | parser chains, RunnableBranch routing, RunnableWithMessageHistory, with_fallbacks."},
    {"n": 5, "name": "Agentic RAG", "where": "The Analysis Agent chooses semantic_txn_search only for vague "
     "lookups; the Advisor plans whether to search the knowledge base and cites sources."},
    {"n": 6, "name": "Intent detection & routing", "where": "Few-shot router -> 4 intents -> Analysis Agent, "
     "Advisor Agent or general responder; shown on every chat answer."},
    {"n": 7, "name": "Vector database", "where": "ChromaDB: finance_kb (76 chunks), transactions (one per row), "
     "labelled_examples (categoriser memory); try a similarity search below."},
]
