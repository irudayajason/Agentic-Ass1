"""Rule-based alert engine and financial profile (pure pandas, no LLM).

The Advisor Agent never decides from intuition that someone is overspending: these rules
find the problems and measure them in rupees, and the LLM explains them and suggests fixes.

Alerts produced:
  negative_month      spent more than earned in a month
  low_savings         savings rate (counting investments as saved) below the target
  category_spike      a category up more than MOM_SPIKE_PCT vs the previous month
  budget_split        needs / wants / savings vs the 50/30/20 rule
  small_upi           lots of small UPI payments in a month
  food_orders         many food & dining orders in a month
  subscriptions       recurring same-amount payments (AutoPay), and new ones
  emi_ratio           EMIs as a share of income
  cash_share          ATM cash as a share of spending
  positive_*          good habits worth keeping (e.g. regular SIP)
"""

from __future__ import annotations

import pandas as pd

from spendsense import config as C
from spendsense.formatting import inr

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "positive": 3}


def _r(x) -> float:
    return round(float(x), 2)


def _months(df: pd.DataFrame) -> pd.Series:
    return df["date"].dt.strftime("%Y-%m")


def _alert(id, severity, title, detail, impact=0.0, month=None, category=None, **metrics) -> dict:
    return {"id": id, "severity": severity, "title": title, "detail": detail, "monthly_impact": _r(impact),
            "month": month, "category": category, "metrics": metrics}


# =================================================================== profile
def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(month=_months(df))
    inc = d[d["type"] == "CREDIT"].groupby("month")["amount"].sum()
    spent = d[d["type"] == "DEBIT"].groupby("month")["amount"].sum()
    invest = d[(d["type"] == "DEBIT") & d["category"].isin(C.SAVINGS_CATEGORIES)].groupby("month")["amount"].sum()
    t = pd.DataFrame({"income": inc, "spent": spent, "invested": invest}).fillna(0.0).sort_index()
    t["net"] = t["income"] - t["spent"]
    # money kept = what's left over + what went into investments
    t["savings_rate_pct"] = ((t["net"] + t["invested"]) / t["income"].where(t["income"] > 0) * 100).round(1)
    return t


def budget_split(df: pd.DataFrame) -> dict:
    """Needs / wants / savings as % of income over the whole statement (50/30/20 check)."""
    debit = df[df["type"] == "DEBIT"]
    income = df.loc[df["type"] == "CREDIT", "amount"].sum()
    needs = debit[debit["category"].isin(C.NEEDS_CATEGORIES)]["amount"].sum()
    wants = debit[debit["category"].isin(C.WANTS_CATEGORIES)]["amount"].sum()
    invested = debit[debit["category"].isin(C.SAVINGS_CATEGORIES)]["amount"].sum()
    saved = income - needs - wants          # leftover + investments
    pct = lambda x: _r(x / income * 100) if income else 0.0  # noqa: E731
    return {"income": _r(income), "needs": _r(needs), "wants": _r(wants), "invested": _r(invested),
            "saved": _r(saved), "needs_pct": pct(needs), "wants_pct": pct(wants), "savings_pct": pct(saved),
            "target": C.BUDGET_SPLIT}


def detect_subscriptions(df: pd.DataFrame) -> list[dict]:
    """Same payee charging (almost) the same amount in 3+ different months, or an AutoPay mandate."""
    d = df[df["type"] == "DEBIT"].assign(month=_months(df))
    subs = []
    for payee, g in d.groupby("payee"):
        months = g["month"].nunique()
        autopay = g["note"].fillna("").str.contains("autopay|mandate", case=False).any() or \
            g["description"].str.contains("autopay|mandate|NETFLIX|SPOTIFY|YOUTUBE", case=False).any()
        same_amount = g["amount"].round(0).nunique() == 1
        if (months >= 3 and same_amount and g["amount"].iloc[0] < 5000) or (autopay and months >= 2):
            subs.append({"payee": payee, "amount": _r(g["amount"].median()), "months": months,
                         "first_month": g["month"].min(), "category": g["category"].mode().iloc[0]})
    first_month = _months(df).min()
    for s in subs:
        s["new"] = s["first_month"] > first_month
    return sorted(subs, key=lambda s: -s["amount"])


def bucket_averages(df: pd.DataFrame) -> dict:
    """Average monthly needs / wants / savings in rupees, with the 50/30/20 targets in rupees."""
    b = budget_split(df)
    n = _months(df).nunique() or 1
    inc = b["income"] / n
    out = {"income": _r(inc)}
    for k, key in (("needs", "needs"), ("wants", "wants"), ("savings", "saved")):
        actual, target = b[key] / n, inc * C.BUDGET_SPLIT[k] / 100
        out[k] = {"actual": _r(actual), "target": _r(target), "gap": _r(actual - target)}
    return out


def savings_levers(df: pd.DataFrame) -> list[dict]:
    """For each 'wants' category: what bringing every month down to your own best month would save.
    Gives the Advisor ready-made, realistic rupee figures instead of letting the LLM invent them."""
    debit = df[df["type"] == "DEBIT"].assign(month=_months(df))
    months = sorted(debit["month"].unique())
    levers = []
    for cat in sorted(C.WANTS_CATEGORIES):
        per = debit[debit["category"] == cat].groupby("month")["amount"].sum().reindex(months, fill_value=0)
        if per.sum() == 0:
            continue
        best_month, best = per.idxmin(), per.min()
        avg = per.mean()
        saving = avg - best
        if saving >= 500:
            levers.append({"category": cat, "monthly_average": _r(avg), "best_month": best_month,
                           "best_month_amount": _r(best), "monthly_saving_if_like_best_month": _r(saving)})
    return sorted(levers, key=lambda l: -l["monthly_saving_if_like_best_month"])


def financial_profile(df: pd.DataFrame) -> dict:
    mt = monthly_table(df)
    debit = df[df["type"] == "DEBIT"].assign(month=_months(df))
    cat_month = debit.pivot_table(index="category", columns="month", values="amount", aggfunc="sum", fill_value=0)
    latest = mt.index.max()
    top_merch = debit.groupby("payee")["amount"].sum().sort_values(ascending=False).head(8)
    return {
        "period": f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        # a month with no income has no savings rate: None (JSON null), never NaN
        "months": [{"month": m, **{k: (None if pd.isna(v) else _r(v)) for k, v in r.items()}} for m, r in mt.iterrows()],
        "average_monthly_income": _r(mt["income"].mean()),
        "average_monthly_spent": _r(mt["spent"].mean()),
        "latest_month": latest,
        "category_by_month": {c: {m: _r(v) for m, v in row.items()} for c, row in cat_month.iterrows()},
        "category_monthly_average": {c: _r(v) for c, v in cat_month.mean(axis=1).sort_values(ascending=False).items()},
        "budget_split": budget_split(df),
        "bucket_averages": bucket_averages(df),
        "savings_levers": savings_levers(df),
        "subscriptions": detect_subscriptions(df),
        "top_merchants": {p: _r(v) for p, v in top_merch.items()},
    }


def profile_text(p: dict) -> str:
    """Compact text version of the profile for prompts (keeps token use low)."""
    tot_in = sum(m["income"] for m in p["months"])
    tot_out = sum(m["spent"] for m in p["months"])
    lines = [f"Statement period: {p['period']}",
             f"Totals for the whole period: income {inr(tot_in)}, spent {inr(tot_out)}, "
             f"net {'+' if tot_in >= tot_out else ''}{inr(tot_in - tot_out)}"]
    lines.append("Per month (income / spent / invested / net / savings rate incl. investments):")
    for m in p["months"]:
        lines.append(f"  {m['month']}: {inr(m['income'])} / {inr(m['spent'])} / {inr(m['invested'])} / "
                     f"{inr(m['net'])} / " + (f"{m['savings_rate_pct']}%" if m["savings_rate_pct"] is not None
                                               else "n/a (no income this month)"))
    months = [m["month"] for m in p["months"]]
    lines.append("Spending by category per month (" + ", ".join(months) + "):")
    for c, avg in p["category_monthly_average"].items():
        vals = " / ".join(f"{inr(p['category_by_month'][c].get(m, 0))}" for m in months)
        lines.append(f"  {c}: {vals} (avg {inr(avg)})")
    b = p["budget_split"]
    lines.append(f"50/30/20 check over the period: needs {b['needs_pct']}%, wants {b['wants_pct']}%, "
                 f"savings {b['savings_pct']}% of income ({inr(b['income'])}).")
    ba = p["bucket_averages"]
    lines.append(f"Monthly averages vs 50/30/20 targets (average income {inr(ba['income'])}): " + "; ".join(
        f"{k} {inr(ba[k]['actual'])} vs target {inr(ba[k]['target'])} ({'+' if ba[k]['gap'] >= 0 else ''}"
        f"{inr(ba[k]['gap'])})" for k in ("needs", "wants", "savings")))
    if p["savings_levers"]:
        lines.append("Savings levers (if every month matched your own best month for that category):")
        for lv in p["savings_levers"]:
            lines.append(f"  {lv['category']}: average {inr(lv['monthly_average'])}/month, best "
                         f"{inr(lv['best_month_amount'])} in {lv['best_month']} -> saves "
                         f"{inr(lv['monthly_saving_if_like_best_month'])}/month")
    subs = [s for s in p["subscriptions"] if s["category"] == "Entertainment"]
    bills = [s for s in p["subscriptions"] if s["category"] != "Entertainment"]
    if subs:
        lines.append("Optional subscriptions (can be cancelled or paused): " + "; ".join(
            f"{s['payee']} {inr(s['amount'])}/month{' (new since ' + s['first_month'] + ')' if s['new'] else ''}"
            for s in subs) + f"; total {inr(sum(s['amount'] for s in subs))}/month")
    if bills:
        lines.append("Recurring bills (essential needs, do not suggest cancelling): " + "; ".join(
            f"{s['payee']} {inr(s['amount'])}/month" for s in bills))
    lines.append("Top payees by total: " + ", ".join(f"{k} {inr(v)}" for k, v in p["top_merchants"].items()))
    return "\n".join(lines)


# =================================================================== alerts
def detect_alerts(df: pd.DataFrame) -> list[dict]:
    alerts: list[dict] = []
    mt = monthly_table(df)
    debit = df[df["type"] == "DEBIT"].assign(month=_months(df))
    income_avg = mt["income"].mean()

    # --- whole months
    for m, r in mt.iterrows():
        if r["net"] < 0:
            alerts.append(_alert("negative_month", "high", f"Spent more than you earned in {m}",
                                 f"Spending {inr(r['spent'])} was {inr(-r['net'])} more than income "
                                 f"{inr(r['income'])}.", impact=-r["net"], month=m,
                                 income=_r(r["income"]), spent=_r(r["spent"])))
        elif r["savings_rate_pct"] < C.LOW_SAVINGS_RATE_PCT:
            target = r["income"] * C.LOW_SAVINGS_RATE_PCT / 100
            gap = target - (r["net"] + r["invested"])
            alerts.append(_alert("low_savings", "medium" if gap >= C.MIN_ALERT_AMOUNT else "low", f"Low savings rate in {m}",
                                 f"You kept {r['savings_rate_pct']}% of income (target "
                                 f"{C.LOW_SAVINGS_RATE_PCT:.0f}%). About {inr(gap)} short.",
                                 impact=gap, month=m, savings_rate_pct=r["savings_rate_pct"]))

    # --- category spikes vs previous month
    cat_month = debit.pivot_table(index="category", columns="month", values="amount", aggfunc="sum", fill_value=0)
    cols = list(cat_month.columns)
    for prev, cur in zip(cols, cols[1:]):
        for cat, row in cat_month.iterrows():
            a, b = row[prev], row[cur]
            if a > 0 and b - a >= C.MIN_ALERT_AMOUNT and (b - a) / a * 100 > C.MOM_SPIKE_PCT:
                pct = (b - a) / a * 100
                sev = "high" if b - a >= 10_000 or (pct > 100 and b - a >= 5_000) else "medium"
                alerts.append(_alert("category_spike", sev, f"{cat} up {pct:.0f}% in {cur}",
                                     f"{inr(b)} in {cur} vs {inr(a)} in {prev} (+{inr(b - a)}).",
                                     impact=b - a, month=cur, category=cat, previous=_r(a), current=_r(b),
                                     change_pct=_r(pct)))

    # --- 50/30/20
    bs = budget_split(df)
    if bs["wants_pct"] > C.BUDGET_SPLIT["wants"]:
        excess = (bs["wants_pct"] - C.BUDGET_SPLIT["wants"]) / 100 * income_avg
        alerts.append(_alert("budget_split", "medium", "Wants are above 30% of income",
                             f"Needs {bs['needs_pct']}%, wants {bs['wants_pct']}%, savings {bs['savings_pct']}% "
                             f"(target 50/30/20). Wants are about {inr(excess)}/month over.",
                             impact=excess, **{k: bs[k] for k in ("needs_pct", "wants_pct", "savings_pct")}))

    # --- habits per month
    small = debit[(debit["mode"] == "UPI") & (debit["amount"] <= C.SMALL_UPI_AMOUNT)]
    for m, g in small.groupby("month"):
        if len(g) >= C.SMALL_UPI_COUNT:
            alerts.append(_alert("small_upi", "low", f"{len(g)} small UPI payments in {m}",
                                 f"{len(g)} payments of ₹{C.SMALL_UPI_AMOUNT:.0f} or less added up to "
                                 f"{inr(g['amount'].sum())}.", impact=g["amount"].sum() * 0.4, month=m,
                                 count=len(g), total=_r(g["amount"].sum())))
    food = debit[debit["category"] == "Food & Dining"]
    for m, g in food.groupby("month"):
        if len(g) >= C.FOOD_ORDERS_PER_MONTH:
            alerts.append(_alert("food_orders", "medium", f"{len(g)} food & dining payments in {m}",
                                 f"{inr(g['amount'].sum())} on food & dining, average {inr(g['amount'].mean())} each.",
                                 impact=g["amount"].sum() * 0.35, month=m, category="Food & Dining",
                                 count=len(g), total=_r(g["amount"].sum())))

    # --- subscriptions
    subs = detect_subscriptions(df)
    ent_subs = [s for s in subs if s["category"] == "Entertainment"]
    if ent_subs:
        total = sum(s["amount"] for s in ent_subs)
        new = [s["payee"] for s in ent_subs if s["new"]]
        alerts.append(_alert("subscriptions", "low", f"{len(ent_subs)} subscriptions, {inr(total)}/month",
                             "Recurring: " + ", ".join(f"{s['payee']} {inr(s['amount'])}" for s in ent_subs)
                             + (f". New since the statement began: {', '.join(new)}." if new else "."),
                             impact=total, category="Entertainment", subscriptions=ent_subs))

    # --- EMI and cash
    emi = debit[debit["category"] == "EMI & Loans"].groupby("month")["amount"].sum().mean() if income_avg else 0
    if income_avg and emi:
        ratio = emi / income_avg * 100
        if ratio > C.EMI_RATIO_LIMIT_PCT:
            alerts.append(_alert("emi_ratio", "high", f"EMIs take {ratio:.0f}% of income",
                                 f"Average EMIs {inr(emi)}/month vs income {inr(income_avg)}.",
                                 impact=emi - income_avg * C.EMI_RATIO_LIMIT_PCT / 100, emi_ratio_pct=_r(ratio)))
        else:
            alerts.append(_alert("positive_emi", "positive", f"EMIs are a manageable {ratio:.0f}% of income",
                                 f"Average EMIs {inr(emi)}/month, under the {C.EMI_RATIO_LIMIT_PCT:.0f}% guideline.",
                                 emi_ratio_pct=_r(ratio)))
    cash = debit[debit["mode"] == "ATM"]["amount"].sum()
    if debit["amount"].sum() and cash / debit["amount"].sum() * 100 > C.CASH_SHARE_LIMIT_PCT:
        share = cash / debit["amount"].sum() * 100
        alerts.append(_alert("cash_share", "low", f"{share:.0f}% of spending was ATM cash",
                             f"{inr(cash)} withdrawn in cash, which isn't tracked by category.",
                             cash=_r(cash), share_pct=_r(share)))

    # --- positives
    inv = debit[debit["category"].isin(C.SAVINGS_CATEGORIES)]
    if not inv.empty and inv["month"].nunique() == len(mt):
        alerts.append(_alert("positive_investing", "positive", "Invested every month",
                             f"{inr(inv['amount'].sum())} invested across {inv['month'].nunique()} months "
                             f"({inr(inv.groupby('month')['amount'].sum().mean())}/month on average).",
                             total=_r(inv["amount"].sum())))
    good = mt[mt["savings_rate_pct"] >= C.LOW_SAVINGS_RATE_PCT]
    if len(good):
        best = good["savings_rate_pct"].idxmax()
        alerts.append(_alert("positive_savings", "positive", f"Best month: {best}",
                             f"You kept {good.loc[best, 'savings_rate_pct']}% of income in {best}.", month=best))

    return sorted(alerts, key=lambda a: (SEVERITY_ORDER[a["severity"]], -a["monthly_impact"]))


def alerts_text(alerts: list[dict], limit: int = 10) -> str:
    if not alerts:
        return "No alerts."
    return "\n".join(f"- [{a['severity']}] {a['title']}: {a['detail']}" for a in alerts[:limit])
