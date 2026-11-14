"""Categorizer Agent: tags every transaction with one of the spending categories.

Three tiers, cheapest first:
  1. Rules: a keyword table of well-known Indian merchants and statement patterns
     (SWIGGY -> Food & Dining, ACH .. EMI -> EMI & Loans, SALARY credit -> Salary & Income ...).
  2. LLM with dynamic few-shot: transactions the rules can't place (local shops, people) are
     grouped by payee and sent in batches. For each group, similar already-labelled payees are
     retrieved from ChromaDB and put in the prompt as examples. The model reasons step by step
     (Chain-of-Thought) and returns a category plus a confidence.
  3. Reflection: anything below the confidence threshold gets a second, self-critical pass.

User corrections from the UI are saved into the same ChromaDB collection, so the next
statement benefits from them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from spendsense import config
from spendsense.schemas import CategoryBatch, ReflectionBatch

CONFIDENCE_THRESHOLD = 0.75   # below this, the reflection pass re-checks the label
BATCH_SIZE = 15
EXAMPLES_PER_GROUP = 3

# =================================================================== 1. rules
# (category, regex) checked in order against "PAYEE NOTE DESCRIPTION" in upper case
RULES: list[tuple[str, str]] = [
    ("Investments", r"\bSIP\b|MUTUAL FUND|\bMF\b|ZERODHA|GROWW|\bNPS\b|\bPPF\b|KUVERA|COIN BY"),
    ("EMI & Loans", r"\bEMI\b|\bLOAN\b|BAJAJ FINSERV|CREDIT CARD PAYMENT|\bCC PAYMENT\b"),
    ("Food & Dining", r"SWIGGY(?! ?INSTAMART)|ZOMATO|DOMINOS|PIZZA|STARBUCKS|MCDONALD|\bKFC\b|BURGER KING|"
                      r"CAFE COFFEE DAY|EATSURE|HALDIRAM|BEHROUZ|FAASOS|BARBEQUE NATION"),
    ("Groceries", r"BLINKIT|ZEPTO|BIGBASKET|\bDMART\b|INSTAMART|JIOMART|NATURE'?S BASKET|MORE RETAIL|"
                  r"RELIANCE FRESH|SPENCER'?S|COUNTRY DELIGHT|MILKBASKET"),
    ("Shopping", r"AMAZON(?! PRIME VIDEO)|FLIPKART|MYNTRA|\bAJIO\b|MEESHO|NYKAA|DECATHLON|CROMA|"
                 r"RELIANCE DIGITAL|LIFESTYLE|WESTSIDE|IKEA|TATA CLIQ|LENSKART"),
    ("Transport", r"\bUBER\b|\bOLA\b|OLACABS|RAPIDO|METRO|IRCTC|REDBUS|PETROL|\bFUEL\b|HPCL|BPCL|"
                  r"INDIAN OIL|FASTAG|NAMMA YATRI|BLUSMART"),
    ("Bills & Utilities", r"ELECTRICITY|AIRTEL|\bJIO\b|VODAFONE|BSNL|FIBERNET|BROADBAND|\bGAS\b|INDANE|"
                          r"BHARAT GAS|WATER BILL|\bDTH\b|TATA PLAY|RECHARGE|BILL ?PAY"),
    ("Entertainment", r"NETFLIX|SPOTIFY|YOUTUBE|PRIME VIDEO|HOTSTAR|JIOCINEMA|BOOKMYSHOW|\bPVR\b|INOX|"
                      r"STEAM|PLAYSTATION|SONYLIV|ZEE5"),
    ("Health", r"PHARMACY|MEDPLUS|APOLLO|\b1MG\b|PHARMEASY|NETMEDS|HOSPITAL|DIAGNOSTIC|"
               r"PATHLAB|THYROCARE"),
    ("Education", r"UDEMY|COURSERA|BYJU|UNACADEMY|SCHOOL FEE|COLLEGE FEE|TUITION|EDX\b|SKILLSHARE"),
]
INCOME_RULE = r"SALARY|\bBONUS\b|INT\.?PD|INTEREST|DIVIDEND|STIPEND"
_COMPILED = [(cat, re.compile(rx)) for cat, rx in RULES]
_INCOME = re.compile(INCOME_RULE)


def _text(row) -> str:
    return f"{row['payee']} {row.get('note', '')} {row['description']}".upper()


def rule_category(row) -> tuple[str | None, str]:
    """Return (category, reason) or (None, '') if no rule applies."""
    text = _text(row)
    if row["mode"] == "ATM":
        return "Others", "ATM cash withdrawal"
    if row["type"] == "CREDIT":
        if _INCOME.search(text) or row["mode"] == "INT":
            return "Salary & Income", "Credit matching salary/bonus/interest"
        if "REFUND" in text or "REVERSAL" in text:
            for cat, rx in _COMPILED:
                if rx.search(text):
                    return cat, f"Refund from a {cat} merchant"
        return None, ""  # other credits (people, unknown) -> LLM
    for cat, rx in _COMPILED:
        m = rx.search(text)
        if m:
            return cat, f"Keyword '{m.group(0)}'"
    return None, ""


# =================================================================== 2. examples store
# A few generic, hand-labelled examples so the LLM has guidance even before any statement
# has been categorised. Names are deliberately different from the sample data.
SEED_EXAMPLES: list[tuple[str, str, str, str]] = [
    # payee, note, type, category
    ("SURESH PROVISION STORE", "Groceries", "DEBIT", "Groceries"),
    ("MANI TEA STALL", "Tea", "DEBIT", "Others"),
    ("GANESH FLOWER SHOP", "", "DEBIT", "Others"),
    ("CITY PARKING", "Parking", "DEBIT", "Others"),
    ("HOTEL MURUGAN IDLI SHOP", "Breakfast", "DEBIT", "Food & Dining"),
    ("ANITHA RAJAN", "Dinner split", "DEBIT", "Transfers to People"),
    ("MOHAN DAS", "Paying back", "CREDIT", "Transfers to People"),
    ("SELVI M", "RENT AUG", "DEBIT", "Rent"),
    ("CITY CARE HOSPITAL", "Consultation", "DEBIT", "Health"),
    ("GOLD GYM", "Membership", "DEBIT", "Health"),
    ("SUNRISE TUITION CENTRE", "Fees", "DEBIT", "Education"),
    ("CORPORATION WATER", "Water tax", "DEBIT", "Bills & Utilities"),
]


def _example_doc(payee: str, note: str, ttype: str) -> str:
    return f"{payee} | {note} | {ttype}"


def _example_id(payee: str, ttype: str) -> str:
    return hashlib.md5(f"{payee}|{ttype}".encode()).hexdigest()


def add_examples(store, items: list[tuple[str, str, str, str]], source: str) -> None:
    """Upsert labelled payees into ChromaDB (one entry per payee+type; later labels win)."""
    if not items:
        return
    uniq = {_example_id(p, t): (p, n, t, c) for p, n, t, c in items}
    store.add_texts(
        texts=[_example_doc(p, n, t) for p, n, t, _ in uniq.values()],
        metadatas=[{"payee": p, "type": t, "category": c, "source": source} for p, _, t, c in uniq.values()],
        ids=list(uniq),
    )


def retrieve_examples(store, query: str, k: int = EXAMPLES_PER_GROUP) -> list[str]:
    return retrieve_examples_scored(store, query, k)[0]


def retrieve_examples_scored(store, query: str, k: int = EXAMPLES_PER_GROUP) -> tuple[list[str], str | None, float]:
    """Examples for the prompt, plus the nearest neighbour's category and similarity (for the no-LLM fallback)."""
    hits = store.similarity_search_with_relevance_scores(query, k=k)
    lines = [f"{d.page_content} -> {d.metadata['category']}" for d, _ in hits]
    if not hits:
        return lines, None, 0.0
    best, score = hits[0]
    return lines, best.metadata["category"], float(score)


VECTOR_FALLBACK_MIN_SCORE = 0.80   # nearest labelled payee must be this similar to borrow its category


# =================================================================== result
@dataclass
class CategorizeResult:
    df: pd.DataFrame
    llm_calls: int = 0
    groups_sent_to_llm: int = 0
    reflection: list[dict] = field(default_factory=list)   # what the self-check reviewed/changed
    trace: list[dict] = field(default_factory=list)        # per payee group, for the UI
    llm_error: str | None = None                           # set when the LLM was unavailable

    def summary(self) -> dict:
        src = self.df["cat_source"].value_counts().to_dict()
        return {
            "transactions": len(self.df),
            "by_source": src,
            "llm_error": self.llm_error,
            "llm_calls": self.llm_calls,
            "groups_sent_to_llm": self.groups_sent_to_llm,
            "reflection_reviewed": len(self.reflection),
            "reflection_changed": sum(1 for r in self.reflection if r["changed"]),
            "category_counts": self.df["category"].value_counts().to_dict(),
        }


# =================================================================== 3. agent
def _default_llm_fns():
    from spendsense.chains import structured_chain
    cat_chain = structured_chain("categorizer", CategoryBatch)
    ref_chain = structured_chain("categorizer_reflection", ReflectionBatch)
    # return_exceptions: one failed batch (e.g. quota exhausted) must not lose the others
    return (lambda batches: cat_chain.batch(batches, return_exceptions=True), lambda inp: ref_chain.invoke(inp))


def _short_error(exc: Exception) -> str:
    msg = str(exc)
    if "rate_limit" in msg or "Rate limit" in msg or "429" in msg:
        return "LLM quota used up (Groq rate limit)"
    return f"LLM unavailable ({type(exc).__name__})"


def categorize(df: pd.DataFrame, store=None, use_llm: bool = True,
               llm_batch_fn: Callable | None = None, reflect_fn: Callable | None = None) -> CategorizeResult:
    """Add category, cat_source, cat_confidence, cat_reason columns. `*_fn` overrides are for tests."""
    df = df.copy()
    df["category"], df["cat_source"], df["cat_confidence"], df["cat_reason"] = None, None, None, ""

    # --- tier 1: rules
    for i, row in df.iterrows():
        cat, why = rule_category(row)
        if cat:
            df.loc[i, ["category", "cat_source", "cat_confidence", "cat_reason"]] = [cat, "rule", 1.0, why]

    result = CategorizeResult(df=df)
    todo = df[df["category"].isna()]
    if todo.empty:
        return result
    if not use_llm:
        df.loc[todo.index, ["category", "cat_source", "cat_confidence", "cat_reason"]] = \
            ["Others", "default", 0.0, "No rule matched (LLM disabled)"]
        return result

    # --- tier 2: LLM with retrieved examples, one entry per (payee, type) group
    if store is None:
        from spendsense.rag.vectorstore import get_store
        store = get_store(config.EXAMPLES_COLLECTION)
    add_examples(store, SEED_EXAMPLES, "seed")
    ruled = df[df["cat_source"] == "rule"]
    add_examples(store, [(r.payee, "", r.type, r.category) for r in ruled.itertuples()], "rule")

    groups = []
    for gid, ((payee, ttype), g) in enumerate(todo.groupby(["payee", "type"], sort=False), 1):
        notes = [n for n in g["note"].dropna().unique() if n][:3]
        vpa = next((v for v in g["vpa"] if v), "")
        amts = g["amount"]
        line = (f"G{gid} | {g['date'].iloc[0]:%Y-%m-%d} | {ttype} | "
                f"₹{amts.min():,.0f}-₹{amts.max():,.0f} over {len(g)} txns | payee={payee} | vpa={vpa} | "
                f"notes={'; '.join(notes) or '-'} | description={g['description'].iloc[0]}")
        examples, nn_cat, nn_score = retrieve_examples_scored(store, f"{payee} {' '.join(notes)} {ttype}")
        groups.append({"gid": f"G{gid}", "payee": payee, "type": ttype, "index": g.index, "line": line,
                       "examples": examples, "count": len(g), "nn_category": nn_cat, "nn_score": nn_score})

    if llm_batch_fn is None or reflect_fn is None:
        default_batch, default_reflect = _default_llm_fns()
        llm_batch_fn, reflect_fn = llm_batch_fn or default_batch, reflect_fn or default_reflect

    chunks = [groups[i:i + BATCH_SIZE] for i in range(0, len(groups), BATCH_SIZE)]
    inputs = []
    for chunk in chunks:
        ex = sorted({e for grp in chunk for e in grp["examples"]})
        inputs.append({"examples": "\n".join(f"- {e}" for e in ex), "transactions": "\n".join(g["line"] for g in chunk)})
    try:
        outputs = llm_batch_fn(inputs)
    except Exception as exc:  # noqa: BLE001 - the upload must still succeed without the LLM
        outputs = [exc] * len(inputs)
    result.llm_calls += len(inputs)
    result.groups_sent_to_llm = len(groups)

    answers, failed = {}, set()
    for chunk, out in zip(chunks, outputs):
        if isinstance(out, Exception):
            result.llm_error = _short_error(out)
            failed.update(g["gid"] for g in chunk)
        else:
            answers.update({a.txn_id: a for a in out.assignments})
    by_gid = {g["gid"]: g for g in groups}
    for g in groups:
        a = answers.get(g["gid"])
        if a:
            g.update(category=a.category, confidence=a.confidence, reasoning=a.reasoning)
        elif g["gid"] in failed and g["nn_category"] and g["nn_score"] >= VECTOR_FALLBACK_MIN_SCORE:
            # No LLM: borrow the category of the most similar already-labelled payee in ChromaDB
            g.update(category=g["nn_category"], confidence=round(g["nn_score"], 2), source="vector",
                     reasoning=f"{result.llm_error}; nearest labelled payee in ChromaDB is {g['nn_category']} "
                               f"(similarity {g['nn_score']:.2f})")
        elif g["gid"] in failed:
            g.update(category="Others", confidence=0.0, source="default",
                     reasoning=f"{result.llm_error}; no similar labelled payee found")
        else:
            g.update(category="Others", confidence=0.0, reasoning="Missing from LLM answer")

    # --- tier 3: reflection on low-confidence groups
    low = [g for g in groups if g["confidence"] < CONFIDENCE_THRESHOLD and "source" not in g]
    ref = None
    if low and not result.llm_error:
        items = "\n".join(f"{g['gid']} | {g['line'].split(' | description=')[0]} | original={g['category']} | "
                          f"reasoning={g['reasoning']}" for g in low)
        try:
            ref = reflect_fn({"items": items})
            result.llm_calls += 1
        except Exception as exc:  # noqa: BLE001 - keep the first-pass labels if the self-check can't run
            result.llm_error = _short_error(exc)
    if ref is not None:
        for it in ref.items:
            g = by_gid.get(it.txn_id)
            if not g:
                continue
            result.reflection.append({"group": g["gid"], "payee": g["payee"], "original": g["category"],
                                      "final": it.final_category, "changed": it.final_category != g["category"],
                                      "critique": it.critique})
            g["reflected"] = True
            g["category"] = it.final_category
            g["reasoning"] = f"{g['reasoning']} | Self-check: {it.critique}"

    # --- write back + learn
    for g in groups:
        src = g.get("source") or ("llm+reflection" if g.get("reflected") else "llm")
        df.loc[g["index"], ["category", "cat_source", "cat_confidence", "cat_reason"]] = \
            [g["category"], src, round(float(g["confidence"]), 2), g["reasoning"]]
        result.trace.append({k: g[k] for k in ("gid", "payee", "type", "count", "examples", "category",
                                               "confidence", "reasoning")} | {"source": src})
    # only LLM decisions become new examples; fallback guesses must not teach the store
    confident = [g for g in groups if "source" not in g
                 and (g["confidence"] >= CONFIDENCE_THRESHOLD or g.get("reflected"))]
    add_examples(store, [(g["payee"], "", g["type"], g["category"]) for g in confident], "llm")
    result.df = df
    return result


def apply_user_correction(df: pd.DataFrame, txn_id: str, category: str, store=None,
                          apply_to_same_payee: bool = True) -> pd.DataFrame:
    """User fixes a category in the UI: update the table and remember it in ChromaDB."""
    if category not in config.CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    df = df.copy()
    row = df.loc[df["txn_id"] == txn_id].iloc[0]
    mask = (df["payee"] == row["payee"]) & (df["type"] == row["type"]) if apply_to_same_payee else df["txn_id"] == txn_id
    df.loc[mask, ["category", "cat_source", "cat_confidence", "cat_reason"]] = [category, "user", 1.0, "Corrected by user"]
    if store is None:
        from spendsense.rag.vectorstore import get_store
        store = get_store(config.EXAMPLES_COLLECTION)
    add_examples(store, [(row["payee"], row.get("note", ""), row["type"], category)], "user")
    return df
