"""Parser Agent: bank / UPI statement (CSV or PDF) -> one clean transactions table.

Strategy, cheapest first:
  1. Read the file into rows (CSV with delimiter/encoding detection, or PDF tables via pdfplumber).
  2. Rule-based layout detection: find the header row and map columns using known synonyms
     ("Narration", "Withdrawal Amt.", "Dr/Cr", ...). Works for most exports with no LLM call.
  3. LLM fallback: if the rules can't find the columns, send the first lines to the LLM with the
     few-shot `parser_mapping` prompt and get a validated `ColumnMapping` back.
  4. Normalise: parse dates and amounts (₹, Indian commas, Dr/Cr), work out debit vs credit,
     drop FAILED rows, repeated headers and summary lines.
  5. Enrich: extract payee, payment mode, UPI VPA and note from the description.
  6. Verify: if there is a balance column, check every row's running balance (self-check).

Output columns (one row per transaction):
  txn_id, date, description, payee, mode, type (DEBIT/CREDIT), amount (always positive),
  balance, ref, vpa, note
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from spendsense.schemas import ColumnMapping

OUTPUT_COLUMNS = ["txn_id", "date", "description", "payee", "mode", "type", "amount",
                  "balance", "ref", "vpa", "note"]

# Header synonyms, compared after normalising (lower case, single spaces)
SYNONYMS: dict[str, list[str]] = {
    "date": ["txn date", "transaction date", "date", "tran date", "posting date", "value date", "value dt"],
    "description": ["description", "narration", "particulars", "transaction details", "details",
                    "remarks", "transaction remarks"],
    "debit": ["debit", "withdrawal amt.", "withdrawal amt", "withdrawal", "withdrawals", "withdrawal amount",
              "debit amount", "debit amt", "dr amount"],
    "credit": ["credit", "deposit amt.", "deposit amt", "deposit", "deposits", "deposit amount",
               "credit amount", "credit amt", "cr amount"],
    "amount": ["amount", "amount (inr)", "amount(inr)", "transaction amount", "amt", "amount (rs.)", "amount in inr"],
    "type": ["type", "dr/cr", "cr/dr", "dr / cr", "transaction type", "debit/credit", "txn type"],
    "balance": ["balance", "closing balance", "available balance", "balance (inr)", "running balance"],
    "ref": ["ref no./cheque no.", "ref no.", "ref no", "chq./ref.no.", "chq/ref no", "upi transaction id",
            "upi ref no", "reference", "reference no", "utr", "transaction id", "cheque no."],
    "status": ["status", "transaction status"],
    "time": ["time", "transaction time"],
}
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d", "%d %b %Y", "%d-%b-%Y",
                "%d %b %y", "%d-%b-%y", "%d.%m.%Y", "%b %d, %Y", "%d %B %Y"]
OK_STATUSES = {"success", "successful", "completed", "complete", "settled", ""}


# =================================================================== result
@dataclass
class ParseResult:
    df: pd.DataFrame
    source_format: str                 # "csv" | "pdf"
    method: str                        # "rules" | "llm_mapping"
    mapping: dict
    rows_read: int
    dropped: dict[str, int] = field(default_factory=dict)
    balance_check: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        df = self.df
        return {
            "transactions": len(df),
            "source_format": self.source_format,
            "method": self.method,
            "mapping": self.mapping,
            "rows_read": self.rows_read,
            "dropped": self.dropped,
            "date_range": [df["date"].min().date().isoformat(), df["date"].max().date().isoformat()] if len(df) else None,
            "total_debit": round(float(df.loc[df["type"] == "DEBIT", "amount"].sum()), 2),
            "total_credit": round(float(df.loc[df["type"] == "CREDIT", "amount"].sum()), 2),
            "modes": df["mode"].value_counts().to_dict(),
            "balance_check": self.balance_check,
            "warnings": self.warnings,
        }


class ParseError(ValueError):
    """Raised with a message that can be shown to the user as-is."""


# =================================================================== 1. read rows
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not read the file's text encoding.")


def read_csv_rows(data: bytes) -> list[list[str]]:
    text = _decode(data)
    # Pick the delimiter that splits the most lines into 3+ fields (title lines above the
    # header confuse csv.Sniffer, so count per line instead)
    lines = [l for l in text.splitlines()[:40] if l.strip()]
    best = max(",;\t|", key=lambda d: sum(1 for l in lines if len(next(csv.reader([l], delimiter=d))) >= 3))
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text), delimiter=best)]


def read_pdf_rows(data: bytes) -> list[list[str]]:
    import pdfplumber

    rows: list[list[str]] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in table:
                        # wrapped cell text comes back with newlines; rejoin it
                        cells = [re.sub(r"\s*\n\s*", " ", c or "").strip() for c in row]
                        if any(cells):
                            rows.append([re.sub(r"/ ", "/", c) for c in cells])
    except Exception as exc:  # noqa: BLE001 - pdfplumber raises many types
        raise ParseError(f"Could not open the PDF: {exc}") from exc
    if not rows:
        raise ParseError("No transaction table found in this PDF. Scanned (image) PDFs are not supported; "
                         "try the CSV export from your bank instead.")
    return rows


# =================================================================== 2. layout detection
def _norm(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip().lower())


def _match_columns(header: list[str]) -> dict[str, str]:
    """Map logical field -> actual header name, preferring the earliest synonym in each list."""
    normed = {_norm(h): h for h in header if h}
    found: dict[str, str] = {}
    used: set[str] = set()
    for fld, names in SYNONYMS.items():
        for name in names:
            if name in normed and normed[name] not in used:
                found[fld] = normed[name]
                used.add(normed[name])
                break
    return found


def detect_layout_rules(rows: list[list[str]], scan: int = 30) -> tuple[int, dict[str, str]] | None:
    """Return (header_row_index, field->column) if the rules recognise the layout."""
    for i, row in enumerate(rows[:scan]):
        cols = _match_columns(row)
        if "date" in cols and "description" in cols and ({"debit", "credit"} & cols.keys() or "amount" in cols):
            return i, cols
    return None


def detect_layout_llm(rows: list[list[str]], mapper: Callable[[str], ColumnMapping] | None = None,
                      n_lines: int = 15) -> tuple[int, dict[str, str], str | None]:
    """Ask the LLM (few-shot parser_mapping prompt) which column is which. Validates the answer."""
    sample = "\n".join(f"line {i}: {','.join(r)}" for i, r in enumerate(rows[:n_lines]))
    if mapper is None:
        from spendsense.chains import structured_chain
        chain = structured_chain("parser_mapping", ColumnMapping)
        mapper = lambda s: chain.invoke({"sample_lines": s})  # noqa: E731
    m = mapper(sample)
    if not 0 <= m.header_row_index < len(rows):
        raise ParseError("The statement layout could not be recognised (bad header row).")
    header = rows[m.header_row_index]
    cols = {
        "date": m.date_col, "description": m.description_col, "debit": m.debit_col, "credit": m.credit_col,
        "amount": m.amount_col, "type": m.type_col, "balance": m.balance_col, "ref": m.ref_col,
        "status": m.status_col,
    }
    # Snap the LLM's names onto the real header names (ignoring case and spacing)
    by_norm = {_norm(h): h for h in header}
    cols = {k: by_norm.get(_norm(v), v) for k, v in cols.items() if v}
    missing = [v for v in cols.values() if v not in header]
    if missing:
        raise ParseError(f"The statement layout could not be recognised (columns not found: {missing}).")
    if not ({"debit", "credit"} & cols.keys() or "amount" in cols):
        raise ParseError("The statement layout could not be recognised (no amount column).")
    return m.header_row_index, cols, m.date_format


# =================================================================== 4. normalise values
_AMOUNT_JUNK = re.compile(r"(₹|rs\.?|inr|\s)", re.I)


def parse_amount(value) -> tuple[float | None, str | None]:
    """'₹1,23,456.00' -> 123456.0. Also returns 'DEBIT'/'CREDIT' if the text carries a sign or Dr/Cr."""
    s = str(value or "").strip()
    if not s or s in {"-", "--"}:
        return None, None
    hint = None
    low = s.lower()
    if low.endswith(("dr", "dr.")):
        hint, s = "DEBIT", re.sub(r"dr\.?$", "", s, flags=re.I)
    elif low.endswith(("cr", "cr.")):
        hint, s = "CREDIT", re.sub(r"cr\.?$", "", s, flags=re.I)
    s = _AMOUNT_JUNK.sub("", s).replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        hint, s = "DEBIT", s[1:-1]
    if s.startswith("-"):
        hint, s = "DEBIT", s[1:]
    elif s.startswith("+"):
        hint, s = "CREDIT", s[1:]
    try:
        return round(float(s), 2), hint
    except ValueError:
        return None, None


def parse_date(value: str, fmt: str | None = None) -> datetime | None:
    s = str(value or "").strip()
    if not s:
        return None
    for f in ([fmt] if fmt else []) + DATE_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _type_from_text(value: str) -> str | None:
    v = _norm(value)
    if v in {"dr", "debit", "d", "withdrawal", "paid", "sent"} or v.startswith(("debit", "paid", "sent")):
        return "DEBIT"
    if v in {"cr", "credit", "c", "deposit", "received"} or v.startswith(("credit", "received")):
        return "CREDIT"
    return None


# =================================================================== 5. enrich description
def _clean_name(s: str) -> str:
    s = re.sub(r"\.(com|in|co\.in)\b", "", s.strip(), flags=re.I)
    return re.sub(r"\s+", " ", s).strip().upper()


_PREFIX = re.compile(
    r"^(?:(?:DEP|WDL|DR|CR)\s*TFR|TO\s+TRANSFER|BY\s+TRANSFER|TRANSFER\s+(?:TO|FROM)|TRF|TFR|INB|MB)\b[\s\-:/]*",
    re.I)
_SKIP_TOKENS = {"DR", "CR", "P2A", "P2M", "P2P", "PAY", "COLLECT", "UPI", "NEFT", "IMPS", "RTGS", "INB", "MB"}


def _pick_name(tokens: list[str]) -> str:
    """First token that looks like a name: skips DR/CR flags, reference numbers, IFSC codes and VPAs."""
    for t in tokens:
        tt = t.strip()
        u = tt.upper()
        if (not tt or u in _SKIP_TOKENS or "@" in tt or re.fullmatch(r"\d+", u)
                or re.fullmatch(r"[A-Z]{0,4}\d{5,}[A-Z0-9]*", u)       # refs like N223005401501, UTR numbers
                or re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", u)):           # IFSC codes like HDFC0000123
            continue
        return tt
    return ""


def extract_details(desc: str) -> dict:
    """Pull payee, mode, VPA and UPI note out of common Indian bank description formats."""
    d = re.sub(r"\s+", " ", desc or "").strip()   # statements often wrap narrations across lines
    up = d.upper()
    out = {"payee": "", "mode": "OTHER", "vpa": "", "note": ""}

    # UPI app style: "Paid to Swiggy" / "Received from Rahul Sharma"
    m = re.match(r"^(paid to|received from|sent to|money sent to|money received from)\s+(.+)$", d, re.I)
    if m:
        return {**out, "payee": _clean_name(m.group(2)), "mode": "UPI"}

    # Bank prefixes before the real narration, e.g. SBI "DEP TFR UPI/CR/..." / "WDL TFR UPI/DR/...",
    # "TO TRANSFER-UPI/DR/...", "BY TRANSFER-NEFT*...": strip them and parse what follows.
    stripped = d
    while (m := _PREFIX.match(stripped)) and m.end() > 0:
        stripped = stripped[m.end():]
    if stripped and stripped != d:
        inner = extract_details(stripped)
        if inner["payee"]:
            return inner
    # A payment keyword somewhere inside the text ("... UPI/DR/...") - parse from there
    m = re.search(r"(?<![A-Z])(UPI|NEFT|IMPS|RTGS)[/*\-]", up)
    if m and m.start() > 0:
        inner = extract_details(d[m.start():])
        if inner["payee"]:
            return inner

    parts = [p.strip() for p in re.split(r"[/]", d)]
    head = parts[0].upper() if parts else ""

    if head == "UPI" and len(parts) >= 3:
        # UPI/DR/ref/PAYEE/bank/vpa/note, UPI/ref/PAYEE/vpa/..., UPI/P2A/ref/PAYEE/...
        out.update(mode="UPI", payee=_clean_name(_pick_name(parts[1:])))
        vpa = next((p for p in parts[1:] if "@" in p), "")
        out["vpa"] = vpa.lower()
        if vpa and parts.index(vpa) + 1 < len(parts):
            out["note"] = "/".join(parts[parts.index(vpa) + 1:]).strip()
        return out
    if up.startswith("UPI-") or up.startswith("UPI "):
        # HDFC-style: UPI-ZOMATO-zomato@hdfc-HDFC0000-ref-note
        bits = [b.strip() for b in re.split(r"[-]", d)]
        vpa = next((b for b in bits if "@" in b), "")
        return {**out, "mode": "UPI", "payee": _clean_name(bits[1]) if len(bits) > 1 else "",
                "vpa": vpa.lower(), "note": bits[-1] if len(bits) > 4 else ""}
    nb = [p.strip() for p in re.split(r"[/*]", d)]
    nb_head = re.split(r"[\s\-:]", nb[0].upper())[0] if nb else ""
    if nb_head in {"NEFT", "IMPS", "RTGS"} and len(nb) >= 3:
        # NEFT/CR/ref/NAME/..., NEFT*IFSC*ref*NAME*..., IMPS/P2A/ref/NAME/...
        name = _pick_name(nb[1:])
        return {**out, "mode": nb_head, "payee": _clean_name(name),
                "note": "/".join(nb[nb.index(name) + 1:]) if name in nb else ""}
    if head in {"ACH", "NACH"} and len(parts) >= 4:
        return {**out, "mode": "ACH", "payee": _clean_name(parts[3]), "note": "/".join(parts[4:])}
    if head == "POS" and len(parts) >= 3:
        return {**out, "mode": "CARD", "payee": _clean_name(parts[2])}
    if head.startswith("ATM") or "CASH WDL" in up or "ATW" in head:
        return {**out, "mode": "ATM", "payee": "ATM CASH"}
    if up.startswith(("INT.PD", "INT PD", "INTEREST", "SB INT", "CREDIT INTEREST")):
        return {**out, "mode": "INT", "payee": "BANK INTEREST"}
    return {**out, "payee": _clean_name(parts[0])[:40] if parts else ""}


# =================================================================== 6. verify
def check_running_balance(df: pd.DataFrame, tol: float = 0.01) -> dict:
    """Self-check: previous balance +/- amount should equal this row's balance."""
    if "balance" not in df or df["balance"].isna().all():
        return {"checked": False, "reason": "no balance column"}
    bal = df["balance"].to_numpy()
    signed = df["amount"].where(df["type"] == "CREDIT", -df["amount"]).to_numpy()
    bad = [str(df["txn_id"].iloc[i]) for i in range(1, len(df))
           if pd.notna(bal[i]) and pd.notna(bal[i - 1]) and abs(bal[i - 1] + signed[i] - bal[i]) > tol]
    return {"checked": True, "rows_checked": len(df) - 1, "mismatches": len(bad), "mismatch_ids": bad[:10],
            "opening_balance": round(float(bal[0] - signed[0]), 2) if pd.notna(bal[0]) else None,
            "closing_balance": round(float(bal[-1]), 2) if pd.notna(bal[-1]) else None}


# =================================================================== main entry point
def parse_statement(data: bytes, filename: str,
                    mapper: Callable[[str], ColumnMapping] | None = None,
                    allow_llm: bool = True) -> ParseResult:
    """Parse an uploaded statement. `mapper` lets tests replace the LLM call."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        rows, fmt = read_pdf_rows(data), "pdf"
    elif ext in {".csv", ".txt", ".tsv"}:
        rows, fmt = read_csv_rows(data), "csv"
    else:
        raise ParseError(f"Unsupported file type '{ext}'. Upload a CSV or PDF statement.")
    if not rows:
        raise ParseError("The file is empty.")

    date_fmt = None
    method = "rules"
    detected = detect_layout_rules(rows)
    if detected:
        header_idx, cols = detected
    elif allow_llm:
        header_idx, cols, date_fmt = detect_layout_llm(rows, mapper)
        method = "llm_mapping"
    else:
        raise ParseError("The statement layout was not recognised.")

    header = rows[header_idx]
    idx = {fld: header.index(name) for fld, name in cols.items()}
    get = lambda r, f: r[idx[f]] if f in idx and idx[f] < len(r) else ""  # noqa: E731

    records, dropped = [], {"not_a_transaction": 0, "failed_or_pending": 0, "no_amount": 0}
    for order, r in enumerate(rows[header_idx + 1:]):
        if len(r) < 2 or [c.strip() for c in r] == header:
            dropped["not_a_transaction"] += 1
            continue
        when = parse_date(get(r, "date"), date_fmt)
        if when is None:  # summary lines, repeated headers, blank rows
            dropped["not_a_transaction"] += 1
            continue
        if "status" in idx and _norm(get(r, "status")) not in OK_STATUSES:
            dropped["failed_or_pending"] += 1
            continue
        if "time" in idx and get(r, "time"):
            t = parse_date(get(r, "time"), "%I:%M %p") or parse_date(get(r, "time"), "%H:%M:%S") \
                or parse_date(get(r, "time"), "%H:%M")
            if t:
                when = when.replace(hour=t.hour, minute=t.minute, second=t.second)

        debit, _ = parse_amount(get(r, "debit")) if "debit" in idx else (None, None)
        credit, _ = parse_amount(get(r, "credit")) if "credit" in idx else (None, None)
        amount, ttype = None, None
        if debit:
            amount, ttype = debit, "DEBIT"
        elif credit:
            amount, ttype = credit, "CREDIT"
        elif "amount" in idx:
            amount, hint = parse_amount(get(r, "amount"))
            ttype = _type_from_text(get(r, "type")) if "type" in idx else None
            ttype = ttype or hint or _type_from_text(get(r, "description").split(" ")[0])
            ttype = ttype or "DEBIT"
        if not amount:
            dropped["no_amount"] += 1
            continue

        desc = get(r, "description")
        bal, _ = parse_amount(get(r, "balance")) if "balance" in idx else (None, None)
        records.append({"date": when, "description": desc, "type": ttype, "amount": amount,
                        "balance": bal, "ref": get(r, "ref"), "_order": order, **extract_details(desc)})

    if not records:
        raise ParseError("No transactions found in the file.")

    df = pd.DataFrame(records)
    # UPI apps list newest first; put everything in date order (stable within a day)
    if df["date"].iloc[0] > df["date"].iloc[-1]:
        df = df.iloc[::-1]
    df = df.sort_values(["date"], kind="stable").reset_index(drop=True)
    df["txn_id"] = [f"T{i:04d}" for i in range(1, len(df) + 1)]
    df = df[OUTPUT_COLUMNS]

    result = ParseResult(df=df, source_format=fmt, method=method, rows_read=len(rows) - header_idx - 1,
                         mapping={"header_row_index": header_idx, **cols, "date_format": date_fmt},
                         dropped={k: v for k, v in dropped.items() if v})
    result.balance_check = check_running_balance(df)
    if result.balance_check.get("mismatches"):
        result.warnings.append(f"{result.balance_check['mismatches']} rows don't match the running balance; "
                               "some transactions may be missing or misread.")
    if dropped["failed_or_pending"]:
        result.warnings.append(f"Skipped {dropped['failed_or_pending']} failed or pending transactions.")
    return result


def parse_file(path: str | Path, **kwargs) -> ParseResult:
    path = Path(path)
    return parse_statement(path.read_bytes(), path.name, **kwargs)
