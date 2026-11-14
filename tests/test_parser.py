"""Parser Agent tests against the generator's ground truth. No LLM calls (the fallback uses a fake mapper)."""

from pathlib import Path

import pandas as pd
import pytest

from spendsense.agents.parser_agent import (
    ParseError, extract_details, parse_amount, parse_file, parse_statement,
)
from spendsense.schemas import ColumnMapping

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def truth() -> pd.DataFrame:
    gt = pd.read_csv(SAMPLES / "ground_truth.csv", dtype={"txn_id": str})
    gt["datetime"] = pd.to_datetime(gt["datetime"])
    return gt


@pytest.mark.parametrize("name", ["bank_statement.csv", "bank_statement.pdf"])
def test_bank_formats_match_ground_truth(name, truth):
    res = parse_file(SAMPLES / name, allow_llm=False)
    df = res.df
    assert res.method == "rules"
    assert len(df) == len(truth)
    assert (df["date"].dt.date.values == truth["datetime"].dt.date.values).all()
    assert (df["amount"].values == truth["amount"].values).all()
    assert (df["type"].values == truth["type"].values).all()
    assert (df["description"].values == truth["description"].values).all()
    assert (df["mode"].values == truth["mode"].values).all()
    # payee: generator's short name must appear in the parsed one (POS names carry the city/branch)
    assert all(t.replace(".COM", "") in p for t, p in zip(truth["payee"], df["payee"]))
    assert res.balance_check["mismatches"] == 0
    assert res.balance_check["opening_balance"] == 85_000.00


def test_upi_app_export(truth):
    res = parse_file(SAMPLES / "upi_statement.csv", allow_llm=False)
    df = res.df
    upi = truth[truth["mode"] == "UPI"]
    assert res.dropped.get("failed_or_pending") == 3
    assert len(df) == len(upi)
    merged = df.merge(upi, left_on="ref", right_on=upi["description"].str.split("/").str[2], how="inner")
    assert len(merged) == len(upi)
    assert (merged["amount_x"] == merged["amount_y"]).all()
    assert (merged["type_x"] == merged["type_y"]).all()
    assert (merged["payee_x"] == merged["payee_y"]).all()
    assert df["date"].is_monotonic_increasing


def test_unknown_layout_uses_llm_mapping():
    csv_text = (
        "Statement of account\n"
        "Posting Dt;Txn Remarks;Amt;Dr / Cr ;Bal\n"
        "05-07-2026;ACH/DR/1/SURAKSHA FINANCE LTD/BIKE LOAN EMI;6450.00;D;44784.10\n"
        "06-07-2026;NEFT/CR/2/ACME TECHNOLOGIES PVT LTD/SALARY;78500.00;C;123284.10\n"
    )
    calls = []

    def fake_mapper(sample: str) -> ColumnMapping:
        calls.append(sample)
        return ColumnMapping(header_row_index=1, date_col="Posting Dt", description_col="Txn Remarks",
                             amount_col="Amt", type_col="Dr / Cr", balance_col="Bal", date_format="%d-%m-%Y")

    res = parse_statement(csv_text.encode(), "odd.csv", mapper=fake_mapper)
    assert calls and res.method == "llm_mapping"
    assert list(res.df["type"]) == ["DEBIT", "CREDIT"]
    assert list(res.df["payee"]) == ["SURAKSHA FINANCE LTD", "ACME TECHNOLOGIES PVT LTD"]
    assert res.balance_check["mismatches"] == 0


def test_llm_mapping_with_wrong_column_is_rejected():
    def bad_mapper(_):
        return ColumnMapping(header_row_index=0, date_col="Nope", description_col="X", amount_col="Y",
                             date_format="%d/%m/%Y")
    with pytest.raises(ParseError):
        parse_statement(b"a,b,c\n1,2,3\n", "x.csv", mapper=bad_mapper)


@pytest.mark.parametrize("raw,expected", [
    ("₹1,23,456.00", (123456.0, None)), ("1,299.50 Dr", (1299.5, "DEBIT")), ("500 CR", (500.0, "CREDIT")),
    ("(250.00)", (250.0, "DEBIT")), ("-75", (75.0, "DEBIT")), ("", (None, None)), ("--", (None, None)),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("desc,payee,mode", [
    ("UPI/DR/123/SWIGGY/YESB/swiggy.order@ybl/Food order", "SWIGGY", "UPI"),
    ("UPI-ZOMATO-zomato@hdfc-HDFC0000-1234-Dinner", "ZOMATO", "UPI"),
    ("POS/99/NETFLIX.COM/MUMBAI", "NETFLIX", "CARD"),
    ("ATM/CASH WDL/1/ANNA NAGAR", "ATM CASH", "ATM"),
    ("Received from Priya Nair", "PRIYA NAIR", "UPI"),
    ("INT.PD:01-07-2026 TO 30-09-2026", "BANK INTEREST", "INT"),
])
def test_extract_details(desc, payee, mode):
    d = extract_details(desc)
    assert (d["payee"], d["mode"]) == (payee, mode)


@pytest.mark.parametrize("desc,payee,mode", [
    # SBI-style prefixes: the payee used to come out as "DEP TFR UPI" / "WDL TFR UPI"
    ("DEP TFR UPI/CR/412345678901/RAHUL K/KKBK/rahul@okhdfc/UPI", "RAHUL K", "UPI"),
    ("WDL TFR UPI/DR/412345678902/SWIGGY/YESB/swiggy@ybl/Payment", "SWIGGY", "UPI"),
    ("DEP TFR  UPI/CR/412345678903/PRIYA  N/SBIN/priya@oksbi/UPI  ", "PRIYA N", "UPI"),
    ("TO TRANSFER-UPI/DR/412345678904/ZOMATO LTD/HDFC/zomato@hdfcbank/Order-", "ZOMATO LTD", "UPI"),
    ("BY TRANSFER-NEFT*HDFC0000001*N123456789012*ACME TECH PVT LTD*SALARY", "ACME TECH PVT LTD", "NEFT"),
    ("DEP TFR NEFT*ICIC0000123*CMS1234567*EMPLOYER NAME", "EMPLOYER NAME", "NEFT"),
    ("WDL TFR INB IMPS/P2A/412345678905/KARTHIK S/SBIN/rent", "KARTHIK S", "IMPS"),
    ("UPI/412345678906/MEDPLUS/medplus@ybl/Medicines", "MEDPLUS", "UPI"),
])
def test_prefixed_bank_formats(desc, payee, mode):
    d = extract_details(desc)
    assert (d["payee"], d["mode"]) == (payee, mode)


def test_prefixed_upi_rows_get_distinct_payees():
    """Two different people must not collapse into one 'DEP TFR UPI' payee."""
    rows = "Txn Date,Description,Debit,Credit,Balance\n" + "\n".join([
        "01/10/2026,DEP TFR UPI/CR/400000000001/RAHUL K/KKBK/rahul@okhdfc/UPI,,500.00,10500.00",
        "02/10/2026,DEP TFR UPI/CR/400000000002/PRIYA N/SBIN/priya@oksbi/UPI,,700.00,11200.00",
        "03/10/2026,WDL TFR UPI/DR/400000000003/SWIGGY/YESB/swiggy@ybl/Food,300.00,,10900.00",
    ])
    df = parse_statement(rows.encode(), "sbi.csv", allow_llm=False).df
    assert list(df["payee"]) == ["RAHUL K", "PRIYA N", "SWIGGY"]
    assert set(df["mode"]) == {"UPI"}


def test_upi_note_and_vpa():
    d = extract_details("UPI/DR/1/RAMESH KUMAR/SBIN/rameshk1968@oksbi/RENT JUL")
    assert d["vpa"] == "rameshk1968@oksbi" and d["note"] == "RENT JUL"


def test_unsupported_extension():
    with pytest.raises(ParseError):
        parse_statement(b"x", "statement.xlsx")
