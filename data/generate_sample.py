"""Synthetic Indian bank + UPI statement generator for SpendSense.

Creates 4 months (Jul-Oct 2026) of realistic transactions for a fictional salaried
person in Chennai, with a deliberate overspending spike in September (festival sale,
lots of food delivery, many small UPI payments) so the Advisor's alerts have
something to find.

Outputs (in data/samples/):
  bank_statement.csv   bank-style export: preamble lines, dd/mm/yyyy dates,
                       Indian lakh-format amounts ("1,23,456.00"), Debit/Credit columns
  upi_statement.csv    UPI-app-style export of the same account's UPI payments:
                       different column names, "12 Jul 2026" dates, "₹" amounts,
                       a few FAILED rows the parser must drop
  bank_statement.pdf   the bank statement as a ruled PDF table (multi-page)
  unusual_layout_oct.csv  October only, in a layout the rules don't know (';', odd
                       headers, D/C flag) to demo the Parser's LLM fallback
  ground_truth.csv     every transaction with its true category, to score the
                       Parser and Categorizer later

All names, account numbers, VPAs and reference numbers are made up.
Deterministic: the same seed always produces the same files.

Run:  .venv/bin/python data/generate_sample.py
"""

from __future__ import annotations

import calendar
import csv
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "samples"
MONTHS = [(2026, 7), (2026, 8), (2026, 9), (2026, 10)]
SPIKE_MONTH = (2026, 9)
OPENING_BALANCE = 85_000.00

HOLDER = "Arjun Mehta"
ACCOUNT_NO = "XXXXXXXX4821"
BANK_NAME = "SpendSense Demo Bank"
BRANCH = "Anna Nagar, Chennai"
EMPLOYER = "ACME TECHNOLOGIES PVT LTD"

rng = random.Random(SEED)


@dataclass
class Txn:
    when: datetime
    description: str        # as the bank prints it
    amount: float
    type: str               # "DEBIT" | "CREDIT"
    category: str           # ground-truth category
    payee: str              # clean merchant / person name
    mode: str               # UPI | NEFT | ACH | CARD | ATM | INT
    ref: str = ""
    upi_status: str = "SUCCESS"
    balance: float = 0.0
    txn_id: str = field(default="")


# ---------------------------------------------------------------- merchants
# (payee name, VPA, typical note) grouped by category
FOOD = [("SWIGGY", "swiggy.order@ybl", "Food order"), ("ZOMATO", "zomato.pay@hdfc", "Food order"),
        ("SARAVANA BHAVAN", "saravanabhavan@okaxis", "Lunch"), ("DOMINOS PIZZA", "dominos.in@icici", "Pizza"),
        ("STARBUCKS", "starbucks.tata@okicici", "Coffee"), ("A2B ADYAR ANANDA BHAVAN", "a2b.annanagar@ybl", "Snacks")]
GROCERY = [("BLINKIT", "blinkit.grofers@hdfc", "Groceries"), ("ZEPTO", "zepto.kirana@ybl", "Groceries"),
           ("BIGBASKET", "bigbasket.bb@icici", "Monthly groceries"), ("NILGIRIS SUPERMARKET", "nilgiris.an@okaxis", "Groceries")]
TRANSPORT = [("UBER", "uber.india@axis", "Ride"), ("OLA", "olacabs.pay@icici", "Ride"),
             ("RAPIDO", "rapido.bike@ybl", "Bike ride"), ("CHENNAI METRO RAIL", "cmrl.travel@sbi", "Metro card recharge")]
SMALL = [("SRI MURUGAN TEA STALL", "murugantea98@okaxis", "Tea"), ("KUMAR FRUITS", "kumarfruits@ybl", "Fruits"),
         ("SRI BALAJI BAKERY", "balajibakery77@oksbi", "Snacks"), ("ANNA NAGAR PARKING", "annanagar.park@paytm", "Parking"),
         ("LAKSHMI FLOWERS", "lakshmiflowers@okicici", "Flowers"), ("RAJA JUICE CORNER", "rajajuice21@ybl", "Juice")]
SHOPPING = [("AMAZON", "amazon.pay@apl", "Order"), ("FLIPKART", "flipkart.pay@axisbank", "Order"),
            ("MYNTRA", "myntra.fashion@icici", "Clothes"), ("DECATHLON", "decathlon.in@hdfc", "Sports gear"),
            ("CROMA", "croma.retail@ybl", "Electronics")]
ENTERTAINMENT = [("BOOKMYSHOW", "bookmyshow@icici", "Movie tickets"), ("PVR CINEMAS", "pvr.cinemas@hdfc", "Movie"),
                 ("STEAM GAMES", "steam.valve@ybl", "Game")]
HEALTH = [("APOLLO PHARMACY", "apollopharmacy@icici", "Medicines"), ("MEDPLUS", "medplus.mart@ybl", "Medicines"),
          ("DR RAO CLINIC", "drraoclinic@okaxis", "Consultation"), ("CULT FIT", "cultfit.gym@hdfc", "Gym membership")]
FRIENDS = [("RAHUL SHARMA", "rahul.sharma92@okaxis"), ("PRIYA NAIR", "priyanair@ybl"),
           ("KARTHIK SUBRAMANIAN", "karthik.s@oksbi"), ("SNEHA IYER", "sneha.iyer@okicici"),
           ("VIKRAM REDDY", "vikramreddy@paytm")]
FRIEND_NOTES_OUT = ["Dinner split", "Trip share", "Movie tickets", "Gift contribution", "Cab share", "Returned"]
FRIEND_NOTES_IN = ["Dinner split", "Trip share", "Paying back", "Cab share"]
BANK_SUFFIX = ["SBIN", "HDFC", "ICIC", "UTIB", "KKBK", "YESB"]


# ---------------------------------------------------------------- helpers
def ref12() -> str:
    return str(rng.randint(10**11, 10**12 - 1))


def at(d: date, start_h: int = 7, end_h: int = 23) -> datetime:
    return datetime.combine(d, time(rng.randint(start_h, end_h - 1), rng.randint(0, 59), rng.randint(0, 59)))


def rand_day(y: int, m: int, lo: int = 1, hi: int | None = None) -> date:
    hi = hi or calendar.monthrange(y, m)[1]
    return date(y, m, rng.randint(lo, hi))


def amt(lo: float, hi: float, step: float = 1.0) -> float:
    return round(round(rng.uniform(lo, hi) / step) * step, 2)


def first_weekday(y: int, m: int) -> date:
    d = date(y, m, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def upi_txn(when, payee, vpa, note, amount, category, credit=False) -> Txn:
    r = ref12()
    flow = "CR" if credit else "DR"
    desc = f"UPI/{flow}/{r}/{payee}/{rng.choice(BANK_SUFFIX)}/{vpa}/{note}"
    return Txn(when, desc, amount, "CREDIT" if credit else "DEBIT", category, payee, "UPI", r)


def pick_upi(merchants, d, lo, hi, category, step=1.0, start_h=7, end_h=23) -> Txn:
    payee, vpa, note = rng.choice(merchants)
    return upi_txn(at(d, start_h, end_h), payee, vpa, note, amt(lo, hi, step), category)


# ---------------------------------------------------------------- generation
def month_txns(y: int, m: int) -> list[Txn]:
    spike = (y, m) == SPIKE_MONTH
    mon = calendar.month_abbr[m].upper()
    out: list[Txn] = []

    # --- fixed monthly items
    d = first_weekday(y, m)
    r = "N" + ref12()
    out.append(Txn(at(d, 9, 11), f"NEFT/CR/{r}/{EMPLOYER}/SALARY {mon} {y}", 78_500.00, "CREDIT",
                   "Salary & Income", EMPLOYER, "NEFT", r))
    out.append(upi_txn(at(date(y, m, 3), 8, 12), "RAMESH KUMAR", "rameshk1968@oksbi", f"RENT {mon}",
                       22_000.00, "Rent"))
    r = ref12()
    out.append(Txn(at(date(y, m, 5), 6, 8), f"ACH/DR/{r}/SURAKSHA FINANCE LTD/BIKE LOAN EMI", 6_450.00, "DEBIT",
                   "EMI & Loans", "SURAKSHA FINANCE LTD", "ACH", r))
    r = ref12()
    out.append(Txn(at(date(y, m, 10), 6, 8), f"ACH/DR/{r}/BSE STAR MF/SIP NIFTY INDEX FUND", 5_000.00, "DEBIT",
                   "Investments", "BSE STAR MF", "ACH", r))

    # --- bills & subscriptions
    power = amt(1_700, 2_400) if m in (7, 8) else amt(1_000, 1_500)  # AC season costs more
    out.append(upi_txn(at(rand_day(y, m, 11, 14)), "TN ELECTRICITY BOARD", "tneb.billpay@sbi",
                       "Electricity bill", power, "Bills & Utilities"))
    out.append(upi_txn(at(rand_day(y, m, 14, 16)), "AIRTEL", "airtel.prepaid@airtel", "Mobile recharge",
                       349.00, "Bills & Utilities"))
    out.append(upi_txn(at(rand_day(y, m, 17, 19)), "ACT FIBERNET", "actfibernet@icici", "Broadband bill",
                       799.00, "Bills & Utilities"))
    out.append(upi_txn(at(rand_day(y, m, 20, 22)), "INDANE GAS", "indanegas@sbi", "LPG cylinder",
                       905.00, "Bills & Utilities")) if m % 2 == 1 else None
    r = ref12()
    out.append(Txn(at(date(y, m, 8), 1, 3), f"POS/{r}/NETFLIX.COM/MUMBAI", 649.00, "DEBIT",
                   "Entertainment", "NETFLIX", "CARD", r))
    out.append(upi_txn(at(date(y, m, 12), 1, 3), "SPOTIFY INDIA", "spotify.autopay@hdfc", "AutoPay mandate",
                       119.00, "Entertainment"))
    if m >= 8:  # new subscription that starts in August -> subscription creep
        out.append(upi_txn(at(date(y, m, 18), 1, 3), "GOOGLE YOUTUBE PREMIUM", "youtube.google@okaxis",
                           "AutoPay mandate", 129.00, "Entertainment"))

    # --- variable spending (counts jump in the spike month)
    for _ in range(30 if spike else rng.randint(11, 14)):
        out.append(pick_upi(FOOD, rand_day(y, m), 150, 750, "Food & Dining", start_h=11))
    for _ in range(rng.randint(5, 7)):
        out.append(pick_upi(GROCERY, rand_day(y, m), 200, 1_200, "Groceries"))
    r = ref12()
    out.append(Txn(at(rand_day(y, m, 1, 10)), f"POS/{r}/DMART AVENUE SUPERMARTS/CHENNAI",
                   amt(1_500, 2_500), "DEBIT", "Groceries", "DMART", "CARD", r))
    for _ in range(rng.randint(10, 14)):
        out.append(pick_upi(TRANSPORT, rand_day(y, m), 60, 420, "Transport", start_h=7))
    for _ in range(2):
        r = ref12()
        out.append(Txn(at(rand_day(y, m)), f"POS/{r}/HP PETROL PUMP KOYAMBEDU/CHENNAI",
                       amt(500, 1_200, 50), "DEBIT", "Transport", "HP PETROL PUMP", "CARD", r))
    for _ in range(34 if spike else rng.randint(14, 20)):
        out.append(pick_upi(SMALL, rand_day(y, m), 10, 150, "Others", step=5, start_h=7, end_h=21))
    for _ in range(9 if spike else rng.randint(1, 3)):
        out.append(pick_upi(SHOPPING, rand_day(y, m, 20 if spike else 1, 30 if spike else None),
                            499, 6_999 if spike else 3_499, "Shopping", step=1))
    for _ in range(4 if spike else rng.randint(1, 2)):
        out.append(pick_upi(ENTERTAINMENT, rand_day(y, m), 250, 950, "Entertainment", start_h=10))
    for _ in range(rng.randint(1, 2)):
        out.append(pick_upi(HEALTH, rand_day(y, m), 180, 1_500, "Health"))

    # --- education (one-offs)
    if m == 8:
        out.append(upi_txn(at(date(y, m, 23)), "UDEMY", "udemy.courses@icici", "Python course", 499.00, "Education"))
    if m == 10:
        out.append(upi_txn(at(date(y, m, 6)), "COURSERA", "coursera.inc@hdfc", "ML specialization",
                           3_999.00, "Education"))

    # --- person-to-person transfers
    for _ in range(rng.randint(2, 4)):
        name, vpa = rng.choice(FRIENDS)
        out.append(upi_txn(at(rand_day(y, m)), name, vpa, rng.choice(FRIEND_NOTES_OUT),
                           amt(200, 2_000, 50), "Transfers to People"))
    for _ in range(rng.randint(1, 3)):
        name, vpa = rng.choice(FRIENDS)
        out.append(upi_txn(at(rand_day(y, m)), name, vpa, rng.choice(FRIEND_NOTES_IN),
                           amt(200, 2_000, 50), "Transfers to People", credit=True))

    # --- cash
    for _ in range(2 if spike else 1):
        r = ref12()
        out.append(Txn(at(rand_day(y, m)), f"ATM/CASH WDL/{r}/ANNA NAGAR W CHENNAI",
                       float(rng.choice([2_000, 3_000, 5_000])), "DEBIT", "Others", "ATM CASH", "ATM", r))

    # --- occasional extras
    if m == 8:  # refund of an Amazon order
        r = ref12()
        out.append(Txn(at(date(y, m, 14)), f"UPI/CR/{r}/AMAZON/UTIB/amazon.pay@apl/Refund",
                       1_299.00, "CREDIT", "Shopping", "AMAZON", "UPI", r))
    if m == 9:  # quarterly savings interest
        out.append(Txn(datetime(y, 9, 30, 23, 59, 0), "INT.PD:01-07-2026 TO 30-09-2026", 412.00, "CREDIT",
                       "Salary & Income", "BANK INTEREST", "INT"))
    if m == 10:  # Diwali bonus
        r = "N" + ref12()
        out.append(Txn(at(date(y, m, 15), 10, 12), f"NEFT/CR/{r}/{EMPLOYER}/DIWALI BONUS", 15_000.00, "CREDIT",
                       "Salary & Income", EMPLOYER, "NEFT", r))
    return out


def failed_upi_attempts(txns: list[Txn]) -> list[Txn]:
    """A few failed UPI attempts. They appear ONLY in the UPI app export (banks don't print them)."""
    picks = rng.sample([t for t in txns if t.mode == "UPI" and t.type == "DEBIT" and t.category == "Food & Dining"], 3)
    out = []
    for t in picks:
        f = Txn(t.when - timedelta(minutes=2), t.description, t.amount, "DEBIT", t.category, t.payee, "UPI",
                ref12(), upi_status="FAILED")
        out.append(f)
    return out


def generate() -> tuple[list[Txn], list[Txn]]:
    txns = [t for y, m in MONTHS for t in month_txns(y, m)]
    txns.sort(key=lambda t: t.when)
    bal = OPENING_BALANCE
    for i, t in enumerate(txns, 1):
        bal += t.amount if t.type == "CREDIT" else -t.amount
        t.balance = round(bal, 2)
        t.txn_id = f"T{i:04d}"
    if min(t.balance for t in txns) < 0:
        raise ValueError("Balance went negative; adjust the generator")
    return txns, failed_upi_attempts(txns)


# ---------------------------------------------------------------- formatting
def inr(x: float) -> str:
    """Indian digit grouping: 123456.5 -> '1,23,456.50'."""
    neg = x < 0
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if neg else "") + whole + "." + frac


def period_str() -> tuple[str, str]:
    (y0, m0), (y1, m1) = MONTHS[0], MONTHS[-1]
    return date(y0, m0, 1).strftime("%d/%m/%Y"), date(y1, m1, calendar.monthrange(y1, m1)[1]).strftime("%d/%m/%Y")


# ---------------------------------------------------------------- writers
def write_bank_csv(txns: list[Txn], path: Path) -> None:
    start, end = period_str()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{BANK_NAME} - Savings Account Statement"])
        w.writerow([f"Account Holder: {HOLDER}"])
        w.writerow([f"Account Number: {ACCOUNT_NO}", f"Branch: {BRANCH}"])
        w.writerow([f"Statement Period: {start} to {end}", f"Opening Balance: {inr(OPENING_BALANCE)}"])
        w.writerow([])
        w.writerow(["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"])
        for t in txns:
            d = t.when.strftime("%d/%m/%Y")
            w.writerow([d, d, t.description, t.ref,
                        inr(t.amount) if t.type == "DEBIT" else "",
                        inr(t.amount) if t.type == "CREDIT" else "",
                        inr(t.balance)])


def write_upi_csv(txns: list[Txn], failed: list[Txn], path: Path) -> None:
    rows = sorted([t for t in txns if t.mode == "UPI"] + failed, key=lambda t: t.when, reverse=True)  # apps list newest first
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Time", "Transaction Details", "Type", "Amount (INR)", "UPI Transaction ID", "Status"])
        for t in rows:
            who = t.payee.title()
            details = f"Received from {who}" if t.type == "CREDIT" else f"Paid to {who}"
            w.writerow([t.when.strftime("%d %b %Y"), t.when.strftime("%I:%M %p"), details, t.type,
                        f"₹{t.amount:,.2f}", t.ref, t.upi_status])


def write_unusual_csv(txns: list[Txn], path: Path) -> None:
    """October only, in a layout the rule-based parser does NOT know: ';' separated, odd header
    names, one amount column with a D/C flag. Used to demo the Parser Agent's LLM fallback."""
    rows = [t for t in txns if t.when.month == 10]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([f"Acct {ACCOUNT_NO} :: e-statement :: Oct 2026"])
        w.writerow(["Posting Dt", "Txn Remarks", "Amt", "D/C", "Bal"])
        for t in rows:
            w.writerow([t.when.strftime("%d-%m-%Y"), t.description, f"{t.amount:.2f}",
                        "D" if t.type == "DEBIT" else "C", f"{t.balance:.2f}"])


def write_ground_truth(txns: list[Txn], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["txn_id", "datetime", "description", "payee", "mode", "type", "amount", "balance", "category"])
        for t in txns:
            w.writerow([t.txn_id, t.when.isoformat(sep=" "), t.description, t.payee, t.mode, t.type,
                        f"{t.amount:.2f}", f"{t.balance:.2f}", t.category])


def write_pdf(txns: list[Txn], path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7, leading=8.5)
    start, end = period_str()
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Account Statement")
    story = [
        Paragraph(f"<b>{BANK_NAME}</b>, Savings Account Statement", styles["Title"]),
        Paragraph(f"Account Holder: {HOLDER} &nbsp;&nbsp; Account Number: {ACCOUNT_NO} &nbsp;&nbsp; "
                  f"Branch: {BRANCH}", styles["Normal"]),
        Paragraph(f"Statement Period: {start} to {end} &nbsp;&nbsp; Opening Balance: INR {inr(OPENING_BALANCE)}",
                  styles["Normal"]),
        Spacer(1, 5 * mm),
    ]
    data = [["Txn Date", "Description", "Ref No.", "Debit", "Credit", "Balance"]]
    for t in txns:
        data.append([t.when.strftime("%d/%m/%Y"), Paragraph(t.description, cell), t.ref,
                     inr(t.amount) if t.type == "DEBIT" else "",
                     inr(t.amount) if t.type == "CREDIT" else "", inr(t.balance)])
    table = LongTable(data, colWidths=[22 * mm, 140 * mm, 30 * mm, 25 * mm, 25 * mm, 28 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
    ]))
    story += [table, Spacer(1, 5 * mm),
              Paragraph(f"Closing Balance: INR {inr(txns[-1].balance)} &nbsp;&nbsp; "
                        f"Total transactions: {len(txns)}", styles["Normal"]),
              Paragraph("This is a synthetic statement generated for a college project. "
                        "All names and numbers are fictional.", styles["Italic"])]
    doc.build(story)


# ---------------------------------------------------------------- main
def summarize(txns: list[Txn], failed: list[Txn]) -> None:
    print(f"Generated {len(txns)} transactions ({len(failed)} extra FAILED UPI rows in the UPI export only)")
    print(f"Opening INR {inr(OPENING_BALANCE)}  ->  closing INR {inr(txns[-1].balance)}")
    print(f"\n{'Month':<9}{'Income':>12}{'Spent':>12}{'Saved':>12}{'Rate':>8}{'Txns':>6}")
    for y, m in MONTHS:
        mt = [t for t in txns if (t.when.year, t.when.month) == (y, m)]
        inc = sum(t.amount for t in mt if t.type == "CREDIT")
        spent = sum(t.amount for t in mt if t.type == "DEBIT")
        tag = "  <- spike" if (y, m) == SPIKE_MONTH else ""
        print(f"{calendar.month_abbr[m]} {y}{inr(inc):>12}{inr(spent):>12}{inr(inc - spent):>12}"
              f"{(inc - spent) / inc * 100:>7.1f}%{len(mt):>6}{tag}")
    print("\nDebit spend by category (all months):")
    cats: dict[str, float] = {}
    for t in txns:
        if t.type == "DEBIT":
            cats[t.category] = cats.get(t.category, 0) + t.amount
    for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<22}{inr(v):>12}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txns, failed = generate()
    write_bank_csv(txns, OUT_DIR / "bank_statement.csv")
    write_upi_csv(txns, failed, OUT_DIR / "upi_statement.csv")
    write_ground_truth(txns, OUT_DIR / "ground_truth.csv")
    write_pdf(txns, OUT_DIR / "bank_statement.pdf")
    write_unusual_csv(txns, OUT_DIR / "unusual_layout_oct.csv")
    summarize(txns, failed)
    print(f"\nFiles written to {OUT_DIR}")


if __name__ == "__main__":
    main()
