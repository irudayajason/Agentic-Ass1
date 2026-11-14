"""Central settings: paths, model names, categories and alert thresholds."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- Paths ---
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CHROMA_DIR = ROOT_DIR / "chroma_db"
KB_DIR = Path(__file__).resolve().parent / "rag" / "knowledge_base"
FRONTEND_DIR = ROOT_DIR / "frontend"

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "openai/gpt-oss-20b")
# Third model with its own free-tier quota, used only when both models above are rate-limited
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = 0.1

# --- Embeddings / vector DB ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EXAMPLES_COLLECTION = "labelled_examples"
TXN_COLLECTION = "transactions"
KB_COLLECTION = "finance_kb"

# --- Spending categories ---
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Shopping",
    "Transport",
    "Bills & Utilities",
    "Rent",
    "EMI & Loans",
    "Entertainment",
    "Health",
    "Education",
    "Investments",
    "Transfers to People",
    "Salary & Income",
    "Others",
]

# --- Advisor alert thresholds ---
MOM_SPIKE_PCT = 30.0          # category up >30% vs previous month
LOW_SAVINGS_RATE_PCT = 20.0   # savings rate below 20% of income
SMALL_UPI_AMOUNT = 200.0      # "small" UPI payment, in rupees
SMALL_UPI_COUNT = 25          # this many small payments in a month triggers an alert
# 50/30/20 rule: needs / wants / savings share of income
BUDGET_SPLIT = {"needs": 50, "wants": 30, "savings": 20}
NEEDS_CATEGORIES = {"Rent", "Groceries", "Bills & Utilities", "EMI & Loans", "Transport", "Health", "Education"}
WANTS_CATEGORIES = {"Food & Dining", "Shopping", "Entertainment", "Others", "Transfers to People"}
SAVINGS_CATEGORIES = {"Investments"}
MIN_ALERT_AMOUNT = 2000.0     # ignore month-over-month spikes smaller than this many rupees
EMI_RATIO_LIMIT_PCT = 30.0
CASH_SHARE_LIMIT_PCT = 10.0
FOOD_ORDERS_PER_MONTH = 20
