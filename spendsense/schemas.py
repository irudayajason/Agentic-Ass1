"""Pydantic models for structured LLM output.

Each chain that must return data (not prose) uses one of these with
`llm.with_structured_output(Model, method="json_schema")`, so the LLM's answer is
validated before any code relies on it.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from spendsense.config import CATEGORIES

IntentName = Literal["transaction_lookup", "spending_analysis", "savings_guidance", "general"]
# Literal built from the config list so prompts, schemas and code agree on category names
Category = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


# ---------------------------------------------------------------- router
class Intent(BaseModel):
    """What the user wants, plus any entities the downstream agent can use."""

    reasoning: str = Field(description="One short sentence explaining the choice")
    intent: IntentName
    category: Optional[str] = Field(None, description="Spending category mentioned, if any")
    merchant: Optional[str] = Field(None, description="Merchant or person mentioned, if any")
    period: Optional[str] = Field(None, description="Time period mentioned, e.g. 'September', 'last month'")
    amount: Optional[float] = Field(None, description="Amount in rupees mentioned, if any")


# ---------------------------------------------------------------- parser
class ColumnMapping(BaseModel):
    """How to read an unfamiliar statement layout. Column names must match the header exactly."""

    header_row_index: int = Field(description="0-based line number of the column header row")
    date_col: str
    description_col: str
    debit_col: Optional[str] = Field(None, description="Column with money going out, if separate")
    credit_col: Optional[str] = Field(None, description="Column with money coming in, if separate")
    amount_col: Optional[str] = Field(None, description="Single amount column, if debit/credit are not separate")
    type_col: Optional[str] = Field(None, description="Column saying debit/credit (DR/CR) for a single amount column")
    balance_col: Optional[str] = None
    ref_col: Optional[str] = None
    status_col: Optional[str] = Field(None, description="Column with success/failed status, if any")
    date_format: str = Field(description="Python strptime format of the dates, e.g. '%d/%m/%Y'")


# ---------------------------------------------------------------- categorizer
class CategoryAssignment(BaseModel):
    txn_id: str
    reasoning: str = Field(description="Brief step-by-step reasoning: payee -> business or person -> clue -> category")
    category: Category
    confidence: float = Field(ge=0, le=1)


class CategoryBatch(BaseModel):
    assignments: list[CategoryAssignment]


class ReflectionItem(BaseModel):
    txn_id: str
    original_category: str
    critique: str = Field(description="What could be wrong with the original label")
    final_category: Category
    changed: bool


class ReflectionBatch(BaseModel):
    items: list[ReflectionItem]


# ---------------------------------------------------------------- advisor
class RetrievalPlan(BaseModel):
    """The agent's decision on whether the knowledge base is needed (Agentic RAG)."""

    need_retrieval: bool
    reason: str
    queries: list[str] = Field(default_factory=list, description="1-3 short search queries if retrieval is needed")


class AdviceCritique(BaseModel):
    """Self-critique of a draft answer (Reflection)."""

    grounded_in_numbers: bool = Field(description="Every claim uses the user's actual figures")
    cites_sources: bool = Field(description="Guidance taken from the knowledge base is cited")
    actionable: bool = Field(description="Each suggestion is specific with a rupee amount or a concrete step")
    issues: list[str] = Field(default_factory=list)
    score: int = Field(ge=1, le=5)
