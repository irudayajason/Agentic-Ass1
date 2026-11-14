"""LCEL chain builders: prompt | model | output parser.

Every agent builds its chains through these helpers, so the LangChain Expression Language
pattern is the same everywhere:

    text chain:        build_prompt(key) | llm | StrOutputParser()
    structured chain:  build_prompt(key) | llm.with_structured_output(Schema)

Both get an automatic backup via LCEL's `.with_fallbacks()`: if the chosen Groq model is
rate-limited (per-minute or per-day quota) or unreachable, the same prompt runs on the other
model. Structured chains also fall back on bad JSON.
"""

from __future__ import annotations

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from spendsense.llm import backups_for, get_llm, rate_limit_errors
from spendsense.prompts import build_prompt


def text_chain(prompt_key: str, llm: BaseChatModel | None = None) -> Runnable:
    """prompt -> LLM -> plain string (falls back to the other model when rate-limited)."""
    llm = llm or get_llm()
    chain = build_prompt(prompt_key) | llm | StrOutputParser()
    backups = backups_for(llm)
    if backups:
        chain = chain.with_fallbacks([build_prompt(prompt_key) | b | StrOutputParser() for b in backups],
                                     exceptions_to_handle=rate_limit_errors())
    return chain


def structured_chain(prompt_key: str, schema: Type[BaseModel], llm: BaseChatModel | None = None,
                     fallback_llm: BaseChatModel | None = None) -> Runnable:
    """prompt -> LLM constrained to a JSON schema -> validated Pydantic object.

    Falls back to `fallback_llm` (default: the other Groq model) on any failure, including
    rate limits and Groq's 'json_validate_failed'.
    """
    llm = llm or get_llm()
    chain = build_prompt(prompt_key) | llm.with_structured_output(schema, method="json_schema")
    backups = [fallback_llm] if fallback_llm is not None else []
    backups += [b for b in backups_for(llm) if fallback_llm is None or b.model_name != getattr(fallback_llm, "model_name", None)]
    if backups:
        chain = chain.with_fallbacks([build_prompt(prompt_key) | b.with_structured_output(schema, method="json_schema")
                                      for b in backups])
    return chain
