"""Offline checks: every prompt builds and formats, and the catalogue is complete. No LLM calls."""

import pytest

from spendsense.config import CATEGORIES
from spendsense.prompts import PROMPTS, build_prompt, prompt_catalog


@pytest.mark.parametrize("key", list(PROMPTS))
def test_prompt_formats_with_dummy_values(key):
    prompt = build_prompt(key)
    values = {v: f"<{v}>" for v in prompt.input_variables}
    messages = prompt.format_messages(**values)
    assert messages[0].type == "system"
    assert messages[-1].type == "human"
    # no unfilled template braces left behind
    for m in messages:
        assert "{categories}" not in m.content


def test_categories_injected():
    text = build_prompt("categorizer").format_messages(examples="-", transactions="-")[0].content
    assert all(c in text for c in CATEGORIES)


def test_router_has_few_shot_examples_between_system_and_user():
    msgs = build_prompt("router").format_messages(question="hi", today="2026-11-14")
    assert [m.type for m in msgs[1:3]] == ["human", "ai"]
    assert len(msgs) == 2 + 2 * len(PROMPTS["router"].few_shot)


def test_every_prompt_labelled_with_agent_and_technique():
    for entry in prompt_catalog():
        assert entry["agent"] and entry["techniques"] and entry["system_prompt"] and entry["user_prompt"]


def test_required_techniques_are_covered():
    techniques = " ".join(t for p in PROMPTS.values() for t in p.techniques)
    for t in ("Few-shot", "Chain-of-Thought", "ReAct", "Reflection", "Zero-shot", "RAG"):
        assert t in techniques, t
