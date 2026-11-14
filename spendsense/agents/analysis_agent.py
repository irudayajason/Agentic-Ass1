"""Analysis Agent: answers spending and lookup questions with a ReAct tool loop.

    LCEL step:  analysis_agent prompt | llm.bind_tools(tools)
    loop:       Thought (model reasoning) -> Action (tool call) -> Observation (tool result
                appended to the scratchpad) -> ... -> final answer when no more tool calls.

Every step is recorded so the UI can show which tools were called, with what arguments, what
came back, and whether the agent chose to use the vector database.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from spendsense.llm import backups_for, get_llm, rate_limit_errors
from spendsense.prompts import build_prompt
from spendsense.tools.pandas_tools import build_tools

MAX_STEPS = 8
OBSERVATION_PREVIEW = 1200   # characters of each tool result kept in the trace


@dataclass
class AgentStep:
    step: int
    thought: str
    tool: str
    args: dict
    observation: str
    seconds: float
    model: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    final_thought: str = ""
    stopped_early: bool = False
    empty_answer_retried: bool = False
    models: list[str] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def used_retrieval(self) -> bool:
        return "semantic_txn_search" in self.tools_used

    def to_dict(self) -> dict:
        return {"answer": self.answer, "final_thought": self.final_thought, "stopped_early": self.stopped_early,
                "empty_answer_retried": self.empty_answer_retried,
                "tools_used": self.tools_used, "used_retrieval": self.used_retrieval, "models": self.models,
                "steps": [s.__dict__ for s in self.steps]}


def data_summary(df: pd.DataFrame) -> str:
    months = ", ".join(sorted(df["date"].dt.strftime("%Y-%m").unique()))
    return (f"{len(df)} transactions from {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} "
            f"(months: {months}). Categories: {', '.join(sorted(df['category'].dropna().unique()))}.")


class AnalysisAgent:
    def __init__(self, get_df, llm=None, chroma_client=None, max_steps: int = MAX_STEPS):
        self.get_df = get_df
        self.tools = build_tools(get_df, chroma_client)
        self.tools_by_name = {t.name: t for t in self.tools}
        llm = llm or get_llm()
        model = llm.bind_tools(self.tools)
        backups = backups_for(llm)
        if backups:   # the other Groq models take over, in order, when this one is rate-limited
            model = model.with_fallbacks([b.bind_tools(self.tools) for b in backups],
                                         exceptions_to_handle=rate_limit_errors())
        self.chain = build_prompt("analysis_agent") | model  # LCEL
        self.max_steps = max_steps

    def run(self, question: str, history: list[BaseMessage] | None = None, intent: str = "spending_analysis",
            entities: dict | None = None, today: str | None = None) -> AgentResult:
        inputs = {
            "question": question,
            "history": history or [],
            "data_summary": data_summary(self.get_df()),
            "intent": intent,
            "entities": json.dumps(entities or {}, ensure_ascii=False),
            "today": today or date.today().isoformat(),
        }
        scratchpad: list[BaseMessage] = []
        result = AgentResult(answer="")
        empty_retry_used = False

        for step in range(1, self.max_steps + 1):
            try:
                msg: AIMessage = self.chain.invoke({**inputs, "scratchpad": scratchpad})
            except Exception as exc:  # noqa: BLE001
                # The API rejects a malformed tool call before we see it. Tell the model and let it retry.
                if "tool_use_failed" not in str(exc) and "Tool call validation" not in str(exc):
                    raise
                err = str(exc).split("'message': '")[-1].split("', 'type'")[0][:300]
                result.steps.append(AgentStep(step=step, thought="", tool="(rejected tool call)", args={},
                                              observation=err, seconds=0.0))
                scratchpad.append(HumanMessage(f"Your last tool call was rejected: {err}. "
                                               "Fix the arguments and try again."))
                continue
            thought = (msg.additional_kwargs.get("reasoning_content") or "").strip()
            model_name = (getattr(msg, "response_metadata", None) or {}).get("model_name", "")
            if model_name and model_name not in result.models:
                result.models.append(model_name)
            if not msg.tool_calls:                       # no Action -> this is the final answer
                answer = (msg.content or "").strip()
                if not answer and not empty_retry_used:
                    # Small models sometimes finish with reasoning only and no text: ask once for the answer
                    empty_retry_used = True
                    result.empty_answer_retried = True
                    scratchpad.append(HumanMessage("You did not write an answer. Write the final answer for the "
                                                   "user now in plain text, using the tool results above."))
                    continue
                result.answer, result.final_thought = answer or self._fallback_answer(result), thought
                return result
            scratchpad.append(msg)
            for call in msg.tool_calls:                  # Action(s)
                t0 = time.time()
                tool = self.tools_by_name.get(call["name"])
                obs = tool.invoke(call["args"]) if tool else json.dumps({"error": f"Unknown tool {call['name']}"})
                scratchpad.append(ToolMessage(content=obs, tool_call_id=call["id"]))   # Observation
                result.steps.append(AgentStep(step=step, thought=thought, tool=call["name"], args=call["args"],
                                              observation=obs[:OBSERVATION_PREVIEW],
                                              seconds=round(time.time() - t0, 3), model=model_name))
                thought = ""   # the model's reasoning belongs to the first call of the step

        # Ran out of steps: ask for an answer from what has been gathered so far
        result.stopped_early = True
        final = self.chain.invoke({**inputs, "question": question + "\n\n(Answer now using the tool results "
                                   "you already have; do not call more tools.)", "scratchpad": scratchpad})
        result.answer = (final.content or "").strip() or self._fallback_answer(result)
        return result

    @staticmethod
    def _fallback_answer(result: "AgentResult") -> str:
        """Never show an empty bubble: say so and point at the data that was found."""
        if result.steps:
            return ("I found the data but couldn't write up the answer this time (the model returned no text). "
                    "Open **How I got this answer** to see the tool results, or ask again.")
        return "Sorry, I couldn't answer that. Please try rephrasing the question."
