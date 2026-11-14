"""Conversation memory: one chat history per session, trimmed to the most recent messages.

Used through LangChain's `RunnableWithMessageHistory`, which reads the history into the
prompt before each turn and appends the new question and answer afterwards. Trimming keeps
prompts small (important on the free Groq tier) while still resolving follow-ups.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

MAX_MESSAGES = 12          # 6 question/answer pairs
MAX_CHARS_PER_MESSAGE = 1500


class TrimmedChatHistory(InMemoryChatMessageHistory):
    max_messages: int = MAX_MESSAGES

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for m in messages:
            if isinstance(m.content, str) and len(m.content) > MAX_CHARS_PER_MESSAGE:
                m = m.model_copy(update={"content": m.content[:MAX_CHARS_PER_MESSAGE] + " …"})
            self.messages.append(m)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
