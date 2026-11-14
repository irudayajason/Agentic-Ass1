"""LLM and embedding factories. Every agent gets its model from here."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

from spendsense import config


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings over fastembed's local ONNX models (no torch, no API)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=4)
def get_llm(model: str | None = None, temperature: float = config.LLM_TEMPERATURE) -> ChatGroq:
    """Main chat model (agents). Pass model=config.ROUTER_MODEL for the fast router."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(
        model=model or config.LLM_MODEL,
        temperature=temperature,
        api_key=config.GROQ_API_KEY,
        # Free tier allows 8,000 tokens/minute per model. The Groq client waits out a 429 using its
        # retry-after (usually under 20s). The timeout stops a stalled request from hanging for the
        # client's 10-minute default.
        timeout=60,
        max_retries=2,   # then the chain's fallback switches to the other model
    )


def get_router_llm() -> ChatGroq:
    return get_llm(config.ROUTER_MODEL, temperature=0.0)


def backups_for(llm) -> list[ChatGroq]:
    """The other Groq models, in the order to try them. Each model has its own per-minute and
    per-day token allowance, so when one is rate-limited another can usually still answer."""
    if not isinstance(llm, ChatGroq):
        return []   # fakes in tests
    order = [config.LLM_MODEL, config.ROUTER_MODEL, config.BACKUP_MODEL]
    return [get_llm(m, temperature=llm.temperature) for m in dict.fromkeys(order) if m and m != llm.model_name]


def backup_for(llm) -> ChatGroq | None:
    """First backup model (kept for callers that only want one)."""
    b = backups_for(llm)
    return b[0] if b else None


def rate_limit_errors() -> tuple[type[Exception], ...]:
    import groq
    return (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)


def get_fast_llm() -> ChatGroq:
    """Smaller model for light steps (planning, critique, summaries). It has its own
    tokens-per-minute allowance on Groq, so splitting work across models keeps answers fast."""
    return get_llm(config.ROUTER_MODEL, temperature=0.1)


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Local ONNX embeddings (downloaded once, ~70 MB). No API calls."""
    return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)
