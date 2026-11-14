"""Phase 0 smoke test: imports, local embeddings, ChromaDB, and a Groq LCEL call.

Run:  .venv/bin/python scripts/check_setup.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step(name, fn):
    try:
        detail = fn()
        print(f"  OK    {name}" + (f"  ->  {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001 - report every failure
        print(f"  FAIL  {name}  ->  {type(exc).__name__}: {exc}")
        return False


def check_imports():
    import chromadb, fastapi, langchain, langchain_core, pandas, pdfplumber, reportlab  # noqa: F401,E401
    return f"python {sys.version.split()[0]}, langchain {langchain.__version__}, chromadb {chromadb.__version__}"


def check_config():
    from spendsense import config
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing from .env")
    return f"models: {config.LLM_MODEL} / {config.ROUTER_MODEL}, {len(config.CATEGORIES)} categories"


def check_embeddings_and_chroma():
    from langchain_chroma import Chroma
    from spendsense.llm import get_embeddings

    with tempfile.TemporaryDirectory() as tmp:
        store = Chroma("smoke", embedding_function=get_embeddings(), persist_directory=tmp)
        store.add_texts([
            "UPI to SWIGGY food order 432",
            "APOLLO PHARMACY medicines 820",
            "UBER trip to office 260",
        ])
        hit = store.similarity_search("medical store payment", k=1)[0].page_content
    if "APOLLO" not in hit:
        raise RuntimeError(f"unexpected top match: {hit}")
    return f"'medical store payment' matched '{hit}'"


def check_groq_lcel():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from spendsense.llm import get_llm, get_router_llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a terse assistant. Reply with one word."),
        ("user", "Which spending category is a Swiggy order?"),
    ])
    out = []
    for llm in (get_llm(), get_router_llm()):
        chain = prompt | llm | StrOutputParser()  # LCEL
        out.append(f"{llm.model_name}: {chain.invoke({}).strip()!r}")
    return "; ".join(out)


if __name__ == "__main__":
    print("SpendSense setup check")
    results = [
        step("Python packages import", check_imports),
        step("Config + .env loaded", check_config),
        step("Local embeddings + ChromaDB similarity search", check_embeddings_and_chroma),
        step("Groq LLM via LCEL chain", check_groq_lcel),
    ]
    print("\nAll checks passed." if all(results) else "\nSome checks failed.")
    sys.exit(0 if all(results) else 1)
