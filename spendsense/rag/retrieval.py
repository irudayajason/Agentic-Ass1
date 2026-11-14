"""Retrieval layer over ChromaDB: the knowledge base and the statement's transactions.

finance_kb    20 markdown notes, split by section with LangChain's MarkdownHeaderTextSplitter,
              re-indexed automatically when any note changes (content hash manifest).
transactions  one document per transaction with rich metadata (month, amount, type, category),
              so agents can combine meaning-based search with exact filters.

These functions are what the agents' retrieval tools call.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from spendsense import config
from spendsense.rag.vectorstore import get_store, reset_collection

KB_MIN_RELEVANCE = 0.55     # measured: relevant chunks score 0.66-0.78, off-topic queries <= 0.52
MANIFEST = config.CHROMA_DIR / "kb_manifest.json"


# =================================================================== knowledge base
def load_kb_documents(kb_dir=config.KB_DIR) -> list[Document]:
    """Split each note into one chunk per '##' section, keeping the note title as context."""
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=True)
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        for i, chunk in enumerate(splitter.split_text(path.read_text(encoding="utf-8"))):
            title = chunk.metadata.get("title", path.stem)
            section = chunk.metadata.get("section", "Overview")
            docs.append(Document(
                # title + section go into the embedded text so short chunks keep their topic
                page_content=f"{title} - {section}\n{chunk.page_content.strip()}",
                metadata={"source": path.name, "title": title, "section": section, "chunk": i},
            ))
    return docs


def _kb_hash(kb_dir=config.KB_DIR) -> str:
    h = hashlib.sha256()
    for path in sorted(kb_dir.glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def ensure_kb_indexed(client=None, force: bool = False) -> dict:
    """Index the KB into ChromaDB if it isn't there yet or a note changed since last time."""
    current = _kb_hash()
    if client is None and not force and MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text())
        if saved.get("hash") == current and get_store(config.KB_COLLECTION)._collection.count():
            return {"indexed": False, "chunks": saved["chunks"], "notes": saved["notes"]}
    docs = load_kb_documents()
    store = reset_collection(config.KB_COLLECTION, client)
    store.add_documents(docs, ids=[f"{d.metadata['source']}#{d.metadata['chunk']}" for d in docs])
    info = {"hash": current, "chunks": len(docs), "notes": len({d.metadata["source"] for d in docs}),
            "indexed_at": datetime.now().isoformat(timespec="seconds")}
    if client is None:
        MANIFEST.write_text(json.dumps(info, indent=2))
    return {"indexed": True, **{k: info[k] for k in ("chunks", "notes")}}


def search_kb(query: str, k: int = 4, min_relevance: float = KB_MIN_RELEVANCE, client=None) -> list[dict]:
    """Similarity search over the knowledge base. Weak matches are dropped."""
    store = get_store(config.KB_COLLECTION, client)
    hits = store.similarity_search_with_relevance_scores(query, k=k)
    return [{"source": d.metadata["source"], "title": d.metadata["title"], "section": d.metadata["section"],
             "text": d.page_content, "score": round(s, 3)} for d, s in hits if s >= min_relevance]


# =================================================================== transactions
def txn_document(row) -> Document:
    """One transaction as searchable text + filterable metadata."""
    date = pd.Timestamp(row["date"])
    note = row.get("note") or ""
    cat = row.get("category") or "Uncategorised"
    text = (f"{date:%d %b %Y} | {row['payee']} | {note} | {row['mode']} {row['type'].lower()} "
            f"₹{row['amount']:,.2f} | {cat} | {row['description']}")
    return Document(page_content=text, metadata={
        "txn_id": row["txn_id"], "date": f"{date:%Y-%m-%d}", "month": f"{date:%Y-%m}",
        "ts": int(date.timestamp()), "amount": float(row["amount"]), "type": row["type"],
        "category": cat, "payee": row["payee"], "mode": row["mode"],
    })


def index_transactions(df: pd.DataFrame, client=None) -> int:
    """Replace the transactions collection with this statement (one statement loaded at a time)."""
    store = reset_collection(config.TXN_COLLECTION, client)
    docs = [txn_document(r) for _, r in df.iterrows()]
    for i in range(0, len(docs), 200):   # batched to keep each embedding call small
        part = docs[i:i + 200]
        store.add_documents(part, ids=[d.metadata["txn_id"] for d in part])
    return len(docs)


def update_transaction_category(row, client=None) -> None:
    """Re-embed one transaction after the user changes its category."""
    doc = txn_document(row)
    get_store(config.TXN_COLLECTION, client).add_documents([doc], ids=[doc.metadata["txn_id"]])


def _build_filter(type: str | None, category: str | None, month: str | None,
                  min_amount: float | None, max_amount: float | None,
                  date_from: str | None, date_to: str | None) -> dict | None:
    conds = []
    if type:
        conds.append({"type": type.upper()})
    if category:
        conds.append({"category": category})
    if month:
        conds.append({"month": month})
    if min_amount is not None:
        conds.append({"amount": {"$gte": float(min_amount)}})
    if max_amount is not None:
        conds.append({"amount": {"$lte": float(max_amount)}})
    if date_from:
        conds.append({"ts": {"$gte": int(pd.Timestamp(date_from).timestamp())}})
    if date_to:
        conds.append({"ts": {"$lte": int((pd.Timestamp(date_to) + pd.Timedelta(days=1)).timestamp()) - 1}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search_transactions(query: str, k: int = 8, type: str | None = None, category: str | None = None,
                        month: str | None = None, min_amount: float | None = None,
                        max_amount: float | None = None, date_from: str | None = None,
                        date_to: str | None = None, client=None) -> list[dict]:
    """Meaning-based search over transactions, optionally narrowed by exact metadata filters."""
    store = get_store(config.TXN_COLLECTION, client)
    where = _build_filter(type, category, month, min_amount, max_amount, date_from, date_to)
    hits = store.similarity_search_with_relevance_scores(query, k=k, filter=where)
    return [{**d.metadata, "text": d.page_content, "score": round(s, 3)} for d, s in hits]


def collection_stats(client=None) -> dict:
    """Counts for the 'Under the Hood' tab."""
    out = {}
    for name in (config.KB_COLLECTION, config.TXN_COLLECTION, config.EXAMPLES_COLLECTION):
        try:
            out[name] = get_store(name, client)._collection.count()
        except Exception:  # noqa: BLE001
            out[name] = 0
    return out
