"""SpendSense web server: FastAPI JSON API + the static HTML/CSS/JS frontend.

Run:  .venv/bin/uvicorn server:app --port 8000
Then open http://localhost:8000

The browser keeps a session id in localStorage and sends it as the `X-Session-Id` header.
The Groq key stays on the server and is never sent to the browser.
"""

from __future__ import annotations

import math
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spendsense import config
from spendsense.agents.parser_agent import ParseError
from spendsense.prompts import prompt_catalog
from spendsense.rag import retrieval
from spendsense.views import ACCEPTANCE_CRITERIA, dashboard, transactions_csv, transactions_page

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_SID = re.compile(r"^[a-zA-Z0-9_-]{6,40}$")


def _json_safe(obj):
    """Replace NaN/inf (which JSON can't represent) with None, anywhere in a response."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """Real statements can produce NaN (e.g. a month with no income); never let that become a 500."""

    def render(self, content) -> bytes:
        return super().render(_json_safe(content))


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class CategoryIn(BaseModel):
    category: str
    apply_to_same_payee: bool = True


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    collection: str = Field("finance_kb", pattern="^(finance_kb|transactions)$")
    k: int = Field(5, ge=1, le=10)


def create_app(orchestrator=None) -> FastAPI:
    state: dict = {"orch": orchestrator, "insights": {}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["orch"] is None:
            from spendsense.orchestrator import Orchestrator
            retrieval.ensure_kb_indexed()
            state["orch"] = Orchestrator()
        yield

    app = FastAPI(title="SpendSense", lifespan=lifespan, default_response_class=SafeJSONResponse)

    @app.middleware("http")
    async def no_stale_frontend(request, call_next):
        # Make the browser revalidate the page, CSS and JS so edits show up without a hard refresh
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def orch():
        return state["orch"]

    def session(sid: str | None):
        if sid and not _SID.match(sid):
            raise HTTPException(400, "Bad session id")
        return orch().session(sid)

    def need_data(sid: str | None):
        s = session(sid)
        if not s.has_data:
            raise HTTPException(409, "No statement loaded yet. Upload one or load the sample.")
        return s

    def _summary(s):
        state["insights"].pop(s.id, None)
        return {"session_id": s.id, **s.ingest_summary}

    # ------------------------------------------------------------------ status
    @app.get("/api/health")
    def health(x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        return {"ok": True, "session_id": s.id, "has_data": s.has_data, "filename": s.filename,
                "models": {"main": config.LLM_MODEL, "fast": config.ROUTER_MODEL},
                "llm_configured": bool(config.GROQ_API_KEY), "collections": retrieval.collection_stats(),
                "ingest": s.ingest_summary if s.has_data else None}

    # ------------------------------------------------------------------ data in
    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 10 MB.")
        try:
            orch().ingest(s.id, data, file.filename or "statement.csv")
        except ParseError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _summary(s)

    @app.post("/api/sample")
    def sample(name: str = "bank_statement.csv", x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        try:
            orch().load_sample(s.id, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _summary(s)

    @app.post("/api/reset")
    def reset(x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        orch().reset(s.id)
        state["insights"].pop(s.id, None)
        return {"ok": True}

    # ------------------------------------------------------------------ views
    @app.get("/api/dashboard")
    def get_dashboard(x_session_id: str | None = Header(None)):
        s = need_data(x_session_id)
        return {"filename": s.filename, **dashboard(s.df, s.alerts)}

    @app.get("/api/transactions")
    def get_transactions(q: str = "", category: str = "", month: str = "", type: str = "",
                         sort: str = "date", desc: bool = False, page: int = Query(1, ge=1),
                         page_size: int = Query(25, ge=5, le=100), x_session_id: str | None = Header(None)):
        s = need_data(x_session_id)
        return transactions_page(s.df, q, category, month, type, sort, desc, page, page_size)

    @app.get("/api/transactions.csv")
    def export_csv(session_id: str = Query(...)):
        s = need_data(session_id)
        return PlainTextResponse(transactions_csv(s.df), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=spendsense_transactions.csv"})

    @app.patch("/api/transactions/{txn_id}")
    def patch_category(txn_id: str, body: CategoryIn, x_session_id: str | None = Header(None)):
        s = need_data(x_session_id)
        if body.category not in config.CATEGORIES:
            raise HTTPException(422, f"Unknown category {body.category}")
        if txn_id not in set(s.df["txn_id"]):
            raise HTTPException(404, "No such transaction")
        changed = orch().set_category(s.id, txn_id, body.category, body.apply_to_same_payee)
        state["insights"].pop(s.id, None)
        return {"ok": True, "changed": changed}

    @app.get("/api/insights")
    def insights(refresh: bool = False, x_session_id: str | None = Header(None)):
        s = need_data(x_session_id)
        if refresh or s.id not in state["insights"]:
            state["insights"][s.id] = s.advisor.insights()
        return state["insights"][s.id]

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat")
    def chat(body: ChatIn, x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        try:
            return orch().chat(s.id, body.message)
        except Exception as exc:  # noqa: BLE001 - show a friendly message, keep the server up
            msg = str(exc)
            if "rate_limit" in msg or "Rate limit" in msg:
                raise HTTPException(429, "The free Groq quota is used up for the moment on all three models. "
                                         "Please wait a few minutes and try again.") from exc
            raise HTTPException(500, f"Something went wrong: {type(exc).__name__}") from exc

    @app.get("/api/history")
    def history(x_session_id: str | None = Header(None)):
        s = session(x_session_id)
        return {"messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content}
                             for m in s.history.messages]}

    # ------------------------------------------------------------------ under the hood
    @app.get("/api/prompts")
    def prompts():
        return {"prompts": prompt_catalog(), "criteria": ACCEPTANCE_CRITERIA}

    @app.post("/api/search")
    def search(body: SearchIn, x_session_id: str | None = Header(None)):
        if body.collection == "finance_kb":
            hits = retrieval.search_kb(body.query, k=body.k, min_relevance=0)
            return {"collection": "finance_kb", "threshold": retrieval.KB_MIN_RELEVANCE, "results": hits}
        need_data(x_session_id)
        return {"collection": "transactions", "results": retrieval.search_transactions(body.query, k=body.k)}

    # ------------------------------------------------------------------ frontend
    @app.get("/")
    def index():
        return FileResponse(config.FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
    return app


app = create_app()
