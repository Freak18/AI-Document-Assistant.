from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import LLM_PROVIDER, DEFAULT_MODEL
from database import DB_ENABLED, SessionLocal, db_health, ensure_database_ready
from gateway import gateway
from models import Conversation, Message
from trace_log import latest_id, log, since
from vector_store import OVERLAP_WORDS, WINDOW_WORDS, store
from workflow import run_evaluator, run_workflow

# Determine upload directory – writable on Vercel (/tmp) and safe locally.
# Use UPLOAD_DIR env var if provided; otherwise default to a local 'uploads' folder.
default_local = Path(__file__).with_name("uploads")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads" if os.getenv("VERCEL") else str(default_local)))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_database_ready()
    store.seed_sample_if_empty()
    gateway.clear_cache()
    log(
        f"[INFO] Vector store ready chunks={store.total_chunks()} "
        f"engine={store.engine} window={WINDOW_WORDS} overlap={OVERLAP_WORDS}"
    )
    log(f"[INFO] LLM gateway ready provider={LLM_PROVIDER} model={DEFAULT_MODEL}")
    db_status = db_health()
    log(f"[INFO] Database status: {db_status['status']}" + (
        f" ({db_status.get('detail', '')})" if db_status.get("detail") else ""
    ))
    yield


app = FastAPI(title="Enterprise AI Architecture Simulator", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None


_mem_conversations: dict[str, list[dict]] = {}


@app.get("/health")
def health() -> dict:
    db_status = db_health()
    return {
        "status": "ok",
        "chunks": store.total_chunks(),
        "engine": store.engine,
        "llm_provider": LLM_PROVIDER,
        "model": DEFAULT_MODEL,
        "cache_entries": gateway.cache_size(),
        "database": db_status["status"],
    }


@app.get("/metrics")
def metrics() -> dict:
    data = gateway.metrics()
    data["total_chunks"] = store.total_chunks()
    data["db_status"] = db_health()["status"]
    if DB_ENABLED and SessionLocal is not None and data.get("total_requests", 0) == 0:
        try:
            from sqlalchemy import func as sqlfunc
            from models import Evaluation, LLMUsage
            db = SessionLocal()
            try:
                agg = db.query(
                    sqlfunc.count(LLMUsage.id),
                    sqlfunc.coalesce(sqlfunc.sum(LLMUsage.total_tokens), 0),
                    sqlfunc.coalesce(sqlfunc.sum(LLMUsage.estimated_cost_usd), 0.0),
                ).first()
                if agg and agg[0]:
                    data["tokens_used"] = int(agg[1] or 0)
                    data["estimated_cost"] = round(float(agg[2] or 0.0), 6)
                latest_eval = db.query(Evaluation).order_by(Evaluation.created_at.desc()).first()
                if latest_eval and latest_eval.directness_score is not None:
                    data["latest_eval_score"] = round(float(latest_eval.directness_score), 2)
            finally:
                db.close()
        except Exception:
            pass
    return data


@app.get("/documents")
def documents() -> dict:
    return {
        "total_chunks": store.total_chunks(),
        "documents": store.document_stats(),
        "window_words": WINDOW_WORDS,
        "overlap_words": OVERLAP_WORDS,
    }


@app.post("/clear-cache")
def clear_cache() -> dict:
    cleared = gateway.clear_cache()
    log("[INFO] Semantic cache cleared by administrator")
    m = gateway.metrics()
    m["total_chunks"] = store.total_chunks()
    return {"status": "cleared", "entries_removed": cleared, "metrics": m}


@app.get("/logs")
def logs(after: int = 0) -> dict:
    return {"logs": since(after)}


def _persist_chat_to_db(conversation_id: str, user_message: str, assistant_answer: str, provider: str, model: str) -> None:
    """Persist conversation and messages to PostgreSQL (best-effort)."""
    if not DB_ENABLED or SessionLocal is None:
        return
    try:
        db = SessionLocal()
        try:
            # Get or create conversation
            conv = db.query(Conversation).filter_by(id=conversation_id).first()
            if not conv:
                conv = Conversation(id=conversation_id, title=user_message[:80])
                db.add(conv)
                db.flush()

            # Persist user message
            user_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                provider=provider,
                model=model,
            )
            db.add(user_msg)

            # Persist assistant message
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_answer,
                provider=provider,
                model=model,
            )
            db.add(assistant_msg)

            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log(f"[WARN] Could not persist chat to DB: {exc}")


def _load_history(conversation_id: str | None, limit: int = 6) -> list[dict]:
    """Load recent conversation turns from PostgreSQL or in-memory fallback."""
    if not conversation_id:
        return []
    if DB_ENABLED and SessionLocal is not None:
        try:
            db = SessionLocal()
            try:
                rows = (
                    db.query(Message)
                    .filter_by(conversation_id=conversation_id)
                    .order_by(Message.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {"role": r.role, "content": r.content}
                    for r in reversed(rows)
                    if r.role in ("user", "assistant")
                ]
            finally:
                db.close()
        except Exception as exc:
            log(f"[WARN] Could not load chat history from DB: {exc}")
    return list(_mem_conversations.get(conversation_id, []))[-limit:]


@app.post("/chat")
async def chat(body: ChatRequest, background: BackgroundTasks) -> dict:
    start = latest_id()
    user_message = body.message.strip()
    conversation_id = body.conversation_id or str(uuid.uuid4())
    history = _load_history(body.conversation_id)

    result = await run_workflow(user_message, history=history)

    if result["blocked"]:
        result["nodes"]["eval"] = "skipped"
    elif os.getenv("VERCEL"):
        # On serverless Vercel, background tasks may be cut off after response; run inline.
        await run_evaluator(user_message, result["answer"])
        result["nodes"]["eval"] = "done"
        result["trace"] = list(result["trace"]) + ["eval"]
    else:
        result["nodes"]["eval"] = "queued"
        result["trace"] = list(result["trace"]) + ["eval"]
        log("[INFO] Evaluator queued in the background")
        background.add_task(run_evaluator, user_message, result["answer"])

    # Persist conversation and messages to PostgreSQL and in-memory fallback
    provider = gateway.metrics().get("provider", "unknown")
    model = gateway.metrics().get("model", "unknown")
    _persist_chat_to_db(conversation_id, user_message, result.get("answer", ""), provider, model)
    turns = _mem_conversations.setdefault(conversation_id, [])
    turns.append({"role": "user", "content": user_message})
    turns.append({"role": "assistant", "content": result.get("answer", "")})
    if len(turns) > 20:
        del turns[: len(turns) - 20]

    result["logs"] = since(start)
    m = gateway.metrics()
    m["total_chunks"] = store.total_chunks()
    result["metrics"] = m
    result["conversation_id"] = conversation_id
    return result


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The PDF is empty")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF exceeds 10 MB limit")

    dest = UPLOAD_DIR / name
    dest.write_bytes(data)

    # A new upload becomes the source of truth: drop any older indexed documents before ingesting the file.
    store.clear_documents()
    gateway.clear_cache()
    _mem_conversations.clear()

    try:
        count, title = store.ingest_pdf(data, name, storage_path=str(dest.resolve()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read that PDF: {exc}") from exc

    retrieval_backend = store.engine
    entry = log(
        f"[INFO] Ingested {name} ('{title}') into {count} overlapping windows "
        f"({WINDOW_WORDS} words, {OVERLAP_WORDS} overlap) stored in {retrieval_backend} "
        f"without LLM calls"
    )
    prompt_message = (
        f"Document uploaded successfully. You can now ask questions about {title}."
    )
    m = gateway.metrics()
    m["total_chunks"] = store.total_chunks()
    return {
        "filename": name,
        "title": title,
        "chunks": count,
        "message": prompt_message,
        "logs": [entry],
        "metrics": m,
        "total_chunks": store.total_chunks(),
    }
