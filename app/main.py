from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .agent import handle_question, handle_question_stream
from .config import get_settings
from .ingest import ingest, ingest_file_list
from .llm import LLMError
from .memory import (
    clear_session,
    delete_conversation,
    get_history,
    get_messages,
    list_conversations,
    save_turn,
)
from .models import (
    AskRequest,
    AskResponse,
    ConversationMessagesResponse,
    ConversationsResponse,
    ConversationSummary,
    ChatMessage,
    DocumentsResponse,
    IndexedDocument,
    SessionRequest,
    SourceReference,
)
from .vector_store import count_documents, delete_by_source, list_sources

_DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


_configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HealthAI Assistant API",
    description=(
        "Healthcare AI Assistant demonstrating RAG, streaming, conversation memory, "
        "and a mock agentic appointment tool. Uses synthetic documents only."
    ),
    version="0.2.0",
)

if _STATIC_DIR.exists():
    # Static files are optional so the API can still run in environments that
    # package only the backend.
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.on_event("startup")
def auto_ingest() -> None:
    """Auto-ingest data/ on first startup if the vector store is empty."""
    try:
        count = count_documents()
    except Exception:
        count = 0

    if count > 0:
        logger.info("Vector store already has %d documents — skipping auto-ingest", count)
        return

    logger.info("Vector store is empty — running auto-ingest from data/")
    try:
        stats = ingest()
        logger.info("Auto-ingest complete: %d chunks from %d files",
                    stats["chunks_total"], stats["files_loaded"])
    except Exception:
        logger.exception("Auto-ingest failed — run POST /ingest manually")


# ── Authentication ────────────────────────────────────────────────────────────

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Gate an endpoint behind the optional shared API key.

    If APP_API_KEY is not configured, authentication is disabled and every
    request is allowed (keeps local/demo use friction-free). When it is set,
    the request must carry a matching `X-API-Key` header.
    """
    configured = get_settings().app_api_key
    if not configured:
        return
    if x_api_key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    logger.exception("LLMError")
    return JSONResponse(status_code=502,
                        content={"error": "llm_error", "message": str(exc)})


@app.exception_handler(ValidationError)
async def validation_error_handler(_: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("ValidationError: %s", exc)
    return JSONResponse(status_code=422,
                        content={"error": "validation_error", "message": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500,
                        content={"error": "internal_server_error",
                                 "message": "An unexpected error occurred."})


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    # The frontend is a single HTML file; no separate build step is required.
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health() -> Dict[str, Any]:
    # Report whether the vector store is reachable and populated so the UI (and
    # any uptime monitor) can distinguish "running" from "ready to answer".
    try:
        doc_count = count_documents()
        store_ok = True
    except Exception:
        logger.exception("Health check: vector store unreachable")
        doc_count, store_ok = 0, False

    return {
        "status": "ok" if store_ok else "degraded",
        "vector_store": "ok" if store_ok else "unreachable",
        "indexed_chunks": doc_count,
    }


@app.get("/documents", summary="List indexed documents", response_model=DocumentsResponse,
         dependencies=[Depends(require_api_key)])
def list_documents_endpoint() -> DocumentsResponse:
    # Powers the "Manage Docs" UI: which documents are in the knowledge base and
    # how many chunks each contributes.
    sources = list_sources()
    docs = [IndexedDocument(document=s["document"], chunks=int(s["chunks"])) for s in sources]
    return DocumentsResponse(documents=docs, total_chunks=sum(d.chunks for d in docs))


@app.delete("/documents/{name}", summary="Delete an indexed document",
            dependencies=[Depends(require_api_key)])
def delete_document_endpoint(name: str) -> Dict[str, Any]:
    # Remove a document's chunks from the vector store and delete the underlying
    # file from data/ so it isn't re-ingested on the next ingest run.
    safe_name = Path(name).name  # guard against path traversal
    removed = delete_by_source(safe_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No indexed document named '{safe_name}'")

    file_path = _DATA_DIR / safe_name
    file_deleted = False
    if file_path.exists() and file_path.is_file():
        try:
            file_path.unlink()
            file_deleted = True
        except OSError:
            logger.exception("Failed to remove data file for %s", safe_name)

    return {"ok": True, "document": safe_name, "chunks_removed": removed, "file_deleted": file_deleted}


@app.post("/session/clear", summary="Clear a conversation's stored history",
          dependencies=[Depends(require_api_key)])
def clear_session_endpoint(payload: SessionRequest) -> Dict[str, Any]:
    removed = clear_session(payload.session_id)
    return {"ok": True, "messages_removed": removed}


@app.get("/conversations", summary="List an owner's saved conversations",
         response_model=ConversationsResponse, dependencies=[Depends(require_api_key)])
def list_conversations_endpoint(owner_id: str) -> ConversationsResponse:
    # owner_id identifies the browser/user; each has its own set of chats.
    convos = [ConversationSummary(**c) for c in list_conversations(owner_id)]
    return ConversationsResponse(conversations=convos)


@app.get("/conversations/{conversation_id}/messages", summary="Get a conversation's full history",
         response_model=ConversationMessagesResponse, dependencies=[Depends(require_api_key)])
def conversation_messages_endpoint(conversation_id: str) -> ConversationMessagesResponse:
    msgs = [ChatMessage(**m) for m in get_messages(conversation_id)]
    return ConversationMessagesResponse(conversation_id=conversation_id, messages=msgs)


@app.delete("/conversations/{conversation_id}", summary="Delete a saved conversation",
            dependencies=[Depends(require_api_key)])
def delete_conversation_endpoint(conversation_id: str) -> Dict[str, Any]:
    removed = delete_conversation(conversation_id)
    return {"ok": True, "conversation_id": conversation_id, "messages_removed": removed}


@app.post("/ingest", summary="Ingest knowledge base", dependencies=[Depends(require_api_key)])
def ingest_endpoint() -> Dict[str, Any]:
    # Rebuild or update the vector store from files already present in data/.
    try:
        stats = ingest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "stats": stats}


@app.post("/upload", summary="Upload and ingest documents", dependencies=[Depends(require_api_key)])
async def upload_endpoint(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    # Uploaded files are saved into data/ first, then passed through the same
    # chunk/embed/store pipeline as the bundled knowledge-base files.
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved: List[Path] = []
    for upload in files:
        filename = Path(upload.filename or "upload").name
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' not supported. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            )
        dest = _DATA_DIR / filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(dest)
        logger.info("Saved uploaded file: %s (%d bytes)", filename, dest.stat().st_size)

    try:
        stats = ingest_file_list(saved)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"ok": True, "files": [p.name for p in saved], "stats": stats}


@app.post("/ask", summary="Ask a question (non-streaming)", response_model=AskResponse,
          dependencies=[Depends(require_api_key)])
def ask_endpoint(payload: AskRequest) -> AskResponse:
    # Non-streaming path: route the question, normalize the route-specific
    # result, and optionally persist the turn for follow-up questions.
    history = get_history(payload.session_id) if payload.session_id else []
    result = handle_question(payload.question, history=history)
    route = result.get("route")
    inner = result.get("result", {})

    if route == "appointment_tool":
        answer = str(inner.get("message", ""))
        slots = inner.get("available_slots", [])
        if slots:
            answer += "\n\nAvailable slots:\n" + "\n".join(
                f"- {s.get('start')} ({s.get('duration_minutes')} min, {s.get('modality')})"
                for s in slots
            )
        if payload.session_id:
            save_turn(payload.session_id, payload.question, answer, owner_id=payload.owner_id)
        return AskResponse(answer=answer, sources=[], confidence="high",
                           confidence_score=1.0, route="appointment_tool")

    if route == "rag":
        answer     = str(inner.get("answer", ""))
        confidence = str(inner.get("confidence", "low"))
        score      = float(inner.get("confidence_score", 0.0) or 0.0)
        sources    = [SourceReference(**s) for s in inner.get("sources", [])]
        if payload.session_id:
            save_turn(payload.session_id, payload.question, answer, owner_id=payload.owner_id)
        return AskResponse(answer=answer, sources=sources, confidence=confidence,
                           confidence_score=score, route="rag")

    raise HTTPException(status_code=500, detail="Unknown agent route")


@app.post("/ask/stream", summary="Ask a question (streaming SSE)",
          dependencies=[Depends(require_api_key)])
async def ask_stream_endpoint(payload: AskRequest) -> StreamingResponse:
    """Stream the answer token by token as Server-Sent Events.

    Event format:
      data: {"token": "..."}            — one per LLM token
      data: {"done": true, ...metadata} — final event
    """
    history = get_history(payload.session_id) if payload.session_id else []

    async def event_stream() -> AsyncIterator[str]:
        # The routing/LLM code is synchronous, so run it in a worker thread and
        # bridge events back into the async SSE response with a queue.
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            try:
                for ev in handle_question_stream(payload.question, history=history):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, {"error": str(exc)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _run)

        full_answer = ""
        while True:
            ev = await q.get()
            if ev is None:
                break
            if "token" in ev:
                full_answer += ev["token"]
            if "done" in ev and payload.session_id:
                save_turn(payload.session_id, payload.question,
                          ev.get("answer", full_answer), owner_id=payload.owner_id)
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
