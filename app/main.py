from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .agent import handle_question, handle_question_stream
from .ingest import ingest, ingest_file_list
from .llm import LLMError
from .memory import get_history, save_turn
from .models import AskRequest, AskResponse, SourceReference
from .vector_store import count_documents

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
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", summary="Ingest knowledge base")
def ingest_endpoint() -> Dict[str, Any]:
    try:
        stats = ingest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "stats": stats}


@app.post("/upload", summary="Upload and ingest documents")
async def upload_endpoint(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
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


@app.post("/ask", summary="Ask a question (non-streaming)", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
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
            save_turn(payload.session_id, payload.question, answer)
        return AskResponse(answer=answer, sources=[], confidence="high", route="appointment_tool")

    if route == "rag":
        answer     = str(inner.get("answer", ""))
        confidence = str(inner.get("confidence", "low"))
        sources    = [SourceReference(**s) for s in inner.get("sources", [])]
        if payload.session_id:
            save_turn(payload.session_id, payload.question, answer)
        return AskResponse(answer=answer, sources=sources, confidence=confidence, route="rag")

    raise HTTPException(status_code=500, detail="Unknown agent route")


@app.post("/ask/stream", summary="Ask a question (streaming SSE)")
async def ask_stream_endpoint(payload: AskRequest) -> StreamingResponse:
    """Stream the answer token by token as Server-Sent Events.

    Event format:
      data: {"token": "..."}            — one per LLM token
      data: {"done": true, ...metadata} — final event
    """
    history = get_history(payload.session_id) if payload.session_id else []

    async def event_stream() -> AsyncIterator[str]:
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
                          ev.get("answer", full_answer))
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
