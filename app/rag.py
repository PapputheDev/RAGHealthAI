from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Sequence

from .llm import LLMError, generate_answer, generate_answer_stream
from .prompts import INSUFFICIENT_CONTEXT_MESSAGE, build_rag_messages
from .vector_store import SearchResult, search_documents

logger = logging.getLogger(__name__)


def _similarity_from_distance(distance: float) -> float:
    try:
        dist = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist))


def _confidence_from_results(results: Sequence[SearchResult]) -> float:
    if not results:
        return 0.0
    sims = [_similarity_from_distance(r.get("distance", 1.0)) for r in results]
    return float(sum(sims) / max(1, len(sims)))


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _build_context(results: Sequence[SearchResult]) -> str:
    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {}) or {}
        source = str(meta.get("source", "unknown"))
        chunk_index = meta.get("chunk_index")
        header = f"[Chunk {i} | source: {source}"
        if chunk_index is not None:
            header += f" | chunk_index: {chunk_index}"
        header += "]"
        lines.append(header)
        lines.append(str(r.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).strip()


def _extract_sources(results: Sequence[SearchResult]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    sources: List[Dict[str, str]] = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        src = str(meta.get("source", "unknown"))
        chunk_text = str(r.get("text", "")).strip()
        excerpt = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        if src not in seen:
            seen.add(src)
            sources.append({"document": src, "chunk": excerpt})
    return sources


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: List[Dict[str, str]]
    confidence: str


def _retrieve(question: str):
    """Shared retrieval step used by both sync and streaming paths."""
    results = search_documents(question, n_results=3)
    sources = _extract_sources(results)
    confidence = _confidence_label(_confidence_from_results(results))
    return results, sources, confidence


def answer_question(
    question: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Answer a question using RAG. Optionally includes conversation history."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    logger.info("RAG question received chars=%d history_turns=%d", len(question), len(history or []))
    results, sources, confidence = _retrieve(question)

    if not results:
        logger.warning("No retrieval results — skipping LLM")
        return asdict(RagResponse(answer=INSUFFICIENT_CONTEXT_MESSAGE, sources=[], confidence="low"))

    context = _build_context(results)
    messages = build_rag_messages(context, question, history or [])

    try:
        answer = generate_answer(messages)
    except LLMError:
        logger.exception("LLM call failed")
        raise

    response = RagResponse(answer=answer, sources=sources, confidence=confidence)
    logger.info("RAG answered confidence=%s sources=%s", confidence,
                ",".join(s["document"] for s in sources) if sources else "(none)")
    return asdict(response)


def answer_question_stream(
    question: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream a RAG answer token by token.

    Yields dicts:
      {"token": "..."}          — one per LLM token
      {"done": True, ...meta}   — final event with sources/confidence/route
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    logger.info("RAG stream question chars=%d", len(question))
    results, sources, confidence = _retrieve(question)

    if not results:
        logger.warning("No retrieval results — skipping LLM")
        yield {"token": INSUFFICIENT_CONTEXT_MESSAGE}
        yield {"done": True, "answer": INSUFFICIENT_CONTEXT_MESSAGE,
               "sources": [], "confidence": "low", "route": "rag"}
        return

    context = _build_context(results)
    messages = build_rag_messages(context, question, history or [])

    full_answer = ""
    try:
        for token in generate_answer_stream(messages):
            full_answer += token
            yield {"token": token}
    except LLMError:
        logger.exception("LLM stream failed")
        raise

    logger.info("RAG stream complete confidence=%s chars=%d", confidence, len(full_answer))
    yield {"done": True, "answer": full_answer,
           "sources": sources, "confidence": confidence, "route": "rag"}
