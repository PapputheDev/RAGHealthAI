from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Sequence

from .llm import LLMError, generate_answer, generate_answer_stream
from .prompts import INSUFFICIENT_CONTEXT_MESSAGE, build_rag_messages
from .vector_store import SearchResult, search_documents

logger = logging.getLogger(__name__)

# If even the best-matching chunk scores below this cosine similarity, the
# knowledge base almost certainly doesn't cover the question. We then skip the
# LLM entirely and return the fixed fallback — cheaper and far less likely to
# hallucinate an answer from barely-related text.
RELEVANCE_FLOOR = 0.35


def _similarity_from_distance(distance: float) -> float:
    # The collection uses Chroma's default "l2" space, which reports the SQUARED
    # Euclidean distance. Our embeddings are L2-normalized, so that distance
    # equals 2 - 2*cosine, giving a 0..4 range. Recover cosine similarity with
    # 1 - dist/2 so the confidence thresholds map to real similarity values.
    # (The old `1 - dist` under-scored every result and made "high" unreachable.)
    try:
        dist = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / 2.0))


def _confidence_from_results(results: Sequence[SearchResult]) -> float:
    # Average retrieved chunk similarities. This is a lightweight heuristic,
    # not a medically validated confidence score.
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
    # Build plain-text context with source headers so the LLM can cite filenames.
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
    # Return unique source documents with short excerpts for the API/UI.
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


def _best_similarity(results: Sequence[SearchResult]) -> float:
    # The single closest chunk is what decides whether the KB covers the topic
    # at all, so the relevance floor is checked against the best match, not the
    # average (which a few weak chunks could drag down).
    if not results:
        return 0.0
    return max(_similarity_from_distance(r.get("distance", 1.0)) for r in results)


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: List[Dict[str, str]]
    confidence: str
    confidence_score: float = 0.0


def _is_insufficient_answer(answer: str) -> bool:
    # The model is instructed to reply with the exact fallback message when the
    # context can't answer the question. In that case the retrieved chunks were
    # not actually used, so we shouldn't show them as sources.
    return answer.strip().rstrip(".").lower() == INSUFFICIENT_CONTEXT_MESSAGE.rstrip(".").lower()


def _retrieve(question: str):
    """Shared retrieval step used by both sync and streaming paths."""
    # Keeping retrieval here makes sync and streaming answers use the same
    # search, source extraction, and confidence calculation.
    results = search_documents(question, n_results=3)
    sources = _extract_sources(results)
    score = round(_confidence_from_results(results), 3)
    confidence = _confidence_label(score)
    return results, sources, confidence, score


def answer_question(
    question: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Answer a question using RAG. Optionally includes conversation history."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    logger.info("RAG question received chars=%d history_turns=%d", len(question), len(history or []))
    results, sources, confidence, score = _retrieve(question)

    if not results or _best_similarity(results) < RELEVANCE_FLOOR:
        # No results, or the best match is too weak to trust: return the fixed
        # fallback without spending an LLM call or risking an answer from
        # barely-related text.
        logger.warning("Retrieval below relevance floor (best=%.3f) — skipping LLM",
                       _best_similarity(results))
        return asdict(RagResponse(answer=INSUFFICIENT_CONTEXT_MESSAGE, sources=[],
                                  confidence="low", confidence_score=score))

    context = _build_context(results)
    messages = build_rag_messages(context, question, history or [])

    try:
        answer = generate_answer(messages)
    except LLMError:
        logger.exception("LLM call failed")
        raise

    if _is_insufficient_answer(answer):
        # Answer came from the fallback, not the documents — drop the sources.
        sources, confidence = [], "low"

    response = RagResponse(answer=answer, sources=sources, confidence=confidence,
                           confidence_score=score)
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
    results, sources, confidence, score = _retrieve(question)

    if not results or _best_similarity(results) < RELEVANCE_FLOOR:
        # Streaming callers still receive a token event followed by final
        # metadata, matching the normal SSE response shape.
        logger.warning("Retrieval below relevance floor (best=%.3f) — skipping LLM",
                       _best_similarity(results))
        yield {"token": INSUFFICIENT_CONTEXT_MESSAGE}
        yield {"done": True, "answer": INSUFFICIENT_CONTEXT_MESSAGE,
               "sources": [], "confidence": "low", "confidence_score": score, "route": "rag"}
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

    if _is_insufficient_answer(full_answer):
        # Answer came from the fallback, not the documents — drop the sources.
        sources, confidence = [], "low"

    logger.info("RAG stream complete confidence=%s chars=%d", confidence, len(full_answer))
    yield {"done": True, "answer": full_answer, "sources": sources,
           "confidence": confidence, "confidence_score": score, "route": "rag"}
