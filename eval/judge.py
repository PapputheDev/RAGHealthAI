"""LLM-as-judge for reference-free RAG scoring.

Given a question, the retrieved context, and the model's answer, an LLM rates:
  - faithfulness: is every claim in the answer supported by the context?
  - relevance:    does the answer actually address the question?

This is the same idea as RAGAS faithfulness/answer-relevance, kept minimal and
dependency-free. It calls the same LLM the app uses, so it costs tokens — that is
why the runner gates it behind an explicit --judge flag.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict

from app.llm import LLMError, generate_answer

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented assistant. "
    "You will be given CONTEXT, a QUESTION, and the assistant's ANSWER. "
    "Judge only what is shown — do not use outside knowledge.\n\n"
    "Return ONLY compact JSON with this exact shape and no extra text:\n"
    '{"faithful": 0 or 1, "relevant": 0 or 1, "reason": "<=20 words"}\n\n'
    "faithful = 1 only if every factual claim in ANSWER is supported by CONTEXT.\n"
    "relevant = 1 only if ANSWER directly addresses QUESTION (a correct refusal "
    "to an unanswerable question is relevant)."
)


def _parse(raw: str) -> Dict[str, object]:
    # Prefer strict JSON; fall back to digit scraping so a chatty model still scores.
    try:
        obj = json.loads(raw.strip())
        return {
            "faithful": int(obj.get("faithful", 0)),
            "relevant": int(obj.get("relevant", 0)),
            "reason": str(obj.get("reason", ""))[:200],
        }
    except Exception:
        f = re.search(r'"?faithful"?\s*[:=]\s*([01])', raw)
        r = re.search(r'"?relevant"?\s*[:=]\s*([01])', raw)
        return {
            "faithful": int(f.group(1)) if f else 0,
            "relevant": int(r.group(1)) if r else 0,
            "reason": "unparsed: " + raw.strip()[:120],
        }


def judge(question: str, context: str, answer: str) -> Dict[str, object]:
    """Score one (question, context, answer) triple. Returns faithful/relevant/reason.

    On LLM failure returns zeros with an error reason so a single bad call does
    not abort the whole run.
    """
    user = (
        f"CONTEXT:\n{context or '(no context retrieved)'}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        raw = generate_answer(messages)
    except LLMError as exc:
        logger.warning("Judge LLM call failed: %s", exc)
        return {"faithful": 0, "relevant": 0, "reason": f"judge error: {exc}"}
    return _parse(raw)
