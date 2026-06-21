"""Pure metric functions for the RAG eval harness.

These are deliberately dependency-free and side-effect-free so they are easy to
unit test and reason about. Retrieval metrics operate on a *ranked, de-duplicated
list of source filenames* (best match first).
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def hit_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """1.0 if any expected source appears in the retrieved list, else 0.0."""
    if not expected:
        return 0.0
    exp = set(expected)
    return 1.0 if any(s in exp for s in retrieved) else 0.0


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> Optional[float]:
    """Fraction of expected sources that were retrieved. None if no expectation."""
    if not expected:
        return None
    ret = set(retrieved)
    found = sum(1 for s in set(expected) if s in ret)
    return found / len(set(expected))


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """1/rank of the first expected source in the ranked list (0 if absent)."""
    if not expected:
        return 0.0
    exp = set(expected)
    for i, src in enumerate(retrieved, start=1):
        if src in exp:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer: str, keywords: Sequence[str]) -> Optional[float]:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return None
    low = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean over the non-None values; None if the list is empty after filtering."""
    nums: List[float] = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def pct(value: Optional[float]) -> str:
    """Format a 0..1 ratio as a percentage string, or '—' when undefined."""
    return "—" if value is None else f"{value * 100:.0f}%"
