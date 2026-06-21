"""Tests for the retrieval confidence math in app/rag.py.

These pin down the squared-L2 -> cosine conversion fix (1 - dist/2) and the
band thresholds, so a regression would fail loudly.
"""
import pytest

from app.rag import (
    _best_similarity,
    _confidence_from_results,
    _confidence_label,
    _extract_sources,
    _is_insufficient_answer,
    _similarity_from_distance,
)
from app.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from tests.factories import make_result


@pytest.mark.parametrize("distance,expected", [
    (0.0, 1.0),    # identical
    (0.4, 0.8),    # strong match (cosine 0.8)
    (0.6, 0.7),    # good match -> exactly the 'high' threshold
    (1.0, 0.5),    # weak
    (4.0, 0.0),    # opposite, clamps at 0
])
def test_similarity_from_distance(distance, expected):
    assert _similarity_from_distance(distance) == pytest.approx(expected)


def test_similarity_clamps_and_handles_bad_input():
    assert _similarity_from_distance(-1.0) == 1.0       # negative distance clamps high
    assert _similarity_from_distance("bad") == 0.0      # unparseable -> 0


@pytest.mark.parametrize("score,label", [
    (1.0, "high"), (0.7, "high"), (0.69, "medium"),
    (0.4, "medium"), (0.39, "low"), (0.0, "low"),
])
def test_confidence_label_thresholds(score, label):
    assert _confidence_label(score) == label


def test_confidence_from_results_averages():
    results = [make_result("a.txt", distance=0.4), make_result("b.txt", distance=0.6)]
    # sims are 0.8 and 0.7 -> mean 0.75
    assert _confidence_from_results(results) == pytest.approx(0.75)
    assert _confidence_from_results([]) == 0.0


def test_best_similarity_uses_max_not_mean():
    results = [make_result("a.txt", distance=1.0), make_result("b.txt", distance=0.4)]
    assert _best_similarity(results) == pytest.approx(0.8)
    assert _best_similarity([]) == 0.0


def test_is_insufficient_answer_matches_fallback():
    assert _is_insufficient_answer(INSUFFICIENT_CONTEXT_MESSAGE)
    assert _is_insufficient_answer("  " + INSUFFICIENT_CONTEXT_MESSAGE.upper() + "  ")
    assert _is_insufficient_answer(INSUFFICIENT_CONTEXT_MESSAGE.rstrip("."))
    assert not _is_insufficient_answer("The no-show fee is $25.")


def test_extract_sources_dedupes_and_truncates():
    long_text = "x" * 250
    results = [
        make_result("a.txt", text=long_text),
        make_result("a.txt", text="dup"),     # same source -> deduped
        make_result("b.txt", text="short"),
    ]
    sources = _extract_sources(results)
    assert [s["document"] for s in sources] == ["a.txt", "b.txt"]
    assert sources[0]["chunk"].endswith("...")
    assert len(sources[0]["chunk"]) == 203   # 200 chars + "..."
