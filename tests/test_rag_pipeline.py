"""Tests for app/rag.py answer logic with retrieval and the LLM mocked.

Covers the two behaviours we added: the relevance floor (skip the LLM and
abstain when the best match is too weak) and dropping sources when the model
returns the insufficient-context fallback.
"""
import app.rag as rag
from app.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from tests.conftest import make_result


def _patch(monkeypatch, results, answer="A real, grounded answer."):
    """Mock retrieval + generation. Returns a dict tracking LLM call count."""
    calls = {"generate": 0}

    def fake_search(question, n_results=3):
        return results

    def fake_generate(messages):
        calls["generate"] += 1
        return answer

    monkeypatch.setattr(rag, "search_documents", fake_search)
    monkeypatch.setattr(rag, "generate_answer", fake_generate)
    return calls


def test_below_relevance_floor_skips_llm_and_abstains(monkeypatch):
    # best similarity 0.2 (distance 1.6) is below RELEVANCE_FLOOR (0.35)
    weak = [make_result("a.txt", distance=1.6)]
    calls = _patch(monkeypatch, weak)

    resp = rag.answer_question("totally unrelated question")

    assert resp["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert resp["sources"] == []
    assert resp["confidence"] == "low"
    assert calls["generate"] == 0   # LLM must not be called


def test_strong_match_returns_answer_with_sources(monkeypatch):
    strong = [make_result("appointment_policy.txt", distance=0.4)]  # sim 0.8
    calls = _patch(monkeypatch, strong, answer="The no-show fee is $25.")

    resp = rag.answer_question("What is the no-show fee?")

    assert resp["answer"] == "The no-show fee is $25."
    assert [s["document"] for s in resp["sources"]] == ["appointment_policy.txt"]
    assert resp["confidence"] == "high"
    assert resp["confidence_score"] >= 0.7
    assert calls["generate"] == 1


def test_insufficient_answer_drops_sources(monkeypatch):
    # Retrieval is strong enough to call the LLM, but the model itself abstains.
    strong = [make_result("a.txt", distance=0.4)]
    _patch(monkeypatch, strong, answer=INSUFFICIENT_CONTEXT_MESSAGE)

    resp = rag.answer_question("something the docs don't cover")

    assert resp["sources"] == []
    assert resp["confidence"] == "low"


def test_stream_below_floor_emits_fallback(monkeypatch):
    weak = [make_result("a.txt", distance=1.6)]
    _patch(monkeypatch, weak)

    events = list(rag.answer_question_stream("unrelated"))
    done = [e for e in events if e.get("done")]

    assert done and done[0]["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert done[0]["sources"] == []


def test_empty_question_raises():
    import pytest
    with pytest.raises(ValueError):
        rag.answer_question("   ")
