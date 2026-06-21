"""Tests for the pure metric functions in eval/metrics.py."""
import pytest

from eval import metrics as M


def test_hit_at_k():
    assert M.hit_at_k(["a.txt", "b.txt"], ["b.txt"]) == 1.0
    assert M.hit_at_k(["a.txt"], ["b.txt"]) == 0.0
    assert M.hit_at_k(["a.txt"], []) == 0.0   # no expectation -> miss by convention


def test_recall_at_k():
    assert M.recall_at_k(["a.txt", "b.txt"], ["a.txt", "b.txt"]) == 1.0
    assert M.recall_at_k(["a.txt"], ["a.txt", "b.txt"]) == pytest.approx(0.5)
    assert M.recall_at_k(["a.txt"], []) is None


def test_reciprocal_rank():
    assert M.reciprocal_rank(["a.txt", "b.txt"], ["a.txt"]) == 1.0
    assert M.reciprocal_rank(["a.txt", "b.txt"], ["b.txt"]) == pytest.approx(0.5)
    assert M.reciprocal_rank(["a.txt", "b.txt"], ["c.txt"]) == 0.0


def test_keyword_coverage():
    assert M.keyword_coverage("The fee is $25", ["$25"]) == 1.0
    assert M.keyword_coverage("Bring your ID and card", ["id", "passport"]) == pytest.approx(0.5)
    assert M.keyword_coverage("anything", []) is None


def test_mean_ignores_none():
    assert M.mean([1.0, None, 0.0]) == pytest.approx(0.5)
    assert M.mean([None, None]) is None
    assert M.mean([]) is None


def test_pct_formatting():
    assert M.pct(1.0) == "100%"
    assert M.pct(0.755) == "76%"
    assert M.pct(None) == "—"
