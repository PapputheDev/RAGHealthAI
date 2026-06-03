"""Unit tests — no LLM or vector DB required."""
from __future__ import annotations

import pytest

from app.agent import _extract_department, _extract_modality, _needs_appointment_tool
from app.models import AskRequest, AskResponse, SourceReference
from app.rag import _build_context, _confidence_label, _extract_sources, _similarity_from_distance


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

class TestSimilarityFromDistance:
    def test_zero_distance_is_max_similarity(self):
        assert _similarity_from_distance(0.0) == 1.0

    def test_one_distance_is_zero_similarity(self):
        assert _similarity_from_distance(1.0) == 0.0

    def test_mid_distance(self):
        assert abs(_similarity_from_distance(0.4) - 0.6) < 1e-9

    def test_clamped_above_one(self):
        assert _similarity_from_distance(-0.5) == 1.0

    def test_clamped_below_zero(self):
        assert _similarity_from_distance(1.5) == 0.0

    def test_invalid_input_returns_zero(self):
        assert _similarity_from_distance("bad") == 0.0  # type: ignore[arg-type]


class TestConfidenceLabel:
    def test_high(self):
        assert _confidence_label(0.9) == "high"

    def test_medium(self):
        assert _confidence_label(0.5) == "medium"

    def test_low(self):
        assert _confidence_label(0.1) == "low"

    def test_boundary_high(self):
        assert _confidence_label(0.7) == "high"

    def test_boundary_medium(self):
        assert _confidence_label(0.4) == "medium"


# ---------------------------------------------------------------------------
# Source extraction and context building
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = [
    {
        "id": "abc",
        "text": "Patients may request refills via telehealth.",
        "metadata": {"source": "telehealth_policy.txt", "chunk_index": 2},
        "distance": 0.2,
    },
    {
        "id": "def",
        "text": "Standard refill timeline is 72 hours.",
        "metadata": {"source": "medication_refill_policy.txt", "chunk_index": 0},
        "distance": 0.35,
    },
]


class TestExtractSources:
    def test_returns_unique_sources(self):
        # Duplicate source should be deduplicated.
        dup_results = _SAMPLE_RESULTS + [
            {
                "id": "ghi",
                "text": "Additional telehealth info.",
                "metadata": {"source": "telehealth_policy.txt", "chunk_index": 3},
                "distance": 0.4,
            }
        ]
        sources = _extract_sources(dup_results)
        docs = [s["document"] for s in sources]
        assert docs.count("telehealth_policy.txt") == 1

    def test_source_has_document_and_chunk_keys(self):
        sources = _extract_sources(_SAMPLE_RESULTS)
        for s in sources:
            assert "document" in s
            assert "chunk" in s

    def test_empty_results_returns_empty(self):
        assert _extract_sources([]) == []


class TestBuildContext:
    def test_includes_source_header(self):
        ctx = _build_context(_SAMPLE_RESULTS)
        assert "telehealth_policy.txt" in ctx

    def test_includes_chunk_text(self):
        ctx = _build_context(_SAMPLE_RESULTS)
        assert "refills via telehealth" in ctx

    def test_empty_results_returns_empty_string(self):
        assert _build_context([]) == ""


# ---------------------------------------------------------------------------
# Agent routing
# ---------------------------------------------------------------------------

class TestNeedsAppointmentTool:
    @pytest.mark.parametrize("q", [
        "Can I book an appointment?",
        "I need to schedule a doctor visit",
        "Are there available slots this week?",
        "I want to see a doctor tomorrow",
    ])
    def test_routes_to_appointment(self, q: str):
        assert _needs_appointment_tool(q) is True

    @pytest.mark.parametrize("q", [
        "What is the HIPAA policy?",
        "How do I request a medication refill?",
        "What are the telehealth guidelines?",
    ])
    def test_routes_to_rag(self, q: str):
        assert _needs_appointment_tool(q) is False


class TestExtractDepartment:
    def test_cardiology(self):
        assert _extract_department("I need a cardiology appointment") == "cardiology"

    def test_heart_keyword(self):
        assert _extract_department("My heart has been racing") == "cardiology"

    def test_skin_keyword(self):
        assert _extract_department("I have a skin rash") == "dermatology"

    def test_default(self):
        assert _extract_department("I just need a checkup") == "primary_care"


class TestExtractModality:
    def test_video(self):
        assert _extract_modality("Can I do a video call?") == "video"

    def test_telehealth(self):
        assert _extract_modality("I prefer telehealth") == "video"

    def test_in_person(self):
        assert _extract_modality("I want an in-person visit") == "in_person"

    def test_default_is_video(self):
        assert _extract_modality("I need to see someone") == "video"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TestAskRequest:
    def test_valid(self):
        req = AskRequest(question="What is the no-show fee?")
        assert req.question == "What is the no-show fee?"

    def test_strips_whitespace(self):
        req = AskRequest(question="  hello  ")
        assert req.question == "hello"

    def test_empty_raises(self):
        with pytest.raises(Exception):
            AskRequest(question="   ")


class TestAskResponse:
    def test_valid_confidence_levels(self):
        for level in ("low", "medium", "high"):
            r = AskResponse(answer="Test answer", confidence=level)
            assert r.confidence == level

    def test_invalid_confidence_raises(self):
        with pytest.raises(Exception):
            AskResponse(answer="Test answer", confidence="unknown")

    def test_sources_default_empty(self):
        r = AskResponse(answer="Test answer")
        assert r.sources == []

    def test_route_optional(self):
        r = AskResponse(answer="Test answer")
        assert r.route is None

    def test_source_reference_fields(self):
        src = SourceReference(document="policy.txt", chunk="Some excerpt...")
        assert src.document == "policy.txt"
        assert src.chunk == "Some excerpt..."
