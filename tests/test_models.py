"""Validation tests for the Pydantic models in app/models.py."""
import pytest
from pydantic import ValidationError

from app.models import AskRequest, AskResponse, ConversationSummary


def test_ask_request_trims_and_requires_question():
    assert AskRequest(question="  hello  ").question == "hello"
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_ask_request_optional_ids():
    req = AskRequest(question="hi", session_id="c1", owner_id="o1")
    assert req.session_id == "c1" and req.owner_id == "o1"
    assert AskRequest(question="hi").owner_id is None


def test_ask_response_defaults_and_score_bounds():
    resp = AskResponse(answer="ok")
    assert resp.confidence == "low"
    assert resp.confidence_score == 0.0
    with pytest.raises(ValidationError):
        AskResponse(answer="ok", confidence_score=1.5)   # > 1.0
    with pytest.raises(ValidationError):
        AskResponse(answer="ok", confidence="great")      # not a valid level


def test_ask_response_rejects_empty_answer():
    with pytest.raises(ValidationError):
        AskResponse(answer="   ")


def test_conversation_summary_requires_id():
    c = ConversationSummary(conversation_id="c1", title="t", updated_at=1.0, message_count=2)
    assert c.message_count == 2
    with pytest.raises(ValidationError):
        ConversationSummary(conversation_id="", title="t", updated_at=1.0, message_count=0)
