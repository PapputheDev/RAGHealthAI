"""Tests for the lightweight router in app/agent.py.

Covers greeting detection, appointment-tool routing, and the entity extraction
used by the synthetic appointment tool. None of these paths call the LLM.
"""
import pytest

from app import agent


@pytest.mark.parametrize("text,expected", [
    ("hi", True),
    ("Hello!", True),
    ("hey there", True),          # <=3 words and contains a greeting word
    ("thanks", True),
    ("What are my HIPAA rights?", False),
    ("book an appointment please", False),
])
def test_is_greeting(text, expected):
    assert agent._is_greeting(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("I want to book an appointment", True),
    ("schedule a doctor visit", True),
    ("what is the no-show fee?", False),
    ("how do refills work?", False),
])
def test_needs_appointment_tool(text, expected):
    assert agent._needs_appointment_tool(text) is expected


@pytest.mark.parametrize("text,dept", [
    ("I need a cardiology appointment", "cardiology"),
    ("skin rash dermatologist", "dermatology"),
    ("my child is sick", "pediatrics"),
    ("just a general checkup", "general_practice"),
    ("book something", "primary_care"),   # default
])
def test_extract_department(text, dept):
    assert agent._extract_department(text) == dept


@pytest.mark.parametrize("text,modality", [
    ("video visit", "video"),
    ("telehealth please", "video"),
    ("in person visit", "in_person"),
    ("at the clinic", "in_person"),
    ("book an appointment", "video"),     # default
])
def test_extract_modality(text, modality):
    assert agent._extract_modality(text) == modality


def test_handle_question_greeting_is_local():
    out = agent.handle_question("hello")
    assert out["route"] == "rag"
    assert out["result"]["confidence"] == "high"
    assert out["result"]["sources"] == []
    assert "Healthcare" in out["result"]["answer"]


def test_handle_question_appointment_returns_slots():
    out = agent.handle_question("book a cardiology appointment via video")
    assert out["route"] == "appointment_tool"
    slots = out["result"]["available_slots"]
    assert len(slots) == 3
    assert all(s["modality"] == "video" for s in slots)
    assert all(s["clinician_type"] == "cardiology" for s in slots)


def test_stream_greeting_yields_done_event():
    events = list(agent.handle_question_stream("hi"))
    done = [e for e in events if e.get("done")]
    assert done and done[0]["route"] == "rag"
    assert done[0]["confidence"] == "high"


def test_handle_question_rejects_empty():
    with pytest.raises(ValueError):
        agent.handle_question("  ")
