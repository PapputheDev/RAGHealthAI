from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterator, List, Literal, Optional

from .rag import answer_question, answer_question_stream

logger = logging.getLogger(__name__)

Route = Literal["appointment_tool", "rag"]

APPOINTMENT_KEYWORDS = (
    "appointment",
    "booking",
    "book",
    "schedule",
    "doctor visit",
    "available slot",
    "see a doctor",
)

_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "yo",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you",
}

_GREETING_RESPONSE = (
    "Hello! I'm a Healthcare Policy & Information Assistant. "
    "Ask me anything about appointment policies, medication refills, "
    "insurance, HIPAA guidelines, or telehealth — I'll answer from the knowledge base."
)


def _is_greeting(question: str) -> bool:
    """Return True for short small-talk that has no place in the RAG pipeline."""
    q = question.lower().strip().rstrip("!?.")
    if q in _GREETINGS:
        return True
    # also catches "hi there", "hey hello", etc.
    words = set(q.split())
    return len(words) <= 3 and bool(words & _GREETINGS)

_DEPARTMENT_MAP: dict[str, list[str]] = {
    "cardiology":      ["cardiology", "cardiologist", "heart", "cardiac"],
    "dermatology":     ["dermatology", "dermatologist", "skin"],
    "orthopedics":     ["orthopedics", "orthopedic", "bone", "joint", "fracture"],
    "pediatrics":      ["pediatrics", "pediatric", "paediatric", "children", "child"],
    "neurology":       ["neurology", "neurologist", "brain", "migraine"],
    "general_practice":["general", "primary care", "gp", "physician", "family doctor"],
}

_MODALITY_KEYWORDS = {
    "video":     ["video", "telehealth", "virtual", "online", "remote"],
    "in_person": ["in person", "in-person", "office", "clinic", "onsite", "on-site"],
}


def _needs_appointment_tool(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in APPOINTMENT_KEYWORDS)


def _extract_department(question: str) -> str:
    q = question.lower()
    for dept, keywords in _DEPARTMENT_MAP.items():
        if any(kw in q for kw in keywords):
            return dept
    return "primary_care"


def _extract_modality(question: str) -> str:
    q = question.lower()
    for modality, keywords in _MODALITY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return modality
    return "video"


@dataclass(frozen=True)
class AppointmentSlot:
    start: str
    duration_minutes: int
    clinician_type: str
    modality: str


@dataclass(frozen=True)
class AppointmentToolResult:
    message: str
    available_slots: List[AppointmentSlot]


def appointment_tool(
    *,
    preferred_date: Optional[date] = None,
    clinician_type: str = "primary_care",
    modality: str = "video",
) -> Dict[str, Any]:
    base_day = preferred_date or (date.today() + timedelta(days=1))
    start_time = datetime.combine(base_day, datetime.min.time()).replace(hour=9, minute=0)
    slots = [
        AppointmentSlot(
            start=(start_time + timedelta(hours=h)).isoformat(timespec="minutes"),
            duration_minutes=20,
            clinician_type=clinician_type,
            modality=modality,
        )
        for h in (0, 2, 4)
    ]
    return asdict(AppointmentToolResult(
        message="Here are the next available appointment slots (synthetic).",
        available_slots=slots,
    ))


def _format_appointment_answer(tool_result: Dict[str, Any]) -> str:
    message = str(tool_result.get("message", ""))
    slots = tool_result.get("available_slots", [])
    if slots:
        lines = "\n".join(
            f"- {s.get('start')} ({s.get('duration_minutes')} min, {s.get('modality')})"
            for s in slots
        )
        return f"{message}\n\nAvailable slots:\n{lines}"
    return message


def handle_question(
    question: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Route question to appointment tool or RAG. Returns JSON-serialisable dict."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if _is_greeting(question):
        logger.info("Routing to greeting handler")
        return {
            "route": "rag",
            "input": {"question": question},
            "result": {"answer": _GREETING_RESPONSE, "sources": [], "confidence": "high"},
        }

    if _needs_appointment_tool(question):
        department = _extract_department(question)
        modality = _extract_modality(question)
        logger.info("Routing to appointment tool department=%s modality=%s", department, modality)
        tool_result = appointment_tool(clinician_type=department, modality=modality)
        return {"route": "appointment_tool", "input": {"question": question}, "result": tool_result}

    logger.info("Routing to RAG")
    rag_result = answer_question(question, history=history or [])
    return {"route": "rag", "input": {"question": question}, "result": rag_result}


def handle_question_stream(
    question: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream question routing results.

    Yields {"token": "..."} events for RAG answers, then a final
    {"done": True, ...metadata} event for both routes.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if _is_greeting(question):
        logger.info("Stream: greeting handler")
        yield {"done": True, "answer": _GREETING_RESPONSE,
               "sources": [], "confidence": "high", "route": "rag"}
        return

    if _needs_appointment_tool(question):
        department = _extract_department(question)
        modality = _extract_modality(question)
        logger.info("Stream: appointment tool department=%s modality=%s", department, modality)
        tool_result = appointment_tool(clinician_type=department, modality=modality)
        answer = _format_appointment_answer(tool_result)
        yield {"done": True, "answer": answer,
               "sources": [], "confidence": "high", "route": "appointment_tool"}
        return

    logger.info("Stream: routing to RAG")
    yield from answer_question_stream(question, history=history or [])
