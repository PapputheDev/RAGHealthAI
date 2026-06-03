from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

MAX_TURNS = 5  # Q&A pairs retained per session

_sessions: Dict[str, Deque[Dict[str, str]]] = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return the conversation history for a session as a messages list."""
    return list(_sessions.get(session_id, []))


def save_turn(session_id: str, question: str, answer: str) -> None:
    """Append a user/assistant turn to the session history."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_TURNS * 2)
    _sessions[session_id].append({"role": "user", "content": question})
    _sessions[session_id].append({"role": "assistant", "content": answer})
