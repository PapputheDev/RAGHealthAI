from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

MAX_TURNS = 5  # Q&A pairs fed back to the LLM as context per conversation

# Conversation history is persisted to a small SQLite database so it survives
# process restarts and can be shared across workers. Two tables:
#   messages       — every user/assistant turn, keyed by conversation id
#   conversations  — one row per saved chat (title + timestamps), keyed by owner
_DB_PATH = Path(__file__).resolve().parent.parent / "memory.db"

# SQLite connections are not safe to share across threads, so we guard the
# module-level connection with a lock. FastAPI runs sync endpoints in a thread
# pool, so concurrent access is real even for a small app.
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL
            )
            """
        )
        # A conversation's id IS the session_id used in the messages table, so a
        # single chat's turns and its metadata line up without extra joins.
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                owner_id        TEXT NOT NULL,
                title           TEXT NOT NULL DEFAULT 'New chat',
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            )
            """
        )
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_owner ON conversations(owner_id)")
        _conn.commit()
        logger.info("Conversation memory ready at %s", _DB_PATH)
    return _conn


def _title_from(question: str) -> str:
    # Use the first question as the chat title, collapsed and trimmed.
    text = " ".join(str(question).strip().split())
    if not text:
        return "New chat"
    return text[:48] + "…" if len(text) > 48 else text


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return the recent conversation history (LLM context window) for a chat."""
    if not session_id:
        return []
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            # Pull the most recent messages, then re-order them chronologically.
            # Limit is MAX_TURNS*2 because each turn stores a user + assistant row.
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, MAX_TURNS * 2),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def get_messages(conversation_id: str) -> List[Dict[str, str]]:
    """Return the FULL message history for a conversation, oldest first.

    Unlike get_history this is not limited to the LLM context window — it is
    used to re-render a saved chat in the UI.
    """
    if not conversation_id:
        return []
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def _upsert_conversation(owner_id: str, conversation_id: str, first_question: str) -> None:
    # Create the conversation row on its first turn (title from the first
    # question); otherwise just bump its updated_at so it sorts to the top.
    now = time.time()
    conn = _get_conn()
    exists = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()
    if exists:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, owner_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, owner_id, _title_from(first_question), now, now),
        )
    conn.commit()


def save_turn(session_id: str, question: str, answer: str, owner_id: str | None = None) -> None:
    """Append a user/assistant turn and (if owner_id is given) track the chat."""
    if not session_id:
        return
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", question),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", answer),
        )
        conn.commit()
        if owner_id:
            _upsert_conversation(owner_id, session_id, question)


def list_conversations(owner_id: str) -> List[Dict[str, object]]:
    """Return an owner's saved conversations, most recently updated first."""
    if not owner_id:
        return []
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT c.conversation_id, c.title, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = c.conversation_id)
            FROM conversations c
            WHERE c.owner_id = ?
            ORDER BY c.updated_at DESC
            """,
            (owner_id,),
        ).fetchall()
    return [
        {"conversation_id": r[0], "title": r[1], "updated_at": float(r[2]), "message_count": int(r[3])}
        for r in rows
    ]


def delete_conversation(conversation_id: str) -> int:
    """Delete a conversation and all of its messages. Returns messages removed."""
    if not conversation_id:
        return 0
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (conversation_id,))
        removed = cur.rowcount
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    logger.info("Deleted conversation (%d messages removed)", removed)
    return removed


def clear_session(session_id: str) -> int:
    """Delete all stored messages for a session id (does not touch metadata)."""
    if not session_id:
        return 0
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        removed = cur.rowcount
    logger.info("Cleared %d messages for session", removed)
    return removed
