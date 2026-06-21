"""Tests for app/memory.py — SQLite history and conversation tracking.

Each test runs against a fresh temp database so they don't touch the real
memory.db and stay independent.
"""
import importlib

import pytest


@pytest.fixture
def mem(tmp_path, monkeypatch):
    memory = importlib.import_module("app.memory")
    # Point the module at an isolated DB and reset the cached connection.
    monkeypatch.setattr(memory, "_DB_PATH", tmp_path / "test_mem.db")
    monkeypatch.setattr(memory, "_conn", None)
    yield memory
    if memory._conn is not None:
        memory._conn.close()
        memory._conn = None


def test_save_and_get_history_is_ordered(mem):
    mem.save_turn("c1", "q1", "a1")
    mem.save_turn("c1", "q2", "a2")
    hist = mem.get_history("c1")
    assert hist == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_history_window_is_limited(mem):
    # MAX_TURNS pairs are kept for the LLM context window.
    for i in range(mem.MAX_TURNS + 3):
        mem.save_turn("c1", f"q{i}", f"a{i}")
    hist = mem.get_history("c1")
    assert len(hist) == mem.MAX_TURNS * 2
    # get_messages returns the FULL history, not the window.
    assert len(mem.get_messages("c1")) == (mem.MAX_TURNS + 3) * 2


def test_unknown_session_is_empty(mem):
    assert mem.get_history("nope") == []
    assert mem.get_messages("nope") == []
    assert mem.list_conversations("nobody") == []


def test_conversation_created_with_title_from_first_question(mem):
    mem.save_turn("c1", "What is the no-show fee?", "It is $25.", owner_id="owner")
    mem.save_turn("c1", "And cancellations?", "24 hours ahead.", owner_id="owner")
    convos = mem.list_conversations("owner")
    assert len(convos) == 1
    assert convos[0]["conversation_id"] == "c1"
    assert convos[0]["title"] == "What is the no-show fee?"
    assert convos[0]["message_count"] == 4


def test_conversations_scoped_to_owner(mem):
    mem.save_turn("c1", "hi", "hello", owner_id="alice")
    mem.save_turn("c2", "hey", "hello", owner_id="bob")
    assert {c["conversation_id"] for c in mem.list_conversations("alice")} == {"c1"}
    assert {c["conversation_id"] for c in mem.list_conversations("bob")} == {"c2"}


def test_long_title_is_truncated(mem):
    long_q = "word " * 40
    mem.save_turn("c1", long_q, "ok", owner_id="owner")
    title = mem.list_conversations("owner")[0]["title"]
    assert title.endswith("…")
    assert len(title) == 49   # 48 chars + ellipsis


def test_save_turn_without_owner_stores_messages_but_no_conversation(mem):
    mem.save_turn("c1", "q", "a")
    assert len(mem.get_messages("c1")) == 2
    assert mem.list_conversations("owner") == []


def test_delete_conversation_removes_messages_and_metadata(mem):
    mem.save_turn("c1", "q", "a", owner_id="owner")
    removed = mem.delete_conversation("c1")
    assert removed == 2
    assert mem.get_messages("c1") == []
    assert mem.list_conversations("owner") == []


def test_clear_session_removes_messages(mem):
    mem.save_turn("c1", "q", "a")
    assert mem.clear_session("c1") == 2
    assert mem.get_history("c1") == []
