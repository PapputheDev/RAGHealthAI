"""Tests for prompt construction in app/prompts.py."""
from app.prompts import build_rag_messages, INSUFFICIENT_CONTEXT_MESSAGE, SYSTEM_PROMPT


def test_messages_include_context_and_question():
    msgs = build_rag_messages("CTX", "What is the fee?", [])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[-1]["role"] == "user"
    assert "CONTEXT:\nCTX" in msgs[-1]["content"]
    assert "QUESTION:\nWhat is the fee?" in msgs[-1]["content"]


def test_empty_context_omits_context_block():
    msgs = build_rag_messages("   ", "hi", [])
    user = msgs[-1]["content"]
    assert "CONTEXT" not in user
    assert user.startswith("QUESTION:")


def test_history_sits_between_system_and_user():
    history = [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]
    msgs = build_rag_messages("CTX", "now", history)
    assert msgs[0]["role"] == "system"
    assert msgs[1:3] == history
    assert msgs[-1]["content"].endswith("QUESTION:\nnow")


def test_fallback_message_is_in_system_prompt():
    # The strict instruction must reference the exact fallback string.
    assert INSUFFICIENT_CONTEXT_MESSAGE in SYSTEM_PROMPT
