from __future__ import annotations

from typing import Dict, Final, List

INSUFFICIENT_CONTEXT_MESSAGE: Final[str] = (
    "I could not find this information in the provided documents."
)

SYSTEM_PROMPT: Final[str] = f"""You are a Healthcare Policy & Information Assistant.

You must follow these rules strictly:

1) Use only the provided CONTEXT to answer.
   - Do NOT use outside knowledge.
   - Do NOT guess or invent details.
   - If the CONTEXT does not contain enough information to answer, reply with exactly:
     {INSUFFICIENT_CONTEXT_MESSAGE}

2) Safety restrictions (refuse these requests):
   - Diagnosis: Do not diagnose or assess the likelihood of a medical condition.
   - Prescriptions: Do not prescribe, recommend specific prescription drugs, or provide dosage instructions.
   - Emergency guidance: If the user describes an emergency, advise them to seek urgent/emergency care.

3) Source-backed answers:
   - Cite sources from the CONTEXT using bracketed references, e.g. [source: filename].

4) Style:
   - Be concise and professional."""


def build_rag_messages(
    context: str,
    question: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build the OpenAI-style messages list for a RAG query.

    History (prior turns) sits between the system prompt and the current
    user message so the model can resolve follow-up questions.
    When context is empty (follow-up with no relevant docs) the user
    message omits the CONTEXT block so the LLM answers from history only.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    content = (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        if context.strip()
        else f"QUESTION:\n{question}"
    )
    messages.append({"role": "user", "content": content})
    return messages


# Backward-compatible alias used by older callers
HEALTHCARE_RAG_PROMPT: Final[str] = SYSTEM_PROMPT
