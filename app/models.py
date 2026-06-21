from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ConfidenceLevel = Literal["low", "medium", "high"]


class SourceReference(BaseModel):
	"""Reference to a source used to answer a question."""

	document: str = Field(..., min_length=1, description="Source document filename")
	chunk: Optional[str] = Field(default=None, description="Excerpt from the source chunk")


class AskRequest(BaseModel):
	"""Client request to ask a question."""

	question: str = Field(..., min_length=1)
	# session_id is optional; when present it enables short-term conversation
	# history for follow-up questions. It doubles as the conversation id.
	session_id: Optional[str] = Field(default=None, description="Conversation ID for memory")
	# owner_id ties a conversation to a browser/user so it appears in their history.
	owner_id: Optional[str] = Field(default=None, description="Stable client/owner ID")

	@field_validator("question")
	@classmethod
	def _non_empty_question(cls, value: str) -> str:
		q = value.strip()
		if not q:
			raise ValueError("question must not be empty")
		return q


class SessionRequest(BaseModel):
	"""Request carrying only a session identifier (e.g. to clear its history)."""

	session_id: str = Field(..., min_length=1, description="Client session ID")


class AskResponse(BaseModel):
	"""Response payload containing the answer and provenance."""

	answer: str = Field(..., min_length=1)
	# Sources are empty for appointment-tool answers and greeting responses.
	sources: List[SourceReference] = Field(default_factory=list)
	confidence: ConfidenceLevel = Field("low", description="Confidence level: low | medium | high")
	confidence_score: float = Field(
		default=0.0, ge=0.0, le=1.0,
		description="Numeric retrieval confidence (0..1) backing the confidence level",
	)
	route: Optional[str] = Field(default=None, description="Agent route taken: rag | appointment_tool")

	@field_validator("answer")
	@classmethod
	def _non_empty_answer(cls, value: str) -> str:
		a = value.strip()
		if not a:
			raise ValueError("answer must not be empty")
		return a


class IndexedDocument(BaseModel):
	"""A source document currently present in the vector store."""

	document: str = Field(..., min_length=1, description="Source document filename")
	chunks: int = Field(..., ge=0, description="Number of indexed chunks for this document")


class DocumentsResponse(BaseModel):
	"""List of documents currently indexed in the knowledge base."""

	documents: List[IndexedDocument] = Field(default_factory=list)
	total_chunks: int = Field(0, ge=0, description="Total number of indexed chunks")


class ChatMessage(BaseModel):
	"""A single stored message in a conversation."""

	role: str = Field(..., description="'user' or 'assistant'")
	content: str = Field(..., description="Message text")


class ConversationSummary(BaseModel):
	"""Lightweight metadata for one saved conversation."""

	conversation_id: str = Field(..., min_length=1)
	title: str = Field(...)
	updated_at: float = Field(..., description="Unix timestamp of the last activity")
	message_count: int = Field(0, ge=0)


class ConversationsResponse(BaseModel):
	"""An owner's list of saved conversations."""

	conversations: List[ConversationSummary] = Field(default_factory=list)


class ConversationMessagesResponse(BaseModel):
	"""Full message history for a single conversation."""

	conversation_id: str = Field(..., min_length=1)
	messages: List[ChatMessage] = Field(default_factory=list)
