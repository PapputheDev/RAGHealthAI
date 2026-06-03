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
	session_id: Optional[str] = Field(default=None, description="Client session ID for conversation memory")

	@field_validator("question")
	@classmethod
	def _non_empty_question(cls, value: str) -> str:
		q = value.strip()
		if not q:
			raise ValueError("question must not be empty")
		return q


class AskResponse(BaseModel):
	"""Response payload containing the answer and provenance."""

	answer: str = Field(..., min_length=1)
	sources: List[SourceReference] = Field(default_factory=list)
	confidence: ConfidenceLevel = Field("low", description="Confidence level: low | medium | high")
	route: Optional[str] = Field(default=None, description="Agent route taken: rag | appointment_tool")

	@field_validator("answer")
	@classmethod
	def _non_empty_answer(cls, value: str) -> str:
		a = value.strip()
		if not a:
			raise ValueError("answer must not be empty")
		return a

