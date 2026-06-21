from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


class Settings(BaseSettings):
	"""Application configuration loaded from environment variables.

	Notes:
	- Call `get_settings()` to load `.env` (via python-dotenv) then validate settings.
	- Secrets are never logged.
	"""

	# We load `.env` ourselves via python-dotenv, so BaseSettings should only read from env.
	model_config = SettingsConfigDict(
		env_file=None,
		extra="ignore",
		case_sensitive=False,
		validate_default=True,
	)

	# Aliases keep environment variable names conventional while exposing
	# Pythonic attribute names to the application code.
	openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
	model_name: str = Field("meta-llama/llama-3.3-70b-instruct", alias="MODEL_NAME")

	# Optional shared API key gating the HTTP API. When unset (the default),
	# authentication is disabled so local/demo use needs no extra config. When
	# set, callers must send a matching `X-API-Key` header.
	app_api_key: Optional[str] = Field(None, alias="APP_API_KEY")

	chunk_size: int = Field(800, alias="CHUNK_SIZE", ge=1)
	chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP", ge=0)

	chroma_db_path: Path = Field(Path("./chroma_db"), alias="CHROMA_DB_PATH")

	@field_validator("openrouter_api_key", mode="before")
	@classmethod
	def _validate_openrouter_api_key(cls, value: object) -> str:
		if value is None:
			raise ValueError("OPENROUTER_API_KEY is required")
		api_key = str(value).strip()
		if not api_key:
			raise ValueError("OPENROUTER_API_KEY must not be empty")
		return api_key

	@field_validator("model_name", mode="before")
	@classmethod
	def _validate_model_name(cls, value: object) -> str:
		if value is None:
			raise ValueError("MODEL_NAME is required")
		model = str(value).strip()
		if not model:
			raise ValueError("MODEL_NAME must not be empty")
		return model

	@field_validator("chroma_db_path", mode="before")
	@classmethod
	def _validate_chroma_db_path(cls, value: object) -> Path:
		if value is None:
			raise ValueError("CHROMA_DB_PATH is required")
		path = Path(str(value)).expanduser()
		# Resolve for consistent logging/behavior; do not require existence.
		return path.resolve(strict=False)

	@field_validator("chunk_overlap")
	@classmethod
	def _validate_chunk_overlap(cls, overlap: int, info) -> int:  # type: ignore[no-untyped-def]
		chunk_size = info.data.get("chunk_size")
		if isinstance(chunk_size, int) and overlap >= chunk_size:
			raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
		return overlap


def load_environment(dotenv_path: Optional[PathLike] = None) -> bool:
	"""Load environment variables from a `.env` file.

	Returns True if a dotenv file was found and loaded.
	"""

	# Do not override existing environment variables; deployed environments
	# should win over local .env values.
	loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
	if loaded:
		logger.debug("Loaded environment variables from .env")
	else:
		logger.debug("No .env file found (or nothing loaded)")
	return loaded


@lru_cache(maxsize=1)
def get_settings(dotenv_path: Optional[PathLike] = None) -> Settings:
	"""Get validated settings (cached).

	Raises:
		pydantic.ValidationError: if required variables are missing/invalid.
	"""

	# Settings are cached so request handlers do not repeatedly parse and
	# validate the environment.
	load_environment(dotenv_path)
	try:
		settings = Settings()  # reads from environment
	except ValidationError as exc:
		logger.error("Invalid configuration: %s", exc)
		raise

	logger.info(
		"Config loaded: MODEL_NAME=%s CHUNK_SIZE=%s CHUNK_OVERLAP=%s CHROMA_DB_PATH=%s",
		settings.model_name,
		settings.chunk_size,
		settings.chunk_overlap,
		str(settings.chroma_db_path),
	)
	return settings
