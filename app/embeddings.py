from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Sequence

logger = logging.getLogger(__name__)


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _safe_preview(text: str, limit: int = 120) -> str:
	compact = " ".join(text.split())
	return compact if len(compact) <= limit else compact[:limit] + "…"


@lru_cache(maxsize=1)
def _get_model():
	"""Singleton loader for the embedding model.

	Uses LRU cache to ensure the model is loaded only once per process.
	"""

	try:
		from sentence_transformers import SentenceTransformer  # type: ignore
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"Missing dependency: sentence-transformers. Install it with `pip install -r requirements.txt`."
		) from exc

	logger.info("Loading embedding model: %s", MODEL_NAME)
	# Loading SentenceTransformers is relatively expensive, so the cached helper
	# keeps one model instance alive for the process.
	model = SentenceTransformer(MODEL_NAME)
	return model


def get_embedding(text: str) -> List[float]:
	"""Embed a single string into a vector.

	Args:
		text: Input text.

	Returns:
		Embedding vector as a list of floats.
	"""

	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a non-empty string")

	model = _get_model()
	logger.debug("Embedding text preview=%r", _safe_preview(text))

	vector = model.encode(
		[text],
		convert_to_numpy=True,
		normalize_embeddings=True,
		show_progress_bar=False,
	)[0]
	# Convert numpy values to plain Python floats so Chroma and JSON tooling can
	# consume them without numpy-specific serialization issues.
	return vector.astype(float).tolist()


def get_embeddings(texts: Sequence[str]) -> List[List[float]]:
	"""Embed a batch of strings.

	Args:
		texts: Sequence of input texts.

	Returns:
		List of embedding vectors.
	"""

	if texts is None:
		raise ValueError("texts must not be None")

	cleaned: List[str] = []
	for idx, item in enumerate(texts):
		if not isinstance(item, str) or not item.strip():
			raise ValueError(f"texts[{idx}] must be a non-empty string")
		cleaned.append(item)

	if not cleaned:
		return []

	model = _get_model()
	logger.debug("Embedding batch size=%d", len(cleaned))

	vectors = model.encode(
		cleaned,
		convert_to_numpy=True,
		normalize_embeddings=True,
		show_progress_bar=False,
	)
	# Match the single-embedding path by returning serializable Python lists.
	return [vec.astype(float).tolist() for vec in vectors]
