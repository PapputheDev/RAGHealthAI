from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import chromadb

from .config import get_settings
from .embeddings import get_embedding, get_embeddings

logger = logging.getLogger(__name__)


COLLECTION_NAME = "healthcare_docs"


class SearchResult(TypedDict):
	id: str
	text: str
	metadata: Dict[str, Any]
	distance: float


@lru_cache(maxsize=1)
def _get_client() -> chromadb.ClientAPI:
	# Cache the persistent client so repeated requests reuse the same ChromaDB
	# connection and storage path.
	settings = get_settings()
	path = str(settings.chroma_db_path)
	logger.info("Initializing ChromaDB persistent client path=%s", path)
	return chromadb.PersistentClient(path=path)


def _get_collection():
	client = _get_client()
	# Use get_or_create to keep the API simple.
	return client.get_or_create_collection(name=COLLECTION_NAME)


def add_documents(
	texts: Sequence[str],
	*,
	metadatas: Optional[Sequence[Dict[str, Any]]] = None,
	ids: Optional[Sequence[str]] = None,
) -> int:
	"""Add documents to the persistent ChromaDB collection.

	Args:
		texts: Document texts.
		metadatas: Optional per-document metadata dicts.
		ids: Optional per-document IDs; if omitted, UUIDs are generated.

	Returns:
		Number of documents added.
	"""

	if texts is None:
		raise ValueError("texts must not be None")
	if len(texts) == 0:
		return 0

	cleaned: List[str] = []
	for idx, text in enumerate(texts):
		if not isinstance(text, str) or not text.strip():
			raise ValueError(f"texts[{idx}] must be a non-empty string")
		cleaned.append(text)

	if metadatas is not None and len(metadatas) != len(cleaned):
		raise ValueError("metadatas must be the same length as texts")
	if ids is not None and len(ids) != len(cleaned):
		raise ValueError("ids must be the same length as texts")

	doc_ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in cleaned]
	doc_metadatas = list(metadatas) if metadatas is not None else [{} for _ in cleaned]

	logger.info("Embedding and adding %d documents to collection=%s", len(cleaned), COLLECTION_NAME)
	# Embeddings are generated before upsert because this collection stores
	# explicit vectors rather than relying on Chroma's default embedding function.
	vectors = get_embeddings(cleaned)

	collection = _get_collection()
	# upsert = insert new + update existing; makes re-ingestion idempotent
	collection.upsert(
		ids=doc_ids,
		documents=cleaned,
		metadatas=doc_metadatas,
		embeddings=vectors,
	)

	logger.info("Added %d documents to collection=%s", len(cleaned), COLLECTION_NAME)
	return len(cleaned)


def search_documents(query: str, *, n_results: int = 5) -> List[SearchResult]:
	"""Search for documents similar to the query.

	Args:
		query: Query string.
		n_results: Maximum number of results to return.

	Returns:
		A list of search results with id/text/metadata/distance.
	"""

	if not isinstance(query, str) or not query.strip():
		raise ValueError("query must be a non-empty string")
	if n_results <= 0:
		raise ValueError("n_results must be > 0")

	collection = _get_collection()
	query_vec = get_embedding(query)
	logger.debug("Searching collection=%s n_results=%d", COLLECTION_NAME, n_results)

	result = collection.query(
		query_embeddings=[query_vec],
		n_results=n_results,
		include=["documents", "metadatas", "distances"],
	)

	# Chroma returns nested lists because it supports batching multiple queries.
	# This app sends one query at a time, so we read the first result set.
	ids = (result.get("ids") or [[]])[0]
	docs = (result.get("documents") or [[]])[0]
	metas = (result.get("metadatas") or [[]])[0]
	dists = (result.get("distances") or [[]])[0]

	out: List[SearchResult] = []
	for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
		out.append(
			{
				"id": str(doc_id),
				"text": str(doc or ""),
				"metadata": dict(meta or {}),
				"distance": float(dist),
			}
		)
	return out


def list_sources() -> List[Dict[str, Any]]:
	"""Return the distinct source documents in the collection with chunk counts.

	Powers the document-management UI so users can see what's indexed and remove
	individual documents.
	"""

	collection = _get_collection()
	# Pull only metadata (no documents/embeddings) to keep this cheap even when
	# the collection is large.
	result = collection.get(include=["metadatas"])
	metas = result.get("metadatas") or []

	counts: Dict[str, int] = {}
	for meta in metas:
		source = str((meta or {}).get("source", "unknown"))
		counts[source] = counts.get(source, 0) + 1

	sources = [{"document": name, "chunks": n} for name, n in sorted(counts.items())]
	logger.info("Listed %d distinct source documents", len(sources))
	return sources


def delete_by_source(source: str) -> int:
	"""Delete every chunk belonging to a given source document.

	Returns the number of chunks removed.
	"""

	if not isinstance(source, str) or not source.strip():
		raise ValueError("source must be a non-empty string")

	collection = _get_collection()
	# Count first so we can report how many chunks were removed; Chroma's delete
	# does not return a count.
	existing = collection.get(where={"source": source}, include=[])
	removed = len((existing.get("ids") or []))
	if removed == 0:
		logger.info("No chunks found for source=%s — nothing to delete", source)
		return 0

	collection.delete(where={"source": source})
	logger.info("Deleted %d chunks for source=%s", removed, source)
	return removed


def count_documents() -> int:
	"""Return the number of stored documents in the collection."""

	collection = _get_collection()
	try:
		count = int(collection.count())
	except Exception:
		# Some older chroma versions may not support count() on certain backends.
		logger.exception("Failed to count documents")
		raise

	logger.info("Collection=%s document_count=%d", COLLECTION_NAME, count)
	return count


def delete_collection() -> None:
	"""Delete the entire collection (destructive)."""

	client = _get_client()
	logger.warning("Deleting collection=%s", COLLECTION_NAME)
	client.delete_collection(name=COLLECTION_NAME)
