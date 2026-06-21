"""Shared pytest setup.

Provide a dummy API key so importing app config never fails in CI, and expose a
small helper for building fake retrieval results.
"""
import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


def make_result(source: str, text: str = "chunk text", distance: float = 0.4,
                chunk_index: int = 0) -> dict:
    """Build a SearchResult-shaped dict like vector_store.search_documents returns."""
    return {
        "id": f"{source}:{chunk_index}",
        "text": text,
        "metadata": {"source": source, "chunk_index": chunk_index},
        "distance": distance,
    }
