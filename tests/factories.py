"""Small helpers for building test fixtures/data.

Kept in a normal importable module (not conftest.py) so tests can import it the
same way under both `pytest` and `python -m pytest`.
"""
from __future__ import annotations

from typing import Any, Dict


def make_result(source: str, text: str = "chunk text", distance: float = 0.4,
                chunk_index: int = 0) -> Dict[str, Any]:
    """Build a SearchResult-shaped dict like vector_store.search_documents returns."""
    return {
        "id": f"{source}:{chunk_index}",
        "text": text,
        "metadata": {"source": source, "chunk_index": chunk_index},
        "distance": distance,
    }
