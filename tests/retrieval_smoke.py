from __future__ import annotations

import logging
import sys
from pathlib import Path


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _similarity_from_distance(distance: float) -> float:
    """Convert Chroma distance into a human-friendly similarity score.

    Notes:
    - Chroma returns a "distance" value; smaller is more similar.
    - The exact scale depends on the distance metric.
    - For smoke testing, we provide a simple derived score: similarity = max(0, 1 - distance).
    """

    return max(0.0, 1.0 - float(distance))


def main() -> int:
    _configure_logging()
    _ensure_repo_on_path()

    logger = logging.getLogger("retrieval_smoke")

    from app.vector_store import count_documents, search_documents

    try:
        total = count_documents()
    except Exception:
        logger.exception("Failed to access ChromaDB. Did you run ingestion?")
        return 2

    if total == 0:
        logger.warning("No documents found in ChromaDB. Run: python .\\app\\ingest.py")
        return 0

    question = "What is the no-show fee and how late can I arrive before I may be rescheduled?"
    logger.info("Sample question: %s", question)

    results = search_documents(question, n_results=3)
    if not results:
        logger.warning("No search results returned")
        return 0

    print("\nTop 3 retrieved chunks:\n")
    for rank, item in enumerate(results, start=1):
        source = item.get("metadata", {}).get("source", "unknown")
        distance = float(item.get("distance", 0.0))
        similarity = _similarity_from_distance(distance)

        print(f"#{rank}")
        print(f"  source: {source}")
        print(f"  distance: {distance:.4f}")
        print(f"  similarity(1-distance): {similarity:.4f}")
        print("  chunk:")
        print("  " + str(item.get("text", "")).replace("\n", "\n  ").strip())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
