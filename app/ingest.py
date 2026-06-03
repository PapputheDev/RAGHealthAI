from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config import get_settings
from .vector_store import COLLECTION_NAME, add_documents

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class IngestStats:
	data_dir: str
	collection_name: str
	files_found: int
	files_loaded: int
	chars_total: int
	chunks_total: int
	documents_added: int


def _default_data_dir() -> Path:
	# Repo structure: project_root/data
	return Path(__file__).resolve().parents[1] / "data"


def _read_text_files(data_dir: Path) -> List[Tuple[Path, str]]:
	if not data_dir.exists() or not data_dir.is_dir():
		raise FileNotFoundError(f"data directory not found: {data_dir}")

	files = sorted(data_dir.glob("*.txt"))
	logger.info("Found %d .txt files in %s", len(files), str(data_dir))

	loaded: List[Tuple[Path, str]] = []
	for path in files:
		try:
			text = path.read_text(encoding="utf-8")
			if text.strip():
				loaded.append((path, text))
			else:
				logger.warning("Skipping empty file: %s", path.name)
		except UnicodeDecodeError:
			# Fallback for files saved with a BOM or different encoding.
			text = path.read_text(encoding="utf-8-sig", errors="replace")
			if text.strip():
				loaded.append((path, text))
			else:
				logger.warning("Skipping unreadable/empty file: %s", path.name)

	return loaded


def _chunk_documents(
	docs: Sequence[Tuple[Path, str]],
	*,
	chunk_size: int = CHUNK_SIZE,
	chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
	try:
		from langchain_text_splitters import RecursiveCharacterTextSplitter
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"Missing dependency: langchain-text-splitters. Install it with `pip install -r requirements.txt`."
		) from exc

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	)

	texts: List[str] = []
	metadatas: List[Dict[str, Any]] = []
	ids: List[str] = []

	for file_path, full_text in docs:
		chunks = splitter.split_text(full_text)
		for idx, chunk in enumerate(chunks):
			chunk_text = chunk.strip()
			if not chunk_text:
				continue

			texts.append(chunk_text)
			metadatas.append(
				{
					"source": file_path.name,
					"chunk_index": idx,
					"chunk_size": chunk_size,
					"chunk_overlap": chunk_overlap,
				}
			)
			ids.append(f"{file_path.stem}:{idx}")

	return texts, metadatas, ids


def ingest(data_dir: Optional[PathLike] = None) -> Dict[str, Any]:
	"""Ingest all `.txt` files from the data folder into ChromaDB.

	Pipeline:
	- Read all .txt files under `data/`
	- Chunk with RecursiveCharacterTextSplitter (sizes from config or defaults)
	- Generate embeddings (SentenceTransformers via embeddings.py)
	- Store in persistent ChromaDB collection

	Returns:
		A dict with ingestion statistics.
	"""

	settings = get_settings()
	chunk_size = settings.chunk_size
	chunk_overlap = settings.chunk_overlap

	target_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
	target_dir = target_dir.expanduser().resolve(strict=False)

	loaded_docs = _read_text_files(target_dir)
	chars_total = sum(len(text) for _, text in loaded_docs)

	texts, metadatas, ids = _chunk_documents(loaded_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	logger.info(
		"Chunked %d files into %d chunks (chunk_size=%d overlap=%d)",
		len(loaded_docs),
		len(texts),
		chunk_size,
		chunk_overlap,
	)

	documents_added = add_documents(texts, metadatas=metadatas, ids=ids)
	stats = IngestStats(
		data_dir=str(target_dir),
		collection_name=COLLECTION_NAME,
		files_found=len(sorted(target_dir.glob("*.txt"))),
		files_loaded=len(loaded_docs),
		chars_total=chars_total,
		chunks_total=len(texts),
		documents_added=documents_added,
	)

	logger.info("Ingestion complete: %s", asdict(stats))
	return asdict(stats)


def ingest_file_list(paths: Sequence[Path]) -> Dict[str, Any]:
	"""Ingest a specific list of already-saved files (used by the upload endpoint).

	Supports .txt (read directly) and .pdf (text extracted via pypdf).
	"""

	settings = get_settings()
	docs: List[Tuple[Path, str]] = []

	for path in paths:
		ext = path.suffix.lower()
		try:
			if ext == ".pdf":
				text = _extract_pdf_text(path)
			else:
				try:
					text = path.read_text(encoding="utf-8")
				except UnicodeDecodeError:
					text = path.read_text(encoding="utf-8-sig", errors="replace")

			if text.strip():
				docs.append((path, text))
			else:
				logger.warning("Skipping empty file: %s", path.name)
		except Exception:
			logger.exception("Failed to read uploaded file: %s", path.name)

	texts, metadatas, ids = _chunk_documents(
		docs,
		chunk_size=settings.chunk_size,
		chunk_overlap=settings.chunk_overlap,
	)

	documents_added = add_documents(texts, metadatas=metadatas, ids=ids) if texts else 0

	logger.info(
		"Upload ingest complete: files=%d chunks=%d docs_added=%d",
		len(docs), len(texts), documents_added,
	)
	return {
		"files_ingested": len(docs),
		"chunks_total": len(texts),
		"documents_added": documents_added,
	}


def _extract_pdf_text(path: Path) -> str:
	try:
		from pypdf import PdfReader
	except ImportError as exc:
		raise RuntimeError(
			"pypdf is required for PDF uploads. Install it: pip install pypdf"
		) from exc

	reader = PdfReader(str(path))
	pages = [page.extract_text() or "" for page in reader.pages]
	return "\n\n".join(p.strip() for p in pages if p.strip())


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)
	result = ingest()
	print(result)

