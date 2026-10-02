"""Document ingestion: extract text, chunk it, embed it, store it in ChromaDB."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from . import config
from .store import get_collection


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number, text)] for a PDF, TXT or MD file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        return pages
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path.name} (use .pdf, .txt or .md)")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking (character-based with overlap, cut on sentence/paragraph boundaries)
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    text = clean(text)
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # prefer to cut at a paragraph, then sentence, then space
            for sep in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # never start a chunk in the middle of a word
        while start < len(text) and not text[start - 1].isspace():
            start += 1
    return chunks


def build_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    for page, text in extract_pages(path):
        for i, piece in enumerate(chunk_text(text)):
            cid = f"{file_hash}_p{page or 0}_c{i}"
            chunks.append(Chunk(id=cid, text=piece, source=path.name, page=page))
    return chunks


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def ingest_files(paths: Iterable[Path], replace: bool = True) -> dict:
    """Index files into the vector store. Returns a small report.

    replace=True removes previous chunks of a file with the same name first,
    so re-uploading an updated document does not leave stale passages behind.
    """
    collection = get_collection()
    report = {"files": 0, "chunks": 0, "skipped": []}
    for path in paths:
        path = Path(path)
        try:
            chunks = build_chunks(path)
        except Exception as exc:  # unreadable / unsupported file
            report["skipped"].append(f"{path.name}: {exc}")
            continue
        if not chunks:
            report["skipped"].append(f"{path.name}: no extractable text (scanned PDF? run OCR first)")
            continue
        if replace:
            collection.delete(where={"source": path.name})
        # Chroma accepts batches; keep them modest for large documents
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "page": c.page or 0} for c in batch],
            )
        report["files"] += 1
        report["chunks"] += len(chunks)
    return report


def list_sources() -> list[str]:
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]}) if data["metadatas"] else []


def delete_source(name: str) -> None:
    get_collection().delete(where={"source": name})
