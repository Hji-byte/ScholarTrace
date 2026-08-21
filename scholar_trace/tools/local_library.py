from __future__ import annotations

import hashlib
import re
from itertools import islice
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from scholar_trace.schema import Paper
from scholar_trace.tools.pdf_loader import sanitize_utf8

LOCAL_RERANK_MAX_PAGES = 2
LOCAL_RERANK_MAX_CHARS = 4000


def _pdf_paths(library_path: Path) -> list[Path]:
    path = library_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Library file must be a PDF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Library path does not exist: {path}")
    return sorted(
        (item for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf"),
        key=lambda item: str(item).lower(),
    )


def _content_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"local-{digest.hexdigest()[:20]}"


def _metadata_value(metadata: Any, key: str) -> str:
    if not metadata:
        return ""
    value = metadata.get(key, "")
    return sanitize_utf8(str(value)).strip() if value else ""


def _metadata_year(metadata: Any) -> int | None:
    creation_date = _metadata_value(metadata, "/CreationDate")
    match = re.search(r"(?:D:)?((?:19|20)\d{2})", creation_date)
    return int(match.group(1)) if match else None


def _metadata_authors(metadata: Any) -> list[str]:
    author = _metadata_value(metadata, "/Author")
    if not author:
        return []
    parts = re.split(r"\s*(?:;|\band\b)\s*", author)
    return [part for part in parts if part]


def _local_rerank_text(reader: Any) -> str:
    """Extract an abstract, or a short opening passage, for paper-level reranking."""
    page_texts: list[str] = []
    for page in islice(getattr(reader, "pages", []), LOCAL_RERANK_MAX_PAGES):
        try:
            text = sanitize_utf8(page.extract_text() or "").strip()
        except Exception:
            continue
        if text:
            page_texts.append(text)
    opening = "\n".join(page_texts).strip()
    if not opening:
        return ""

    abstract_match = re.search(r"(?i)\babstract\b\s*[:—-]?\s*", opening)
    if abstract_match:
        abstract = opening[abstract_match.end() :]
        next_section = re.search(
            r"(?im)^\s*(?:keywords?|index terms|(?:1|i)[.\s]+introduction|introduction)"
            r"\s*[:—.-]?\s*",
            abstract,
        )
        if next_section:
            abstract = abstract[: next_section.start()]
        abstract = " ".join(abstract.split()).strip()
        if abstract:
            return abstract[:LOCAL_RERANK_MAX_CHARS]

    return " ".join(opening.split())[:LOCAL_RERANK_MAX_CHARS].strip()


def paper_from_local_pdf(path: Path) -> Paper:
    resolved = path.expanduser().resolve()
    metadata: Any = None
    rerank_text = ""
    try:
        reader = PdfReader(str(resolved))
        metadata = reader.metadata
        rerank_text = _local_rerank_text(reader)
    except Exception:
        # A malformed metadata section should not prevent the full parser from
        # attempting the PDF later during Ingest.
        metadata = None
    title = _metadata_value(metadata, "/Title") or resolved.stem
    paper_id = _content_id(resolved)
    return Paper(
        paper_id=paper_id,
        source_id=paper_id,
        title=title,
        authors=_metadata_authors(metadata),
        year=_metadata_year(metadata),
        abstract=rerank_text,
        source="local",
        local_pdf_path=str(resolved),
    )


def discover_local_papers(library_path: Path) -> list[Paper]:
    paths = _pdf_paths(library_path)
    if not paths:
        raise ValueError(f"No PDF files found in library path: {library_path}")
    papers: list[Paper] = []
    seen_ids: set[str] = set()
    for path in paths:
        paper = paper_from_local_pdf(path)
        if paper.paper_id not in seen_ids:
            seen_ids.add(paper.paper_id)
            papers.append(paper)
    return papers
