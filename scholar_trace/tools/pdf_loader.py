from pathlib import Path

import requests
from langchain_core.documents import Document
from pypdf import PdfReader

from scholar_trace.schema import Paper


def sanitize_utf8(text: str) -> str:
    """Replace unpaired Unicode surrogates emitted by some PDF extractors."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def paper_metadata(paper: Paper) -> dict:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "local_pdf_path": paper.local_pdf_path,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "source": paper.source,
    }


def document_from_abstract(paper: Paper) -> Document:
    text = sanitize_utf8(f"Title: {paper.title}\n\nAbstract: {paper.abstract}")
    return Document(
        page_content=text,
        metadata={**paper_metadata(paper), "page": None},
    )


def download_pdf(paper: Paper, directory: Path, timeout: int = 30) -> Path | None:
    if not paper.pdf_url:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{paper.paper_id}.pdf"
    if path.exists():
        return path
    response = requests.get(paper.pdf_url, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def documents_from_pdf(paper: Paper, pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    total_pages = len(reader.pages)
    for page_number, page in enumerate(reader.pages):
        text = sanitize_utf8(page.extract_text() or "")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    **paper_metadata(paper),
                    "source_path": str(pdf_path),
                    "page": page_number,
                    "total_pages": total_pages,
                },
            )
        )
    return docs

# 为了测试方便，可以return真正的pdf内容，也可以return一个由abstract构成的Document
def load_paper_documents(
    paper: Paper,
    pdf_dir: Path | None = None,
    use_pdf: bool = False,
    pdf_failure_policy: str = "fallback_abstract",
) -> list[Document]:
    if use_pdf and pdf_dir is not None:
        try:
            pdf_path = (
                Path(paper.local_pdf_path)
                if paper.local_pdf_path
                else download_pdf(paper, pdf_dir)
            )
            if paper.local_pdf_path and not pdf_path.is_file():
                raise FileNotFoundError(f"Local PDF does not exist: {pdf_path}")
            if pdf_path is not None:
                docs = documents_from_pdf(paper, pdf_path)
                if docs:
                    return docs
        except Exception:
            pass
        if pdf_failure_policy == "skip":
            return []
    return [document_from_abstract(paper)]
