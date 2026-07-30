from research_agent.schema import Paper
from research_agent.tools import pdf_loader
from research_agent.tools.pdf_loader import (
    document_from_abstract,
    documents_from_pdf,
    load_paper_documents,
    sanitize_utf8,
)


def test_sanitize_utf8_replaces_unpaired_surrogates():
    cleaned = sanitize_utf8("before\ud83eafter")

    assert cleaned == "before?after"
    assert cleaned.encode("utf-8")


def test_document_from_abstract_uses_paper_metadata():
    paper = Paper(
        paper_id="p1",
        title="A Paper",
        abstract="This is the abstract.",
        url="https://example.com/paper",
        pdf_url="https://example.com/paper.pdf",
        source="test",
    )

    doc = document_from_abstract(paper)

    assert doc.page_content.startswith("Title: A Paper")
    assert doc.metadata["paper_id"] == "p1"
    assert doc.metadata["pdf_url"] == "https://example.com/paper.pdf"
    assert doc.metadata["page"] is None


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakeReader:
    def __init__(self, path: str):
        self.path = path
        self.pages = [FakePage("First page text."), FakePage("")]


def test_documents_from_pdf_adds_page_metadata(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(pdf_loader, "PdfReader", FakeReader)

    paper = Paper(
        paper_id="p1",
        title="A Paper",
        url="https://example.com/paper",
        pdf_url="https://example.com/paper.pdf",
        source="test",
    )

    docs = documents_from_pdf(paper, pdf_path)

    assert len(docs) == 1
    assert docs[0].page_content == "First page text."
    assert docs[0].metadata["paper_id"] == "p1"
    assert docs[0].metadata["page"] == 0
    assert docs[0].metadata["total_pages"] == 2
    assert docs[0].metadata["source_path"] == str(pdf_path)


def test_load_paper_documents_falls_back_to_abstract_when_pdf_fails(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("research_agent.tools.pdf_loader.download_pdf", fail_download)
    paper = Paper(
        paper_id="p1",
        title="A Paper",
        abstract="Fallback abstract.",
        pdf_url="https://example.com/forbidden.pdf",
    )

    docs = load_paper_documents(paper, pdf_dir=tmp_path, use_pdf=True)

    assert len(docs) == 1
    assert "Fallback abstract." in docs[0].page_content
    assert docs[0].metadata["page"] is None


def test_load_paper_documents_skips_when_pdf_fails_in_strict_mode(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("research_agent.tools.pdf_loader.download_pdf", fail_download)
    paper = Paper(
        paper_id="p1",
        title="A Paper",
        abstract="Should not be used.",
        pdf_url="https://example.com/forbidden.pdf",
    )

    docs = load_paper_documents(
        paper,
        pdf_dir=tmp_path,
        use_pdf=True,
        pdf_failure_policy="skip",
    )

    assert docs == []
