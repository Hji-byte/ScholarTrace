from langchain_core.documents import Document

from research_agent.retrieval.chunking import chunk_documents


def test_chunk_metadata_preserved():
    docs = [
        Document(
            page_content="Speculative decoding reduces latency. " * 80,
            metadata={"paper_id": "p1", "title": "Paper One", "url": "u"},
        )
    ]
    chunks = chunk_documents(docs, chunk_size=120, chunk_overlap=10)
    assert chunks
    assert chunks[0].metadata["paper_id"] == "p1"
    assert chunks[0].metadata["chunk_id"].startswith("p1-chunk-")

