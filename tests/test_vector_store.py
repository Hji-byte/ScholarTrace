from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_agent.retrieval.vector_store import ChromaEvidenceStore


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 10), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10), 1.0]


class LimitedBatchEmbeddings(TinyEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > 10:
            raise ValueError("batch too large")
        return super().embed_documents(texts)


class FlakyEmbeddings(TinyEmbeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("temporary embedding connection failure")
        return super().embed_documents(texts)


class CountingEmbeddings(TinyEmbeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_documents(texts)


def test_chroma_retriever_returns_metadata(tmp_path: Path):
    store = ChromaEvidenceStore(tmp_path, TinyEmbeddings(), collection_name="test")
    store.add_documents(
        [
            Document(
                page_content="Speculative decoding uses a draft model.",
                metadata={
                    "chunk_id": "c1",
                    "paper_id": "p1",
                    "title": "Spec Decoding",
                    "url": "https://example.com",
                },
            )
        ]
    )
    chunks = store.similarity_search("draft model", k=1)
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].paper_id == "p1"


def test_chroma_vector_query_returns_metadata(tmp_path: Path):
    embeddings = TinyEmbeddings()
    store = ChromaEvidenceStore(tmp_path, embeddings, collection_name="vector-query")
    store.add_documents(
        [
            Document(
                page_content="retrieval augmented generation",
                metadata={"chunk_id": "c1", "paper_id": "p1", "title": "RAG"},
            )
        ]
    )
    chunks = store.similarity_search_by_vector(
        embeddings.embed_query("retrieval"),
        k=1,
    )
    assert chunks[0].chunk_id == "c1"
    assert isinstance(chunks[0].score, float)


def test_chroma_add_documents_batches_embedding_requests(tmp_path: Path):
    store = ChromaEvidenceStore(tmp_path, LimitedBatchEmbeddings(), collection_name="batched")
    documents = [
        Document(
            page_content=f"Document {index}",
            metadata={
                "chunk_id": f"c{index}",
                "paper_id": "p1",
                "title": "Batch Test",
                "url": "https://example.com",
            },
        )
        for index in range(25)
    ]

    count = store.add_documents(documents, batch_size=10)

    assert count == 25


def test_chroma_add_documents_retries_transient_embedding_errors(tmp_path: Path):
    embeddings = FlakyEmbeddings()
    store = ChromaEvidenceStore(tmp_path, embeddings, collection_name="retry")
    documents = [
        Document(
            page_content="Retry me.",
            metadata={
                "chunk_id": "retry-c1",
                "paper_id": "p1",
                "title": "Retry Test",
                "url": "https://example.com",
            },
        )
    ]

    count = store.add_documents(
        documents,
        retry_attempts=2,
        retry_backoff_seconds=0,
    )

    assert count == 1
    assert embeddings.calls == 2


def test_chroma_add_documents_skips_ids_already_indexed(tmp_path: Path):
    embeddings = CountingEmbeddings()
    store = ChromaEvidenceStore(tmp_path, embeddings, collection_name="resume")
    document = Document(
        page_content="Embed only once.",
        metadata={
            "chunk_id": "resume-c1",
            "paper_id": "p1",
            "title": "Resume Test",
            "url": "https://example.com",
        },
    )

    store.add_documents([document])
    store.add_documents([document])

    assert embeddings.calls == 1
    assert store.vector_store._collection.count() == 1
