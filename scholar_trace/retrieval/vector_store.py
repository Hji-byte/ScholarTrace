from pathlib import Path
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from scholar_trace.schema import EvidenceChunk


TRANSIENT_EMBEDDING_ERROR_NAMES = {
    "ConnectionError",
    "ConnectTimeout",
    "ProtocolError",
    "ReadTimeout",
    "RemoteDisconnected",
    "Timeout",
}


def is_transient_embedding_error(exc: Exception) -> bool:
    names = {type(exc).__name__}
    names.update(type(parent).__name__ for parent in type(exc).__mro__)
    if names & TRANSIENT_EMBEDDING_ERROR_NAMES:
        return True
    text = str(exc).lower()
    return "connection" in text or "remote end closed" in text


class ChromaEvidenceStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection_name: str = "papers"):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 20,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 5.0,
    ) -> int:
        if not documents:
            return 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            ids = [str(doc.metadata["chunk_id"]) for doc in batch]
            existing = self.vector_store.get_by_ids(ids)
            existing_ids = {str(doc.id) for doc in existing if doc.id}
            pending = [
                doc
                for doc in batch
                if str(doc.metadata["chunk_id"]) not in existing_ids
            ]
            if not pending:
                continue
            pending_ids = [str(doc.metadata["chunk_id"]) for doc in pending]
            for attempt in range(1, retry_attempts + 1):
                try:
                    self.vector_store.add_documents(pending, ids=pending_ids)
                    break
                except Exception as exc:
                    if attempt >= retry_attempts or not is_transient_embedding_error(exc):
                        raise
                    time.sleep(retry_backoff_seconds * attempt)
        return len(documents)

    def similarity_search(self, query: str, k: int = 12) -> list[EvidenceChunk]:
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return self._to_evidence_chunks(results)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 12,
    ) -> list[EvidenceChunk]:
        results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding,
            k=k,
        )
        return self._to_evidence_chunks(results)

    @staticmethod
    def _to_evidence_chunks(
        results: list[tuple[Document, float]],
    ) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for doc, score in results:
            metadata = dict(doc.metadata)
            chunks.append(
                EvidenceChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    paper_id=str(metadata.get("paper_id", "")),
                    title=str(metadata.get("title", "")),
                    text=doc.page_content,
                    url=str(metadata.get("url", "")),
                    page=metadata.get("page") if isinstance(metadata.get("page"), int) else None,
                    section=metadata.get("section"),
                    score=float(score),
                    metadata=metadata,
                )
            )
        return chunks
