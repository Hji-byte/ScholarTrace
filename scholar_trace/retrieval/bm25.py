from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer
from langchain_chroma import Chroma
from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._+#-][a-z0-9+#-]+)*", re.IGNORECASE)


def tokenize_for_bm25(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def load_chroma_documents(
    persist_dir: Path,
    collection_name: str = "papers",
) -> list[Document]:
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    payload = store.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    for item_id, text, raw_metadata in zip(
        payload.get("ids", []),
        payload.get("documents", []),
        payload.get("metadatas", []),
        strict=True,
    ):
        metadata = dict(raw_metadata or {})
        metadata.setdefault("chunk_id", str(item_id))
        documents.append(
            Document(
                id=str(item_id),
                page_content=str(text or ""),
                metadata=metadata,
            )
        )
    documents.sort(key=lambda document: str(document.metadata.get("chunk_id", document.id or "")))
    return documents


@dataclass(frozen=True)
class BM25Hit:
    document: Document
    score: float


class BM25Index:
    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer(
            lower=True,
            splitter=tokenize_for_bm25,
            stopwords=[],
            stemmer=None,
        )
        corpus_tokens = self.tokenizer.tokenize(
            [document.page_content for document in documents],
            update_vocab=True,
            show_progress=False,
        )
        self.retriever = bm25s.BM25(
            method="lucene",
            k1=k1,
            b=b,
        )
        self.retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int = 15) -> list[BM25Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        query_tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            show_progress=False,
        )
        document_indices, scores = self.retriever.retrieve(
            query_tokens,
            k=min(k, len(self.documents)),
            show_progress=False,
        )
        hits: list[BM25Hit] = []
        for document_index, raw_score in zip(document_indices[0], scores[0], strict=True):
            score = float(raw_score)
            if score <= 0:
                continue
            hits.append(BM25Hit(document=self.documents[int(document_index)], score=score))
        hits.sort(
            key=lambda hit: (
                -hit.score,
                str(hit.document.metadata.get("chunk_id", hit.document.id or "")),
            )
        )
        return hits
