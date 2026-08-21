from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings

from scholar_trace.config import Settings
from scholar_trace.reranking.qwen3 import (
    evidence_chunk_rerank_document,
    rerank_documents_with_qwen3,
)
from scholar_trace.retrieval.bm25 import BM25Index, load_chroma_documents
from scholar_trace.retrieval.vector_store import ChromaEvidenceStore
from scholar_trace.schema import EvidenceChunk


SUPPORTED_RETRIEVERS = {"dense", "bm25"}


@dataclass(frozen=True)
class HybridRetrievalResult:
    chunks: list[EvidenceChunk]
    unique_candidate_count: int
    ranked_list_count: int
    reranker_tokens: int | None


def _chunk_payload(chunk: EvidenceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "title": chunk.title,
        "text": chunk.text,
        "url": chunk.url,
        "page": chunk.page,
        "section": chunk.section,
        "metadata": chunk.metadata,
    }


def _bm25_chunk(hit) -> EvidenceChunk:
    metadata = dict(hit.document.metadata)
    return EvidenceChunk(
        chunk_id=str(metadata.get("chunk_id", hit.document.id or "")),
        paper_id=str(metadata.get("paper_id", "")),
        title=str(metadata.get("title", "")),
        text=hit.document.page_content,
        url=str(metadata.get("url", "")),
        page=metadata.get("page") if isinstance(metadata.get("page"), int) else None,
        section=metadata.get("section"),
        score=hit.score,
        metadata=metadata,
    )


def fuse_hybrid_rankings(
    ranked_lists: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Fuse all query/retriever ranked lists with one global RRF pass."""
    if min(rrf_k, top_k) <= 0:
        raise ValueError("rrf_k and top_k must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        retriever = str(ranked_list.get("retriever", ""))
        if retriever not in SUPPORTED_RETRIEVERS:
            raise ValueError(f"Unsupported retriever: {retriever}")
        query_index = int(ranked_list["query_index"])
        query_type = str(ranked_list["query_type"])
        query = str(ranked_list["query"])
        seen_in_list: set[str] = set()
        for item in ranked_list.get("chunks", []):
            chunk_id = str(item["chunk_id"])
            if not chunk_id or chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            rank = int(item["rank"])
            if rank <= 0:
                raise ValueError(f"Invalid rank for {chunk_id}: {rank}")
            entry = fused.setdefault(
                chunk_id,
                {
                    **item["chunk"],
                    "rrf_score": 0.0,
                    "best_source_rank": rank,
                    "query_hits": [],
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["best_source_rank"] = min(int(entry["best_source_rank"]), rank)
            hit = {
                "query_index": query_index,
                "query_type": query_type,
                "query": query,
                "retriever": retriever,
                "rank": rank,
            }
            if retriever == "dense":
                hit["dense_distance"] = item.get("dense_distance")
            else:
                hit["bm25_score"] = item.get("bm25_score")
            entry["query_hits"].append(hit)

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            int(item["best_source_rank"]),
            str(item["chunk_id"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        retrievers = {str(hit["retriever"]) for hit in item["query_hits"]}
        queries = {int(hit["query_index"]) for hit in item["query_hits"]}
        item["rank"] = rank
        item["selected"] = rank <= top_k
        item["matched_list_count"] = len(item["query_hits"])
        item["matched_query_count"] = len(queries)
        item["matched_retriever_count"] = len(retrievers)
        item["matched_retrievers"] = sorted(retrievers)
    return ranked


def retrieve_hybrid_evidence(
    question: str,
    queries: list[str],
    settings: Settings,
    embeddings: Embeddings,
) -> HybridRetrievalResult:
    """Dense+BM25 Top-K per query, global equal-weight RRF, then Qwen3 rerank."""
    if settings.max_chunks_per_query <= 0:
        raise ValueError("max_chunks_per_query must be positive")
    if settings.retrieval_rrf_k <= 0:
        raise ValueError("retrieval_rrf_k must be positive")
    store = ChromaEvidenceStore(settings.chroma_persist_dir, embeddings)
    documents = load_chroma_documents(settings.chroma_persist_dir)
    bm25 = BM25Index(documents)
    ranked_lists: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        query_type = "original" if query_index == 0 else "subquestion"
        dense_chunks = store.similarity_search(query, k=settings.max_chunks_per_query)
        ranked_lists.append({
            "query_index": query_index,
            "query_type": query_type,
            "query": query,
            "retriever": "dense",
            "chunks": [
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "dense_distance": chunk.score,
                    "chunk": _chunk_payload(chunk),
                }
                for rank, chunk in enumerate(dense_chunks, start=1)
            ],
        })
        bm25_hits = bm25.search(query, k=settings.max_chunks_per_query)
        ranked_lists.append({
            "query_index": query_index,
            "query_type": query_type,
            "query": query,
            "retriever": "bm25",
            "chunks": [
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "bm25_score": hit.score,
                    "chunk": _chunk_payload(chunk),
                }
                for rank, hit in enumerate(bm25_hits, start=1)
                for chunk in [_bm25_chunk(hit)]
            ],
        })

    fused = fuse_hybrid_rankings(
        ranked_lists,
        rrf_k=settings.retrieval_rrf_k,
        top_k=len(documents),
    )
    candidates = [
        EvidenceChunk(
            chunk_id=str(item["chunk_id"]),
            paper_id=str(item["paper_id"]),
            title=str(item["title"]),
            text=str(item["text"]),
            url=str(item.get("url", "")),
            page=item.get("page"),
            section=item.get("section"),
            score=float(item["rrf_score"]),
            metadata={
                **dict(item.get("metadata") or {}),
                "rrf_rank": int(item["rank"]),
                "rrf_score": float(item["rrf_score"]),
                "matched_query_count": int(item["matched_query_count"]),
                "matched_list_count": int(item["matched_list_count"]),
                "matched_retrievers": list(item["matched_retrievers"]),
                "query_hits": item["query_hits"],
            },
        )
        for item in fused
    ]
    top_n = min(
        settings.max_evidence_chunks if settings.max_evidence_chunks > 0 else len(candidates),
        len(candidates),
    )
    ranking, reranker_tokens = rerank_documents_with_qwen3(
        question,
        [evidence_chunk_rerank_document(chunk) for chunk in candidates],
        settings,
        top_n=top_n,
    )
    selected = []
    for rank, (index, score) in enumerate(ranking, start=1):
        chunk = candidates[index]
        selected.append(
            chunk.model_copy(
                update={
                    "score": score,
                    "metadata": {
                        **chunk.metadata,
                        "reranker_rank": rank,
                        "reranker_score": score,
                    },
                }
            )
        )
    return HybridRetrievalResult(
        chunks=selected,
        unique_candidate_count=len(candidates),
        ranked_list_count=len(ranked_lists),
        reranker_tokens=reranker_tokens,
    )
