from __future__ import annotations

from typing import Any

import requests

from scholar_trace.config import Settings
from scholar_trace.schema import EvidenceChunk, Paper


def qwen3_rerank_url(settings: Settings) -> str:
    if settings.qwen_rerank_base_url:
        return settings.qwen_rerank_base_url.rstrip("/") + "/reranks"
    if not settings.dashscope_workspace_id:
        raise ValueError(
            "DASHSCOPE_WORKSPACE_ID or QWEN_RERANK_BASE_URL is required for "
            "Qwen3-Rerank"
        )
    return (
        f"https://{settings.dashscope_workspace_id}.cn-beijing.maas.aliyuncs.com"
        "/compatible-api/v1/reranks"
    )


def paper_rerank_document(paper: Paper) -> str:
    title = " ".join(paper.title.split())
    abstract = " ".join(paper.abstract.split())
    return f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"


def evidence_chunk_rerank_document(chunk: EvidenceChunk) -> str:
    title = " ".join(chunk.title.split())
    text = " ".join(chunk.text.split())
    return f"Title: {title}\n\nPassage: {text}" if title else text


def rerank_documents_with_qwen3(
    query: str,
    documents: list[str],
    settings: Settings,
    *,
    top_n: int | None = None,
    session: Any = requests,
) -> tuple[list[tuple[int, float]], int | None]:
    """Return document indexes and scores from Qwen3-Rerank without an instruct."""
    if not documents:
        return [], 0
    requested_top_n = len(documents) if top_n is None else top_n
    if requested_top_n <= 0 or requested_top_n > len(documents):
        raise ValueError("top_n must be between 1 and the number of documents")
    if not settings.dashscope_api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required for Qwen3-Rerank"
        )

    request_payload = {
        "model": settings.qwen_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": requested_top_n,
    }
    response = session.post(
        qwen3_rerank_url(settings),
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json=request_payload,
        timeout=settings.qwen_rerank_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        message = payload.get("message", "response has no results list")
        raise RuntimeError(f"Qwen3 rerank failed: {message}")
    if len(results) != requested_top_n:
        raise RuntimeError(
            f"Qwen3 rerank returned {len(results)} results; expected {requested_top_n}"
        )

    seen_indexes: set[int] = set()
    ranked: list[tuple[int, float]] = []
    for result in sorted(
        results, key=lambda item: float(item["relevance_score"]), reverse=True
    ):
        index = int(result["index"])
        if index < 0 or index >= len(documents) or index in seen_indexes:
            raise RuntimeError(f"Qwen3 rerank returned invalid document index: {index}")
        seen_indexes.add(index)
        ranked.append((index, float(result["relevance_score"])))

    usage = payload.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    return ranked, int(total_tokens) if total_tokens is not None else None


def rerank_with_qwen3(
    question: str,
    papers: list[Paper],
    settings: Settings,
    *,
    top_n: int | None = None,
    session: Any = requests,
) -> tuple[list[Paper], int | None]:
    """Rerank one candidate pool in one request and preserve input-index mapping."""
    document_ranking, total_tokens = rerank_documents_with_qwen3(
        question,
        [paper_rerank_document(paper) for paper in papers],
        settings,
        top_n=top_n,
        session=session,
    )
    ranked: list[Paper] = []
    for rank, (index, score) in enumerate(document_ranking, start=1):
        ranked.append(
            papers[index].model_copy(
                update={
                    "rank": rank,
                    "reranker_score": score,
                    "rrf_score": None,
                }
            )
        )
    return ranked, total_tokens
