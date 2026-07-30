from __future__ import annotations

from typing import Any

import requests

from research_agent.config import Settings
from research_agent.schema import Paper


def qwen3_rerank_url(settings: Settings) -> str:
    if settings.qwen_rerank_base_url:
        return settings.qwen_rerank_base_url.rstrip("/") + "/reranks"
    if not settings.dashscope_workspace_id:
        raise ValueError(
            "DASHSCOPE_WORKSPACE_ID or QWEN_RERANK_BASE_URL is required when "
            "PAPER_RANKING_STRATEGY=qwen3_rerank"
        )
    return (
        f"https://{settings.dashscope_workspace_id}.cn-beijing.maas.aliyuncs.com"
        "/compatible-api/v1/reranks"
    )


def paper_rerank_document(paper: Paper) -> str:
    title = " ".join(paper.title.split())
    abstract = " ".join(paper.abstract.split())
    return f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"


def rerank_with_qwen3(
    question: str,
    papers: list[Paper],
    settings: Settings,
    *,
    session: Any = requests,
) -> tuple[list[Paper], int | None]:
    """Rerank one candidate pool in one request and preserve input-index mapping."""
    if not papers:
        return [], 0
    if not settings.dashscope_api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required when PAPER_RANKING_STRATEGY=qwen3_rerank"
        )

    request_payload = {
        "model": settings.qwen_rerank_model,
        "query": question,
        "documents": [paper_rerank_document(paper) for paper in papers],
        "top_n": len(papers),
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
    if len(results) != len(papers):
        raise RuntimeError(
            f"Qwen3 rerank returned {len(results)} results for {len(papers)} documents"
        )

    seen_indexes: set[int] = set()
    ranked: list[Paper] = []
    for rank, result in enumerate(
        sorted(results, key=lambda item: float(item["relevance_score"]), reverse=True),
        start=1,
    ):
        index = int(result["index"])
        if index < 0 or index >= len(papers) or index in seen_indexes:
            raise RuntimeError(f"Qwen3 rerank returned invalid document index: {index}")
        seen_indexes.add(index)
        ranked.append(
            papers[index].model_copy(
                update={
                    "rank": rank,
                    "reranker_score": float(result["relevance_score"]),
                    "rrf_score": None,
                }
            )
        )

    usage = payload.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    return ranked, int(total_tokens) if total_tokens is not None else None
