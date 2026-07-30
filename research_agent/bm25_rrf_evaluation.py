from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

from research_agent.dense_rrf_evaluation import (
    DEFAULT_PLANS_PATH,
    _write_json_atomic,
    _write_jsonl_atomic,
    select_ingest_summary,
)
from research_agent.paper_evaluation import load_jsonl, select_questions_by_id
from research_agent.retrieval.bm25 import BM25Index, load_chroma_documents


DEFAULT_RESULTS_PATH = Path("evaluation/results/bm25-rrf-k15-top30-results.jsonl")


def _document_payload(document: Any) -> dict[str, Any]:
    metadata = dict(document.metadata)
    page = metadata.get("page")
    return {
        "chunk_id": str(metadata.get("chunk_id", document.id or "")),
        "paper_id": str(metadata.get("paper_id", "")),
        "title": str(metadata.get("title", "")),
        "text": document.page_content,
        "url": str(metadata.get("url", "")),
        "page": page if isinstance(page, int) else None,
        "section": metadata.get("section"),
        "metadata": metadata,
    }


def fuse_bm25_rankings(
    query_results: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for query_result in query_results:
        for item in query_result["chunks"]:
            rank = int(item["rank"])
            chunk_id = str(item["chunk_id"])
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
            entry["query_hits"].append(
                {
                    "query_index": int(query_result["query_index"]),
                    "query": str(query_result["query"]),
                    "rank": rank,
                    "bm25_score": float(item["bm25_score"]),
                }
            )
    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["rrf_score"]), int(item["best_source_rank"]), str(item["chunk_id"])),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["selected"] = rank <= top_k
        item["matched_query_count"] = len(item["query_hits"])
    return ranked


def run_bm25_rrf_evaluation(
    plans_path: Path = DEFAULT_PLANS_PATH,
    summary_dir: Path = Path("evaluation/results"),
    output_path: Path = DEFAULT_RESULTS_PATH,
    question_ids: list[str] | None = None,
    k_per_query: int = 15,
    top_k: int = 30,
    rrf_k: int = 60,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    force: bool = False,
) -> tuple[Path, Path]:
    if min(k_per_query, top_k, rrf_k) <= 0:
        raise ValueError("k_per_query, top_k, and rrf_k must be positive")
    all_plans = load_jsonl(plans_path)
    plans = select_questions_by_id(all_plans, question_ids)
    failed = [str(plan.get("question_id")) for plan in plans if plan.get("status") != "ok"]
    if failed:
        raise ValueError(f"Plans are not successful: {', '.join(failed)}")
    configuration = {
        "method": "bm25_rrf",
        "query_set": "original_question_plus_subquestions",
        "k_per_query": k_per_query,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_library": "bm25s",
        "bm25_library_version": version("bm25s"),
        "bm25_method": "lucene",
        "tokenizer": "lowercase_cs_terms_v1",
    }
    existing = load_jsonl(output_path) if output_path.exists() else []
    records_by_id = {str(row["question_id"]): row for row in existing if row.get("question_id")}
    order = [str(plan["question_id"]) for plan in all_plans]
    for plan_record in plans:
        question_id = str(plan_record["question_id"])
        previous = records_by_id.get(question_id)
        if not force and previous and previous.get("status") == "ok" and previous.get("configuration") == configuration:
            continue
        started = time.perf_counter()
        try:
            summary_path = select_ingest_summary(summary_dir, question_id)
            ingest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if ingest_summary.get("question") != plan_record.get("question"):
                raise ValueError(f"Question mismatch in {summary_path}")
            chroma_dir = Path(str(ingest_summary["chroma_persist_dir"]))
            if not chroma_dir.exists():
                raise FileNotFoundError(f"Missing Chroma directory: {chroma_dir}")
            documents = load_chroma_documents(chroma_dir)
            index = BM25Index(documents, k1=bm25_k1, b=bm25_b)
            queries = [str(plan_record["question"]), *map(str, plan_record["plan"]["subquestions"])]
            query_results = []
            for query_index, query in enumerate(queries):
                hits = index.search(query, k=k_per_query)
                query_results.append(
                    {
                        "query_index": query_index,
                        "query_type": "original" if query_index == 0 else "subquestion",
                        "query": query,
                        "chunks": [
                            {
                                "rank": rank,
                                "chunk_id": str(hit.document.metadata.get("chunk_id", hit.document.id or "")),
                                "bm25_score": hit.score,
                                "chunk": _document_payload(hit.document),
                            }
                            for rank, hit in enumerate(hits, start=1)
                        ],
                    }
                )
            ranked = fuse_bm25_rankings(query_results, rrf_k=rrf_k, top_k=top_k)
            selected = ranked[:top_k]
            record = {
                "status": "ok",
                "question_id": question_id,
                "question": plan_record["question"],
                "configuration": configuration,
                "ingest_summary": str(summary_path),
                "chroma_persist_dir": str(chroma_dir),
                "corpus_chunk_count": len(documents),
                "query_count": len(queries),
                "retrieved_occurrence_count": sum(len(row["chunks"]) for row in query_results),
                "unique_candidate_count": len(ranked),
                "selected_chunk_count": len(selected),
                "selected_paper_count": len({item["paper_id"] for item in selected}),
                "query_results": query_results,
                "ranked_chunks": ranked,
                "elapsed_seconds": time.perf_counter() - started,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            record = {
                "status": "error",
                "question_id": question_id,
                "question": plan_record.get("question"),
                "configuration": configuration,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        records_by_id[question_id] = record
        _write_jsonl_atomic(output_path, [records_by_id[qid] for qid in order if qid in records_by_id])
    successful = [row for row in records_by_id.values() if row.get("status") == "ok"]
    errors = [row for row in records_by_id.values() if row.get("status") == "error"]
    summary_path = output_path.with_name(output_path.stem + "-summary.json")
    _write_json_atomic(
        summary_path,
        {
            "method": "bm25_rrf",
            "configuration": configuration,
            "successful_questions": len(successful),
            "failed_questions": len(errors),
            "failed_question_ids": [row["question_id"] for row in errors],
            "total_queries": sum(int(row["query_count"]) for row in successful),
            "mean_unique_candidates": fmean(float(row["unique_candidate_count"]) for row in successful) if successful else 0.0,
            "mean_selected_papers": fmean(float(row["selected_paper_count"]) for row in successful) if successful else 0.0,
            "mean_elapsed_seconds": fmean(float(row["elapsed_seconds"]) for row in successful) if successful else 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_path, summary_path
