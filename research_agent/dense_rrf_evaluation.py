from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_core.embeddings import Embeddings

from research_agent.config import Settings
from research_agent.llm import build_embeddings
from research_agent.paper_evaluation import load_jsonl, select_questions_by_id
from research_agent.retrieval.vector_store import ChromaEvidenceStore


DEFAULT_PLANS_PATH = Path("evaluation/datasets/retrieval_plans.jsonl")
DEFAULT_RESULTS_PATH = Path("evaluation/results/dense-rrf-k12-top30-results.jsonl")
DEFAULT_QUERY_CACHE_PATH = Path("evaluation/results/retrieval-query-embeddings.jsonl")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, target: Path, attempts: int = 10) -> None:
    for attempt in range(1, attempts + 1):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(0.2 * attempt)


class QueryEmbeddingCache:
    def __init__(self, path: Path, model: str):
        self.path = path
        self.model = model
        self.records = load_jsonl(path) if path.exists() else []
        self.by_key = {
            str(record["cache_key"]): record
            for record in self.records
            if record.get("cache_key") and record.get("embedding")
        }

    def _key(self, question_id: str, query: str) -> str:
        raw = f"{self.model}\n{question_id}\n{query}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get_or_create(
        self,
        question_id: str,
        query: str,
        embeddings: Embeddings,
    ) -> tuple[list[float], bool]:
        key = self._key(question_id, query)
        cached = self.by_key.get(key)
        if cached:
            return [float(value) for value in cached["embedding"]], True
        vector = [float(value) for value in embeddings.embed_query(query)]
        record = {
            "cache_key": key,
            "embedding_model": self.model,
            "question_id": question_id,
            "query": query,
            "embedding": vector,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.records.append(record)
        self.by_key[key] = record
        _write_jsonl_atomic(self.path, self.records)
        return vector, False


def select_ingest_summary(summary_dir: Path, question_id: str) -> Path:
    matches = sorted(summary_dir.glob(f"pdf-ingest-{question_id}-*-summary.json"))
    if not matches:
        raise FileNotFoundError(f"No PDF ingest summary for {question_id}")
    return matches[0]


def _chunk_payload(chunk: Any) -> dict[str, Any]:
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


def fuse_dense_rankings(
    query_results: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for query_result in query_results:
        query_index = int(query_result["query_index"])
        query = str(query_result["query"])
        for item in query_result["chunks"]:
            chunk_id = str(item["chunk_id"])
            rank = int(item["rank"])
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
            entry["best_source_rank"] = min(entry["best_source_rank"], rank)
            entry["query_hits"].append(
                {
                    "query_index": query_index,
                    "query": query,
                    "rank": rank,
                    "dense_distance": item.get("dense_distance"),
                }
            )

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            int(item["best_source_rank"]),
            str(item["chunk_id"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["selected"] = rank <= top_k
        item["matched_query_count"] = len(item["query_hits"])
    return ranked


def run_dense_rrf_evaluation(
    settings: Settings,
    plans_path: Path = DEFAULT_PLANS_PATH,
    summary_dir: Path = Path("evaluation/results"),
    output_path: Path = DEFAULT_RESULTS_PATH,
    query_cache_path: Path = DEFAULT_QUERY_CACHE_PATH,
    question_ids: list[str] | None = None,
    k_per_query: int = 12,
    top_k: int = 30,
    rrf_k: int = 60,
    force: bool = False,
    embeddings: Embeddings | None = None,
) -> tuple[Path, Path]:
    if min(k_per_query, top_k, rrf_k) <= 0:
        raise ValueError("k_per_query, top_k, and rrf_k must be positive")
    all_plans = load_jsonl(plans_path)
    plans = select_questions_by_id(all_plans, question_ids)
    failed_plans = [p.get("question_id") for p in plans if p.get("status") != "ok"]
    if failed_plans:
        raise ValueError(f"Plans are not successful: {', '.join(map(str, failed_plans))}")

    configuration = {
        "method": "dense_rrf",
        "query_set": "original_question_plus_subquestions",
        "k_per_query": k_per_query,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "embedding_model": settings.qwen_embedding_model,
    }
    existing = load_jsonl(output_path) if output_path.exists() else []
    records_by_id = {
        str(record["question_id"]): record
        for record in existing
        if record.get("question_id")
    }
    order = [str(plan["question_id"]) for plan in all_plans]
    embedding_client = embeddings or build_embeddings(settings)
    cache = QueryEmbeddingCache(query_cache_path, settings.qwen_embedding_model)

    for plan_record in plans:
        question_id = str(plan_record["question_id"])
        previous = records_by_id.get(question_id)
        if (
            not force
            and previous
            and previous.get("status") == "ok"
            and previous.get("configuration") == configuration
        ):
            continue
        started = time.perf_counter()
        try:
            summary_path = select_ingest_summary(summary_dir, question_id)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("question") != plan_record.get("question"):
                raise ValueError(f"Question mismatch in {summary_path}")
            chroma_dir = Path(str(summary["chroma_persist_dir"]))
            if not chroma_dir.exists():
                raise FileNotFoundError(f"Missing Chroma directory: {chroma_dir}")
            store = ChromaEvidenceStore(chroma_dir, embedding_client)
            queries = [str(plan_record["question"])] + [
                str(value) for value in plan_record["plan"]["subquestions"]
            ]
            query_results: list[dict[str, Any]] = []
            cache_hits = 0
            for query_index, query in enumerate(queries):
                vector, was_cached = cache.get_or_create(question_id, query, embedding_client)
                cache_hits += int(was_cached)
                chunks = store.similarity_search_by_vector(vector, k=k_per_query)
                query_results.append(
                    {
                        "query_index": query_index,
                        "query_type": "original" if query_index == 0 else "subquestion",
                        "query": query,
                        "chunks": [
                            {
                                "rank": rank,
                                "chunk_id": chunk.chunk_id,
                                "dense_distance": chunk.score,
                                "chunk": _chunk_payload(chunk),
                            }
                            for rank, chunk in enumerate(chunks, start=1)
                        ],
                    }
                )
            ranked = fuse_dense_rankings(query_results, rrf_k=rrf_k, top_k=top_k)
            selected = ranked[:top_k]
            record = {
                "status": "ok",
                "question_id": question_id,
                "question": plan_record["question"],
                "configuration": configuration,
                "ingest_summary": str(summary_path),
                "chroma_persist_dir": str(chroma_dir),
                "query_count": len(queries),
                "query_embedding_cache_hits": cache_hits,
                "retrieved_occurrence_count": sum(len(r["chunks"]) for r in query_results),
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
        ordered = [records_by_id[qid] for qid in order if qid in records_by_id]
        _write_jsonl_atomic(output_path, ordered)

    successful = [record for record in records_by_id.values() if record.get("status") == "ok"]
    errors = [record for record in records_by_id.values() if record.get("status") == "error"]
    summary_path = output_path.with_name(output_path.stem + "-summary.json")
    _write_json_atomic(
        summary_path,
        {
            "method": "dense_rrf",
            "configuration": configuration,
            "successful_questions": len(successful),
            "failed_questions": len(errors),
            "failed_question_ids": [record["question_id"] for record in errors],
            "total_queries": sum(int(record["query_count"]) for record in successful),
            "mean_unique_candidates": (
                fmean(float(record["unique_candidate_count"]) for record in successful)
                if successful else 0.0
            ),
            "mean_selected_papers": (
                fmean(float(record["selected_paper_count"]) for record in successful)
                if successful else 0.0
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_path, summary_path
