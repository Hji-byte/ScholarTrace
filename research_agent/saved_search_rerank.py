from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.agent.search import RankedPaper, add_ranked_results, candidate_papers
from research_agent.config import Settings
from research_agent.paper_evaluation import (
    DEFAULT_GOLD_PATH,
    DEFAULT_K_VALUES,
    DEFAULT_QUESTIONS_PATH,
    aggregate_metrics,
    evaluate_ranked_papers,
    index_by_question_id,
    load_jsonl,
)
from research_agent.reranking.qwen3 import rerank_with_qwen3
from research_agent.schema import Paper


def candidates_from_saved_search(
    search_records: list[dict[str, Any]],
) -> dict[str, list[Paper]]:
    rankings_by_question: dict[str, dict[str, RankedPaper]] = defaultdict(dict)
    for record in search_records:
        if record.get("status") != "ok":
            continue
        question_id = str(record["question_id"])
        query_index = int(record["query_index"])
        papers = []
        for raw_paper in record.get("papers", []):
            paper_data = dict(raw_paper)
            paper_data.pop("source_rank", None)
            papers.append(Paper.model_validate(paper_data))
        add_ranked_results(rankings_by_question[question_id], papers, query_index)
    return {
        question_id: candidate_papers(rankings)
        for question_id, rankings in rankings_by_question.items()
    }


def run_saved_search_rerank(
    settings: Settings,
    raw_search_path: Path,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
    output_dir: Path = Path("evaluation/results"),
    top_k: int = 20,
) -> tuple[Path, Path]:
    if top_k < max(DEFAULT_K_VALUES):
        raise ValueError(f"top_k must be at least {max(DEFAULT_K_VALUES)}")

    questions = load_jsonl(questions_path)
    questions_by_id = index_by_question_id(questions, questions_path)
    gold_by_id = index_by_question_id(load_jsonl(gold_path), gold_path)
    candidates_by_id = candidates_from_saved_search(load_jsonl(raw_search_path))
    missing = [
        question_id
        for question_id in questions_by_id
        if question_id not in candidates_by_id
    ]
    if missing:
        raise ValueError(f"No saved candidates for: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_id = f"qwen3-rerank-{stamp}"
    results_path = output_dir / f"{experiment_id}-results.jsonl"
    summary_path = output_dir / f"{experiment_id}-summary.json"
    records: list[dict[str, Any]] = []

    for question_id, question_record in questions_by_id.items():
        question = str(question_record["question"])
        candidates = candidates_by_id[question_id]
        started = time.perf_counter()
        try:
            ranked, total_tokens = rerank_with_qwen3(
                question,
                candidates,
                settings,
            )
            selected = ranked[:top_k]
            metrics, matches = evaluate_ranked_papers(
                selected,
                list(gold_by_id[question_id].get("key_papers", [])),
            )
            record = {
                "status": "ok",
                "experiment_id": experiment_id,
                "question_id": question_id,
                "domain": question_record.get("domain"),
                "topic": question_record.get("topic"),
                "question_type": question_record.get("question_type"),
                "difficulty": question_record.get("difficulty"),
                "year_from": question_record.get("year_from"),
                "year_to": question_record.get("year_to"),
                "question": question,
                "candidate_count": len(candidates),
                "latency_seconds": round(time.perf_counter() - started, 6),
                "reranker_tokens": total_tokens,
                "metrics": metrics,
                "gold_matches": matches,
                "papers": [paper.model_dump(mode="json") for paper in selected],
            }
        except Exception as exc:
            record = {
                "status": "error",
                "experiment_id": experiment_id,
                "question_id": question_id,
                "question": question,
                "candidate_count": len(candidates),
                "latency_seconds": round(time.perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful = [record for record in records if record["status"] == "ok"]
    summary = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_raw_search_path": str(raw_search_path),
        "question_count": len(records),
        "successful_count": len(successful),
        "error_count": len(records) - len(successful),
        "top_k": top_k,
        "model": settings.qwen_rerank_model,
        "total_reranker_tokens": sum(
            int(record.get("reranker_tokens") or 0) for record in successful
        ),
        "macro_metrics": aggregate_metrics(records),
        "errors": [
            {"question_id": record["question_id"], "error": record["error"]}
            for record in records
            if record["status"] == "error"
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results_path, summary_path
