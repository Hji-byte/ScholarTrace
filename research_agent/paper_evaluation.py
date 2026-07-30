from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any

from research_agent.agent.planner import plan_research
from research_agent.agent.search import RRF_K, search_node
from research_agent.config import Settings
from research_agent.db.database import ResearchDatabase
from research_agent.llm import build_chat_model
from research_agent.schema import Paper
from research_agent.tools.arxiv_search import (
    ARXIV_MAX_RETRIES,
    ARXIV_SORT_BY,
    ARXIV_SORT_ORDER,
    ARXIV_TIMEOUT_SECONDS,
    arxiv_source_id,
)


DEFAULT_QUESTIONS_PATH = Path("evaluation/datasets/cs_questions_v2.jsonl")
DEFAULT_GOLD_PATH = Path("evaluation/datasets/key_papers_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluation/results")
DEFAULT_K_VALUES = (5, 10, 20)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            records.append(record)
    return records


def index_by_question_id(records: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Missing question_id in {source}")
        if question_id in indexed:
            raise ValueError(f"Duplicate question_id {question_id!r} in {source}")
        indexed[question_id] = record
    return indexed


def normalize_source_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"\s+", "", normalized)
    if normalized.startswith("arxiv:"):
        normalized = re.sub(r"v\d+$", "", normalized)
    return normalized


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def paper_source_id(paper: Paper) -> str:
    return paper.source_id or arxiv_source_id(paper.url) or arxiv_source_id(paper.pdf_url)


def match_one_gold_paper(
    gold: dict[str, Any],
    candidates: list[Paper],
    fuzzy_threshold: float = 0.94,
    ambiguity_margin: float = 0.03,
) -> dict[str, Any]:
    gold_source_id = normalize_source_id(str(gold.get("source_id") or ""))
    gold_title = normalize_title(str(gold.get("title") or ""))

    if gold_source_id:
        for candidate in candidates:
            if normalize_source_id(paper_source_id(candidate)) == gold_source_id:
                return _match_record(gold, candidate, "source_id", 1.0)

    if gold_title:
        for candidate in candidates:
            if normalize_title(candidate.title) == gold_title:
                return _match_record(gold, candidate, "normalized_title", 1.0)

    similarities = sorted(
        (
            (SequenceMatcher(None, gold_title, normalize_title(candidate.title)).ratio(), candidate)
            for candidate in candidates
            if gold_title and candidate.title
        ),
        key=lambda item: (item[0], -(item[1].rank or 10**9)),
        reverse=True,
    )
    if similarities:
        best_score, best_candidate = similarities[0]
        second_score = similarities[1][0] if len(similarities) > 1 else 0.0
        if best_score >= fuzzy_threshold and best_score - second_score >= ambiguity_margin:
            return _match_record(gold, best_candidate, "fuzzy_title", best_score)

    return {
        "gold_source_id": gold.get("source_id", ""),
        "gold_title": gold.get("title", ""),
        "required": bool(gold.get("required")),
        "matched": False,
        "rank": None,
        "match_method": None,
        "match_score": None,
    }


def _match_record(
    gold: dict[str, Any],
    candidate: Paper,
    method: str,
    score: float,
) -> dict[str, Any]:
    return {
        "gold_source_id": gold.get("source_id", ""),
        "gold_title": gold.get("title", ""),
        "required": bool(gold.get("required")),
        "matched": True,
        "rank": candidate.rank,
        "match_method": method,
        "match_score": round(score, 6),
        "candidate_paper_id": candidate.paper_id,
        "candidate_source_id": paper_source_id(candidate),
        "candidate_title": candidate.title,
    }


def evaluate_ranked_papers(
    candidates: list[Paper],
    key_papers: list[dict[str, Any]],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    ranked_candidates = [
        paper.model_copy(update={"rank": paper.rank or rank})
        for rank, paper in enumerate(candidates, start=1)
    ]
    matches = [match_one_gold_paper(gold, ranked_candidates) for gold in key_papers]
    required_matches = [match for match in matches if match["required"]]
    metrics: dict[str, float] = {}

    for k in k_values:
        strict_hits = sum(
            1 for match in required_matches if match["rank"] is not None and match["rank"] <= k
        )
        broad_hits = sum(
            1 for match in matches if match["rank"] is not None and match["rank"] <= k
        )
        metrics[f"strict_recall@{k}"] = (
            strict_hits / len(required_matches) if required_matches else 0.0
        )
        metrics[f"broad_recall@{k}"] = broad_hits / len(matches) if matches else 0.0

    required_ranks = [match["rank"] for match in required_matches if match["rank"] is not None]
    metrics["mrr"] = 1.0 / min(required_ranks) if required_ranks else 0.0
    return metrics, matches


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    successful = [record for record in records if record.get("status") == "ok"]
    if not successful:
        return {}
    metric_names = successful[0]["metrics"].keys()
    aggregate = {
        metric_name: fmean(float(record["metrics"][metric_name]) for record in successful)
        for metric_name in metric_names
    }
    aggregate["mean_latency_seconds"] = fmean(
        float(record["latency_seconds"]) for record in successful
    )
    return aggregate


def select_questions_by_id(
    questions: list[dict[str, Any]],
    question_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not question_ids:
        return questions
    requested = list(dict.fromkeys(question_id.strip() for question_id in question_ids if question_id.strip()))
    available = {str(question.get("question_id")) for question in questions}
    unknown = [question_id for question_id in requested if question_id not in available]
    if unknown:
        raise ValueError(f"Unknown question ids: {', '.join(unknown)}")
    requested_set = set(requested)
    return [
        question
        for question in questions
        if str(question.get("question_id")) in requested_set
    ]


def run_paper_evaluation(
    settings: Settings,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    top_k: int = 20,
    question_ids: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    if top_k < max(DEFAULT_K_VALUES):
        raise ValueError(f"top_k must be at least {max(DEFAULT_K_VALUES)}")

    questions = load_jsonl(questions_path)
    gold_by_id = index_by_question_id(load_jsonl(gold_path), gold_path)
    questions = select_questions_by_id(questions, question_ids)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        questions = questions[:limit]

    missing_gold = [
        str(question.get("question_id"))
        for question in questions
        if str(question.get("question_id")) not in gold_by_id
    ]
    if missing_gold:
        raise ValueError(f"Missing gold papers for: {', '.join(missing_gold)}")

    evaluation_settings = settings.model_copy(
        update={
            "use_pdf": False,
            "pdf_candidate_limit": max(settings.pdf_candidate_limit, top_k),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_name = evaluation_settings.experiment_id or "arxiv-paper-eval"
    experiment_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", experiment_name).strip("-")
    experiment_id = f"{experiment_name or 'arxiv-paper-eval'}-{stamp}"
    results_path = output_dir / f"{experiment_id}-results.jsonl"
    raw_search_path = output_dir / f"{experiment_id}-raw-search.jsonl"
    summary_path = output_dir / f"{experiment_id}-summary.json"

    db = ResearchDatabase(evaluation_settings.sqlite_db_path)
    llm = build_chat_model(evaluation_settings)
    planner = plan_research(llm)
    search = search_node(evaluation_settings, db)
    records: list[dict[str, Any]] = []

    for question in questions:
        question_id = str(question["question_id"])
        run_id = f"{experiment_id}-{question_id}-{uuid.uuid4().hex[:8]}"
        question_text = str(question["question"])
        db.create_run(run_id, question_text)
        started = time.perf_counter()
        try:
            state: dict[str, Any] = {
                "run_id": run_id,
                "question": question_text,
                "year_from": int(question["year_from"]),
                "year_to": int(question["year_to"]),
                "trace": [],
            }
            state.update(planner(state))
            state.update(search(state))
            with raw_search_path.open("a", encoding="utf-8") as handle:
                for search_result in state.get("paper_search_results", []):
                    raw_record = {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "question_id": question_id,
                        "year_from": question["year_from"],
                        "year_to": question["year_to"],
                        **search_result,
                    }
                    handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            if state.get("search_failure_count"):
                raise RuntimeError(
                    f"{state['search_failure_count']} arXiv search request(s) failed"
                )
            candidates = state.get("papers", [])[:top_k]
            if len(candidates) < top_k:
                raise RuntimeError(
                    f"Only {len(candidates)} eligible papers remained after filtering; "
                    f"{top_k} are required"
                )
            out_of_range = [
                paper
                for paper in candidates
                if paper.year is None
                or paper.year < int(question["year_from"])
                or paper.year > int(question["year_to"])
            ]
            if out_of_range:
                raise RuntimeError(
                    f"Year audit failed for {len(out_of_range)} candidate paper(s)"
                )
            metrics, matches = evaluate_ranked_papers(
                candidates,
                list(gold_by_id[question_id].get("key_papers", [])),
            )
            record = {
                "status": "ok",
                "experiment_id": experiment_id,
                "run_id": run_id,
                "question_id": question_id,
                "domain": question.get("domain"),
                "topic": question.get("topic"),
                "question_type": question.get("question_type"),
                "difficulty": question.get("difficulty"),
                "year_from": question.get("year_from"),
                "year_to": question.get("year_to"),
                "question": question_text,
                "plan": state["plan"].model_dump(mode="json"),
                "search_intents": [
                    intent.model_dump(mode="json")
                    for intent in state["plan"].search_intents
                ],
                "latency_seconds": round(time.perf_counter() - started, 6),
                "metrics": metrics,
                "gold_matches": matches,
                "papers": [paper.model_dump(mode="json") for paper in candidates],
                "trace": state.get("trace", []),
            }
        except Exception as exc:
            record = {
                "status": "error",
                "experiment_id": experiment_id,
                "run_id": run_id,
                "question_id": question_id,
                "question": question_text,
                "latency_seconds": round(time.perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_path": str(questions_path),
        "gold_path": str(gold_path),
        "raw_search_path": str(raw_search_path),
        "retrieval_source": "arxiv",
        "top_k": top_k,
        "question_count": len(records),
        "successful_count": sum(record.get("status") == "ok" for record in records),
        "error_count": sum(record.get("status") == "error" for record in records),
        "configuration": {
            "model": evaluation_settings.qwen_chat_model,
            "paper_ranking_strategy": evaluation_settings.paper_ranking_strategy,
            "reranker_model": (
                evaluation_settings.qwen_rerank_model
                if evaluation_settings.paper_ranking_strategy == "qwen3_rerank"
                else None
            ),
            "search_results_per_query": evaluation_settings.search_results_per_query,
            "paper_candidate_limit": evaluation_settings.pdf_candidate_limit,
            "rrf_k": (
                RRF_K if evaluation_settings.paper_ranking_strategy == "rrf" else None
            ),
            "arxiv_sort_by": ARXIV_SORT_BY,
            "arxiv_sort_order": ARXIV_SORT_ORDER,
            "arxiv_timeout_seconds": ARXIV_TIMEOUT_SECONDS,
            "arxiv_max_retries": ARXIV_MAX_RETRIES,
            "year_filter": "arxiv_submitted_date_query_with_local_audit",
            "year_semantics": "arxiv_first_submission_year",
            "use_pdf": False,
        },
        "macro_metrics": aggregate_metrics(records),
        "errors": [
            {"question_id": record["question_id"], "error": record["error"]}
            for record in records
            if record.get("status") == "error"
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results_path, raw_search_path, summary_path
