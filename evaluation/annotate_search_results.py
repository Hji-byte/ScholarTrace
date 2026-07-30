from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agent.llm_retry import invoke_with_retry
from research_agent.config import get_settings
from research_agent.llm import build_chat_model
from research_agent.tools.json_utils import extract_json_object


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
QUESTIONS_PATH = ROOT / "evaluation" / "datasets" / "cs_questions_v2.jsonl"
MAIN_RESULTS = RESULTS_DIR / "arxiv-paper-eval-20260726-140242-results.jsonl"
MAIN_RAW = RESULTS_DIR / "arxiv-paper-eval-20260726-140242-raw-search.jsonl"
RECOVERY_RESULTS = {
    "csq028": RESULTS_DIR / "arxiv-paper-eval-20260727-095247-results.jsonl",
    "csq029": RESULTS_DIR / "arxiv-paper-eval-20260727-095426-results.jsonl",
    "csq030": RESULTS_DIR / "arxiv-paper-eval-20260727-095606-results.jsonl",
}
RRF_K = 60
ALLOWED_LABELS = {"0", "1", "2", "?"}


SYSTEM_PROMPT = """You are labeling search results for a computer-science literature-search evaluation.
Judge each paper only from the supplied research question, title, and abstract. Do not use outside knowledge.

Use exactly one label:
- 2 (Directly relevant): The paper's main contribution directly addresses at least one explicit, substantive component of the research question and could serve as core evidence in the final literature review. For a broad survey or comparison question, a paper does not need to answer the entire question; directly covering one named method, dimension, benchmark, threat, or comparison axis is enough.
- 1 (Partially relevant): The paper is meaningfully related and may provide background, an application example, a supporting technique, or adjacent evidence, but its main contribution does not directly answer the question's explicit focus.
- 0 (Irrelevant): Keyword collision, wrong task/domain, or no substantive help in answering the question.
- ? (Unclear): The title and abstract do not contain enough information to decide. Use this sparingly; uncertainty about quality or importance is not a reason to use ?.

Return strict JSON only in this shape:
{"annotations":[{"item_id":"csq001-r01","label":"2","rationale":"short Chinese reason"}]}

Rules:
- Return every supplied item exactly once and preserve item_id exactly.
- Rationale must be a concise Chinese sentence grounded in the title/abstract.
- Relevance is to the research question, not general paper quality.
- Do not reward a paper merely for sharing broad terms such as LLM, evaluation, security, scheduling, or graph.
"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_title(title: str) -> str:
    return " ".join(title.lower().split())


def reconstruct_csq017() -> list[dict]:
    rankings: dict[str, dict] = {}
    for result in read_jsonl(MAIN_RAW):
        if result.get("question_id") != "csq017" or result.get("status") != "ok":
            continue
        query_index = int(result["query_index"])
        for source_rank, raw_paper in enumerate(result.get("papers", []), start=1):
            paper = dict(raw_paper)
            paper.pop("source_rank", None)
            key = normalized_title(paper.get("title", ""))
            if not key:
                continue
            entry = rankings.setdefault(
                key,
                {
                    "paper": paper,
                    "rrf_score": 0.0,
                    "query_indexes": set(),
                    "best_rank": math.inf,
                },
            )
            entry["rrf_score"] += 1 / (RRF_K + source_rank)
            entry["query_indexes"].add(query_index)
            entry["best_rank"] = min(entry["best_rank"], source_rank)

    ranked = sorted(
        rankings.values(),
        key=lambda item: (
            item["rrf_score"],
            len(item["query_indexes"]),
            -item["best_rank"],
            item["paper"].get("year") or 0,
            1 if item["paper"].get("abstract") else 0,
        ),
        reverse=True,
    )
    papers = []
    for rank, item in enumerate(ranked, start=1):
        paper = item["paper"]
        paper["rank"] = rank
        paper["rrf_score"] = item["rrf_score"]
        papers.append(paper)
    return papers


def load_candidates() -> list[dict]:
    questions = {row["question_id"]: row for row in read_jsonl(QUESTIONS_PATH)}
    result_by_question = {
        row["question_id"]: row
        for row in read_jsonl(MAIN_RESULTS)
        if row.get("status") == "ok"
    }
    source_run = {
        question_id: MAIN_RESULTS.name for question_id in result_by_question
    }
    for question_id, path in RECOVERY_RESULTS.items():
        recovered = next(
            row for row in read_jsonl(path)
            if row.get("question_id") == question_id and row.get("status") == "ok"
        )
        result_by_question[question_id] = recovered
        source_run[question_id] = path.name

    rows: list[dict] = []
    for question_id in sorted(questions):
        question = questions[question_id]
        if question_id == "csq017":
            papers = reconstruct_csq017()
            run_name = f"{MAIN_RAW.name} (RRF reconstructed)"
        else:
            papers = result_by_question[question_id]["papers"]
            run_name = source_run[question_id]
        for paper in papers:
            rank = int(paper["rank"])
            rows.append(
                {
                    "item_id": f"{question_id}-r{rank:02d}",
                    **question,
                    "rank": rank,
                    "paper_id": paper.get("paper_id", ""),
                    "source_id": paper.get("source_id", ""),
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract", ""),
                    "url": paper.get("url", ""),
                    "pdf_url": paper.get("pdf_url", ""),
                    "rrf_score": paper.get("rrf_score"),
                    "source_run": run_name,
                }
            )
    return rows


def load_candidates_from_results(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    results = read_jsonl(results_path)
    failed = [result.get("question_id") for result in results if result.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"Cannot collect candidates from failed questions: {failed}")
    for result in sorted(results, key=lambda item: item["question_id"]):
        question_id = result["question_id"]
        for paper in result.get("papers", []):
            rank = int(paper["rank"])
            rows.append(
                {
                    "item_id": f"{question_id}-r{rank:02d}",
                    "question_id": question_id,
                    "domain": result.get("domain"),
                    "topic": result.get("topic"),
                    "question_type": result.get("question_type"),
                    "difficulty": result.get("difficulty"),
                    "year_from": result.get("year_from"),
                    "year_to": result.get("year_to"),
                    "question": result["question"],
                    "rank": rank,
                    "paper_id": paper.get("paper_id", ""),
                    "source_id": paper.get("source_id", ""),
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract", ""),
                    "url": paper.get("url", ""),
                    "pdf_url": paper.get("pdf_url", ""),
                    "rrf_score": paper.get("rrf_score"),
                    "reranker_score": paper.get("reranker_score"),
                    "source_run": results_path.name,
                }
            )
    return rows


def load_existing_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["item_id"]: row for row in read_jsonl(path)}


def annotate_batch(llm, batch: list[dict]) -> list[dict]:
    question = batch[0]["question"]
    papers = "\n\n".join(
        f"Item: {row['item_id']}\nTitle: {row['title']}\nAbstract: {row['abstract']}"
        for row in batch
    )
    expected_ids = {row["item_id"] for row in batch}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research question:\n{question}\n\nPapers:\n{papers}"),
    ]
    last_error = ""
    for attempt in range(1, 4):
        response = invoke_with_retry(llm, messages, attempts=3, backoff_seconds=5)
        try:
            data = extract_json_object(str(response.content))
            annotations = data.get("annotations", [])
            returned_ids = {item.get("item_id") for item in annotations}
            if returned_ids != expected_ids or len(annotations) != len(batch):
                raise ValueError(
                    f"expected ids {sorted(expected_ids)}, got {sorted(str(x) for x in returned_ids)}"
                )
            for item in annotations:
                item["label"] = str(item.get("label", ""))
                if item["label"] not in ALLOWED_LABELS:
                    raise ValueError(f"invalid label for {item['item_id']}: {item['label']}")
                if not str(item.get("rationale", "")).strip():
                    raise ValueError(f"missing rationale for {item['item_id']}")
            return annotations
        except Exception as exc:
            last_error = str(exc)
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response failed validation: "
                        f"{last_error}. Return corrected strict JSON containing every item exactly once."
                    )
                )
            )
    raise RuntimeError(f"Annotation batch failed validation: {last_error}")


def write_candidates(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_annotation(rows: list[dict], output_path: Path, batch_size: int) -> None:
    existing = load_existing_annotations(output_path)
    pending_by_question: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["item_id"] not in existing:
            pending_by_question[row["question_id"]].append(row)
    if not pending_by_question:
        print(f"No pending items; {len(existing)} annotations already exist.")
        return

    llm = build_chat_model(get_settings())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(existing)
    total = len(rows)
    with output_path.open("a", encoding="utf-8") as handle:
        for question_id in sorted(pending_by_question):
            question_rows = pending_by_question[question_id]
            for start in range(0, len(question_rows), batch_size):
                batch = question_rows[start : start + batch_size]
                annotations = annotate_batch(llm, batch)
                rows_by_id = {row["item_id"]: row for row in batch}
                for annotation in annotations:
                    source = rows_by_id[annotation["item_id"]]
                    record = {
                        "item_id": annotation["item_id"],
                        "question_id": source["question_id"],
                        "rank": source["rank"],
                        "label": annotation["label"],
                        "rationale": str(annotation["rationale"]).strip(),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += len(batch)
                print(f"Annotated {completed}/{total} ({question_id}, ranks {batch[0]['rank']}-{batch[-1]['rank']})", flush=True)


def audit(rows: list[dict], annotations_path: Path) -> None:
    annotations = load_existing_annotations(annotations_path)
    candidate_ids = {row["item_id"] for row in rows}
    annotation_ids = set(annotations)
    missing = sorted(candidate_ids - annotation_ids)
    extra = sorted(annotation_ids - candidate_ids)
    invalid = sorted(
        item_id for item_id, item in annotations.items()
        if item.get("label") not in ALLOWED_LABELS
    )
    counts = defaultdict(int)
    per_question = defaultdict(int)
    for item_id in candidate_ids & annotation_ids:
        counts[annotations[item_id]["label"]] += 1
        per_question[annotations[item_id]["question_id"]] += 1
    print(json.dumps({
        "candidates": len(rows),
        "annotations": len(annotations),
        "label_counts": dict(sorted(counts.items())),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "per_question_counts": dict(sorted(per_question.items())),
    }, ensure_ascii=False, indent=2))
    if missing or extra or invalid:
        raise RuntimeError("Annotation audit failed")


def merge_annotation_parts(part_paths: list[Path], output_path: Path) -> None:
    merged: list[dict] = []
    for part_path in part_paths:
        merged.extend(read_jsonl(part_path))
    merged.sort(key=lambda row: (row["question_id"], int(row["rank"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {len(merged)} annotations from {len(part_paths)} parts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--results-input", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--merge-parts", nargs="*", type=Path)
    args = parser.parse_args()

    rows = (
        load_candidates_from_results(args.results_input)
        if args.results_input
        else load_candidates()
    )
    write_candidates(rows, args.candidates)
    print(f"Collected {len(rows)} candidates across {len({row['question_id'] for row in rows})} questions.")
    if args.collect_only:
        return
    if args.merge_parts:
        merge_annotation_parts(args.merge_parts, args.annotations)
        audit(rows, args.annotations)
        return
    if not args.audit_only:
        run_annotation(rows, args.annotations, max(1, args.batch_size))
    audit(rows, args.annotations)


if __name__ == "__main__":
    main()
