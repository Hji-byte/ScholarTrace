from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agent.llm_retry import invoke_with_retry
from research_agent.config import get_settings
from research_agent.llm import build_chat_model
from research_agent.tools.json_utils import extract_json_object


DEFAULT_RESULTS = Path("evaluation/results/dense-rrf-k15-top30-results.jsonl")
DEFAULT_CANDIDATES = Path("evaluation/results/dense-rrf-k15-top30-chunk-candidates.jsonl")
DEFAULT_ANNOTATIONS = Path("evaluation/results/dense-rrf-k15-top30-chunk-annotations.jsonl")
DEFAULT_LABELED = Path("evaluation/results/dense-rrf-k15-top30-chunk-labeled.jsonl")
ALLOWED_LABELS = {"0", "1", "2", "?"}


SYSTEM_PROMPT = """You are labeling retrieved evidence chunks for a computer-science research-agent evaluation.
Judge each chunk only against the supplied research question and the chunk text. The paper title is context, but a relevant paper does not automatically make every chunk relevant.

Use exactly one label:
- 2 (Direct evidence): The chunk itself contains a substantive method, definition, metric, benchmark description, comparison, empirical finding, limitation, or other evidence that directly helps answer at least one explicit part of the research question.
- 1 (Partial evidence): The chunk is meaningfully related and useful as background or supporting context, but does not itself provide direct evidence for an explicit part of the question.
- 0 (Irrelevant): The chunk does not substantively help answer the question. This includes keyword collisions, unrelated tasks/domains, generic boilerplate, and bibliography/reference-list fragments that merely name relevant work.
- ? (Unclear): Extraction damage or missing context makes relevance genuinely impossible to determine. Use this sparingly.

Return strict JSON only:
{"annotations":[{"item_id":"csq001-c001","label":"2","rationale":"简短中文理由"}]}

Rules:
- Return every supplied item exactly once and preserve item_id.
- Give one concise Chinese rationale grounded in the supplied chunk.
- Relevance is about evidence in this exact chunk, not paper quality or topic similarity.
- Do not use outside knowledge.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_candidates(results_path: Path) -> list[dict]:
    results = read_jsonl(results_path)
    failed = [row.get("question_id") for row in results if row.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"Failed retrieval records: {failed}")
    candidates: list[dict] = []
    for result in sorted(results, key=lambda row: row["question_id"]):
        question_id = str(result["question_id"])
        selected = [chunk for chunk in result["ranked_chunks"] if chunk["selected"]]
        for chunk in selected:
            rank = int(chunk["rank"])
            candidates.append(
                {
                    "item_id": f"{question_id}-c{rank:03d}",
                    "question_id": question_id,
                    "question": result["question"],
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "title": chunk["title"],
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "url": chunk.get("url", ""),
                    "rrf_score": chunk["rrf_score"],
                    "matched_query_count": chunk["matched_query_count"],
                    "query_hits": chunk["query_hits"],
                    "source_results": results_path.name,
                }
            )
    return candidates


def annotate_batch(llm, batch: list[dict]) -> list[dict]:
    question = batch[0]["question"]
    chunks = "\n\n".join(
        f"Item: {row['item_id']}\nPaper title: {row['title']}\nPage: {row['page']}\nChunk:\n{row['text']}"
        for row in batch
    )
    expected_ids = {row["item_id"] for row in batch}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research question:\n{question}\n\nRetrieved chunks:\n{chunks}"),
    ]
    last_error = ""
    for _ in range(3):
        response = invoke_with_retry(llm, messages, attempts=3, backoff_seconds=5)
        try:
            annotations = extract_json_object(str(response.content)).get("annotations", [])
            returned_ids = {item.get("item_id") for item in annotations}
            if len(annotations) != len(batch) or returned_ids != expected_ids:
                raise ValueError(f"Expected {sorted(expected_ids)}, got {sorted(map(str, returned_ids))}")
            for item in annotations:
                item["label"] = str(item.get("label", ""))
                if item["label"] not in ALLOWED_LABELS:
                    raise ValueError(f"Invalid label: {item}")
                if not str(item.get("rationale", "")).strip():
                    raise ValueError(f"Missing rationale: {item}")
            return annotations
        except Exception as exc:
            last_error = str(exc)
            messages.append(
                HumanMessage(
                    content=f"Validation failed: {last_error}. Return corrected strict JSON with every item once."
                )
            )
    raise RuntimeError(f"Annotation batch failed: {last_error}")


def run_annotations(candidates: list[dict], annotations_path: Path, batch_size: int) -> None:
    existing_rows = read_jsonl(annotations_path) if annotations_path.exists() else []
    existing = {row["item_id"]: row for row in existing_rows}
    pending: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        if candidate["item_id"] not in existing:
            pending[candidate["question_id"]].append(candidate)
    if not pending:
        print(f"No pending items; {len(existing)} annotations already exist.")
        return
    llm = build_chat_model(get_settings())
    completed = len(existing)
    with annotations_path.open("a", encoding="utf-8") as handle:
        for question_id in sorted(pending):
            rows = pending[question_id]
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                annotations = annotate_batch(llm, batch)
                source_by_id = {row["item_id"]: row for row in batch}
                for annotation in annotations:
                    source = source_by_id[annotation["item_id"]]
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
                print(f"Annotated {completed}/{len(candidates)} ({question_id} {batch[0]['rank']}-{batch[-1]['rank']})", flush=True)


def dcg(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels[:k], 1))


def build_outputs(candidates: list[dict], annotations_path: Path, labeled_path: Path) -> dict:
    annotation_rows = read_jsonl(annotations_path) if annotations_path.exists() else []
    annotations = {row["item_id"]: row for row in annotation_rows}
    candidate_ids = {row["item_id"] for row in candidates}
    missing = sorted(candidate_ids - set(annotations))
    extra = sorted(set(annotations) - candidate_ids)
    invalid = sorted(item_id for item_id, row in annotations.items() if row.get("label") not in ALLOWED_LABELS)
    if missing or extra or invalid:
        raise RuntimeError(f"Annotation audit failed: missing={missing}, extra={extra}, invalid={invalid}")

    labeled = [
        {**candidate, "auto_label": annotations[candidate["item_id"]]["label"],
         "auto_rationale": annotations[candidate["item_id"]]["rationale"]}
        for candidate in candidates
    ]
    write_jsonl(labeled_path, labeled)
    by_question: dict[str, list[dict]] = defaultdict(list)
    for row in labeled:
        by_question[row["question_id"]].append(row)
    per_question = []
    for question_id, rows in sorted(by_question.items()):
        rows.sort(key=lambda row: row["rank"])
        labels = [0 if row["auto_label"] == "?" else int(row["auto_label"]) for row in rows]
        binary = [int(label > 0) for label in labels]
        ideal = sorted(labels, reverse=True)
        metrics = {
            f"precision_at_{k}": sum(binary[:k]) / min(k, len(binary))
            for k in (5, 10, 20, 30)
        }
        metrics.update({
            f"ndcg_at_{k}": dcg(labels, k) / dcg(ideal, k) if dcg(ideal, k) else 0.0
            for k in (5, 10, 20, 30)
        })
        per_question.append({
            "question_id": question_id,
            "candidate_count": len(rows),
            "label_counts": dict(Counter(row["auto_label"] for row in rows)),
            **metrics,
        })
    counts = Counter(row["auto_label"] for row in labeled)
    metric_names = [key for key in per_question[0] if key.startswith(("precision_", "ndcg_"))]
    summary = {
        "source_results": candidates[0]["source_results"] if candidates else "",
        "candidate_count": len(candidates),
        "question_count": len(by_question),
        "label_counts": dict(sorted(counts.items())),
        "aggregate_metrics_macro": {
            name: fmean(float(row[name]) for row in per_question) for name in metric_names
        },
        "per_question": per_question,
    }
    summary_path = labeled_path.with_name(labeled_path.stem + "-summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    candidates = collect_candidates(args.results)
    write_jsonl(args.candidates, candidates)
    print(f"Collected {len(candidates)} chunks across {len({r['question_id'] for r in candidates})} questions.")
    if args.collect_only:
        return
    if not args.audit_only:
        run_annotations(candidates, args.annotations, max(1, args.batch_size))
    summary = build_outputs(candidates, args.annotations, args.labeled)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
