from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from research_agent.agent.graph import ingest_node
from research_agent.config import get_settings
from research_agent.llm import build_embeddings
from research_agent.paper_evaluation import load_jsonl
from research_agent.schema import Paper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--question-id", default="csq001")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument(
        "--experiment-id",
        help="Reuse an existing experiment ID and Chroma directory to resume safely.",
    )
    args = parser.parse_args()

    matching = [
        row
        for row in load_jsonl(args.results)
        if row.get("status") == "ok" and row.get("question_id") == args.question_id
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one successful result for {args.question_id}, found {len(matching)}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_id = args.experiment_id or f"pdf-ingest-{args.question_id}-{stamp}"
    settings = get_settings().model_copy(
        update={
            "pdf_candidate_limit": 20,
            "max_papers": 10,
            "use_pdf": True,
            "pdf_failure_policy": "skip",
            "min_pdf_papers": 6,
            "chroma_persist_dir": Path("data/chroma") / experiment_id,
        }
    )
    papers = [Paper.model_validate(row) for row in matching[0]["papers"][:20]]
    embeddings = build_embeddings(settings)

    started = time.perf_counter()
    output = ingest_node(settings, embeddings)(
        {
            "run_id": experiment_id,
            "question": matching[0]["question"],
            "papers": papers,
            "trace": [],
        }
    )
    elapsed = time.perf_counter() - started

    ingested_details = []
    for paper in output["papers"]:
        pdf_path = Path("data/pdfs") / f"{paper.paper_id}.pdf"
        page_count = len(PdfReader(str(pdf_path)).pages) if pdf_path.exists() else None
        ingested_details.append(
            {
                "rank": paper.rank,
                "paper_id": paper.paper_id,
                "source_id": paper.source_id,
                "title": paper.title,
                "pdf_path": str(pdf_path),
                "pdf_bytes": pdf_path.stat().st_size if pdf_path.exists() else None,
                "page_count": page_count,
            }
        )

    summary = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(args.results),
        "question_id": args.question_id,
        "question": matching[0]["question"],
        "configuration": {
            "pdf_candidate_limit": settings.pdf_candidate_limit,
            "max_papers": settings.max_papers,
            "use_pdf": settings.use_pdf,
            "pdf_failure_policy": settings.pdf_failure_policy,
            "min_pdf_papers": settings.min_pdf_papers,
            "embedding_model": settings.qwen_embedding_model,
        },
        "candidate_count": len(papers),
        "ingested_paper_count": len(output["papers"]),
        "minimum_pdf_target_met": len(output["papers"]) >= settings.min_pdf_papers,
        "chunks_indexed": output["chunks_indexed"],
        "elapsed_seconds": round(elapsed, 6),
        "chroma_persist_dir": str(settings.chroma_persist_dir),
        "trace": output["trace"],
        "ingested_papers": ingested_details,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{experiment_id}-summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
