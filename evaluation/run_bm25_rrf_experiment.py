from __future__ import annotations

import argparse
from pathlib import Path

from research_agent.bm25_rrf_evaluation import DEFAULT_RESULTS_PATH, run_bm25_rrf_evaluation
from research_agent.dense_rrf_evaluation import DEFAULT_PLANS_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS_PATH)
    parser.add_argument("--summary-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--k-per-query", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = run_bm25_rrf_evaluation(
        plans_path=args.plans,
        summary_dir=args.summary_dir,
        output_path=args.output,
        question_ids=args.question_ids,
        k_per_query=args.k_per_query,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        force=args.force,
    )
    print(output)
    print(summary)


if __name__ == "__main__":
    main()
