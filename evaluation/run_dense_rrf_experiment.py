from __future__ import annotations

import argparse
from pathlib import Path

from research_agent.config import get_settings
from research_agent.dense_rrf_evaluation import (
    DEFAULT_PLANS_PATH,
    DEFAULT_QUERY_CACHE_PATH,
    DEFAULT_RESULTS_PATH,
    run_dense_rrf_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS_PATH)
    parser.add_argument("--summary-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE_PATH)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--k-per-query", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = run_dense_rrf_evaluation(
        get_settings(),
        plans_path=args.plans,
        summary_dir=args.summary_dir,
        output_path=args.output,
        query_cache_path=args.query_cache,
        question_ids=args.question_ids,
        k_per_query=args.k_per_query,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        force=args.force,
    )
    print(output)
    print(summary)


if __name__ == "__main__":
    main()
