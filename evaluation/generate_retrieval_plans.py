from __future__ import annotations

import argparse
from pathlib import Path

from research_agent.config import get_settings
from research_agent.paper_evaluation import DEFAULT_QUESTIONS_PATH
from research_agent.retrieval_plans import (
    DEFAULT_RETRIEVAL_PLANS_PATH,
    generate_retrieval_plans,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RETRIEVAL_PLANS_PATH)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate selected plans even when successful records already exist.",
    )
    args = parser.parse_args()

    output_path = generate_retrieval_plans(
        get_settings(),
        questions_path=args.questions,
        output_path=args.output,
        question_ids=args.question_ids,
        force=args.force,
    )
    print(output_path)


if __name__ == "__main__":
    main()
