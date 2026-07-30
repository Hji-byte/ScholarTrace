from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from research_agent.agent.planner import plan_research
from research_agent.config import Settings
from research_agent.llm import build_chat_model
from research_agent.paper_evaluation import load_jsonl, select_questions_by_id


DEFAULT_RETRIEVAL_PLANS_PATH = Path("evaluation/datasets/retrieval_plans.jsonl")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def generate_retrieval_plans(
    settings: Settings,
    questions_path: Path,
    output_path: Path = DEFAULT_RETRIEVAL_PLANS_PATH,
    question_ids: list[str] | None = None,
    force: bool = False,
    llm: BaseChatModel | None = None,
) -> Path:
    all_questions = load_jsonl(questions_path)
    questions = all_questions
    if question_ids:
        questions = select_questions_by_id(all_questions, question_ids)

    existing_records = load_jsonl(output_path) if output_path.exists() else []
    records_by_id = {
        str(record["question_id"]): record
        for record in existing_records
        if record.get("question_id")
    }
    question_order = [str(question["question_id"]) for question in all_questions]
    planner = plan_research(llm or build_chat_model(settings))

    for question in questions:
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        existing = records_by_id.get(question_id)
        if (
            not force
            and existing
            and existing.get("status") == "ok"
            and existing.get("question") == question_text
        ):
            continue

        try:
            output = planner(
                {
                    "run_id": f"retrieval-plan-{question_id}",
                    "question": question_text,
                    "trace": [],
                }
            )
            plan = output["plan"]
            record = {
                "status": "ok",
                "question_id": question_id,
                "domain": question.get("domain"),
                "topic": question.get("topic"),
                "question_type": question.get("question_type"),
                "difficulty": question.get("difficulty"),
                "question": question_text,
                "plan": plan.model_dump(mode="json"),
                "planner_model": settings.qwen_chat_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trace": output.get("trace", []),
            }
        except Exception as exc:
            record = {
                "status": "error",
                "question_id": question_id,
                "question": question_text,
                "planner_model": settings.qwen_chat_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }

        records_by_id[question_id] = record
        ordered_records = [
            records_by_id[item_id]
            for item_id in question_order
            if item_id in records_by_id
        ]
        _write_jsonl_atomic(output_path, ordered_records)

    return output_path
