import argparse
import re
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.agent.graph import build_graph
from research_agent.agent.reader import read_evidence
from research_agent.agent.verifier import verify_claims
from research_agent.agent.writer import write_report
from research_agent.config import get_settings
from research_agent.db.database import ResearchDatabase
from research_agent.llm import build_chat_model, build_embeddings
from research_agent.paper_evaluation import (
    DEFAULT_GOLD_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTIONS_PATH,
    run_paper_evaluation,
)
from research_agent.saved_search_rerank import run_saved_search_rerank


console = Console()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60] or "report"


def save_report(output_dir: Path, question: str, markdown: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{stamp}-{slugify(question)}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def chroma_dir_for_run(base_dir: Path, run_id: str, experiment_id: str = "") -> Path:
    experiment_slug = slugify(experiment_id) if experiment_id else run_id
    return base_dir / experiment_slug


def run(question: str) -> Path:
    settings = get_settings()
    run_id = str(uuid.uuid4())
    settings = settings.model_copy(
        update={
            "chroma_persist_dir": chroma_dir_for_run(
                settings.chroma_persist_dir,
                run_id,
                settings.experiment_id,
            )
        }
    )
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run(run_id, question)

    llm = build_chat_model(settings)
    embeddings = build_embeddings(settings)

    console.print(Panel.fit(question, title="Research Question"))
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        app = build_graph(settings, llm, embeddings, db, checkpointer=checkpointer)
        state = app.invoke(
            {"run_id": run_id, "question": question, "trace": []},
            {"configurable": {"thread_id": run_id}},
        )

    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))
    if state.get("verified_claims"):
        db.save_claims(run_id, state["verified_claims"], stage="verified")

    console.print("\n[bold]Trace[/bold]")
    for item in state.get("trace", []):
        console.print(f"- {item}")
    console.print(f"\n[green]Report saved:[/green] {report_path}")
    return report_path


def resume_from_chunks(run_id: str) -> Path:
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    db = ResearchDatabase(settings.sqlite_db_path)
    question = db.get_run_question(run_id)
    chunks = db.load_chunks(run_id)
    if settings.max_evidence_chunks:
        chunks = chunks[: settings.max_evidence_chunks]
    if not chunks:
        raise ValueError(f"No evidence chunks found for run: {run_id}")

    llm = build_chat_model(settings)
    state = {
        "run_id": run_id,
        "question": question,
        "evidence_chunks": chunks,
        "trace": [f"Resumed from {len(chunks)} saved evidence chunks."],
    }

    console.print(Panel.fit(question, title=f"Resume Run {run_id}"))
    state.update(read_evidence(llm, batch_size=settings.reader_batch_size)(state))
    if state.get("claims"):
        db.save_claims(run_id, state["claims"], stage="raw")

    state.update(verify_claims(llm, batch_size=settings.verifier_batch_size)(state))
    if state.get("verified_claims"):
        db.save_claims(run_id, state["verified_claims"], stage="verified")

    state.update(write_report(llm)(state))

    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))
    console.print("\n[bold]Trace[/bold]")
    for item in state.get("trace", []):
        console.print(f"- {item}")
    console.print(f"\n[green]Report saved:[/green] {report_path}")
    return report_path


def resume_from_claims(run_id: str) -> Path:
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    db = ResearchDatabase(settings.sqlite_db_path)
    question = db.get_run_question(run_id)
    chunks = db.load_chunks(run_id)
    claims = db.load_claims(run_id, stage="raw")
    if settings.max_evidence_chunks:
        chunks = chunks[: settings.max_evidence_chunks]
    if not chunks:
        raise ValueError(f"No evidence chunks found for run: {run_id}")
    if not claims:
        raise ValueError(f"No raw claims found for run: {run_id}")

    llm = build_chat_model(settings)
    state = {
        "run_id": run_id,
        "question": question,
        "evidence_chunks": chunks,
        "claims": claims,
        "trace": [f"Resumed from {len(claims)} saved raw claims and {len(chunks)} evidence chunks."],
    }

    console.print(Panel.fit(question, title=f"Verify Run {run_id}"))
    state.update(verify_claims(llm, batch_size=settings.verifier_batch_size)(state))
    if state.get("verified_claims"):
        db.save_claims(run_id, state["verified_claims"], stage="verified")

    state.update(write_report(llm)(state))
    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))

    console.print("\n[bold]Trace[/bold]")
    for item in state.get("trace", []):
        console.print(f"- {item}")
    console.print(f"\n[green]Report saved:[/green] {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG assistant for CS literature reviews.")
    parser.add_argument("question", nargs="?", help="Research question to investigate.")
    parser.add_argument("--from-run", help="Resume reader/verifier/writer from saved evidence chunks.")
    parser.add_argument("--verify-from-run", help="Resume verifier/writer from saved raw claims.")
    parser.add_argument(
        "--paper-eval",
        action="store_true",
        help="Run the arXiv-only paper retrieval evaluation.",
    )
    parser.add_argument("--eval-questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--eval-gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-limit", type=int, help="Run only the first N questions.")
    parser.add_argument(
        "--eval-question-ids",
        nargs="+",
        help="Run only the listed benchmark question IDs, preserving dataset order.",
    )
    parser.add_argument("--eval-top-k", type=int, default=20)
    parser.add_argument(
        "--rerank-saved-search",
        type=Path,
        help="Rerank a saved raw-search JSONL without rerunning planner or arXiv.",
    )
    args = parser.parse_args()
    selected_modes = sum(
        bool(mode)
        for mode in [
            args.from_run,
            args.verify_from_run,
            args.paper_eval,
            args.rerank_saved_search,
            args.question,
        ]
    )
    if selected_modes > 1:
        parser.error(
            "use only one question, --from-run, --verify-from-run, --paper-eval, "
            "or --rerank-saved-search"
        )
    if args.paper_eval:
        results_path, raw_search_path, summary_path = run_paper_evaluation(
            get_settings(),
            questions_path=args.eval_questions,
            gold_path=args.eval_gold,
            output_dir=args.eval_output,
            limit=args.eval_limit,
            top_k=args.eval_top_k,
            question_ids=args.eval_question_ids,
        )
        console.print(f"[green]Per-question results:[/green] {results_path}")
        console.print(f"[green]Raw search results:[/green] {raw_search_path}")
        console.print(f"[green]Summary:[/green] {summary_path}")
    elif args.rerank_saved_search:
        results_path, summary_path = run_saved_search_rerank(
            get_settings(),
            raw_search_path=args.rerank_saved_search,
            questions_path=args.eval_questions,
            gold_path=args.eval_gold,
            output_dir=args.eval_output,
            top_k=args.eval_top_k,
        )
        console.print(f"[green]Reranked results:[/green] {results_path}")
        console.print(f"[green]Summary:[/green] {summary_path}")
    elif args.from_run:
        resume_from_chunks(args.from_run)
    elif args.verify_from_run:
        resume_from_claims(args.verify_from_run)
    elif args.question:
        run(args.question)
    else:
        parser.error("question is required unless --from-run is provided")


if __name__ == "__main__":
    main()
