import argparse
import re
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from langgraph.checkpoint.sqlite import SqliteSaver

from scholar_trace.agent.graph import build_graph
from scholar_trace.agent.reader import read_evidence
from scholar_trace.agent.verifier import verify_claims
from scholar_trace.agent.writer import write_report
from scholar_trace.config import get_settings
from scholar_trace.db.database import ResearchDatabase
from scholar_trace.llm import build_chat_model, build_embeddings


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


def run(
    question: str,
    year_from: int | None = None,
    year_to: int | None = None,
    source_mode: str = "arxiv",
    library_path: Path | None = None,
) -> Path:
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError("year_from must be less than or equal to year_to")
    if source_mode not in {"arxiv", "library", "hybrid"}:
        raise ValueError(f"Unsupported source mode: {source_mode}")
    if source_mode in {"library", "hybrid"} and library_path is None:
        raise ValueError(f"library_path is required for source mode '{source_mode}'")
    if source_mode == "arxiv" and library_path is not None:
        raise ValueError("library_path requires library or hybrid source mode")
    if library_path is not None and not library_path.expanduser().exists():
        raise FileNotFoundError(f"Library path does not exist: {library_path}")
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
    db.create_run(run_id, question, year_from=year_from, year_to=year_to)
    console.print(f"[bold]Run ID:[/bold] {run_id}")

    llm = build_chat_model(settings)
    embeddings = build_embeddings(settings)

    console.print(Panel.fit(question, title="Research Question"))
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        app = build_graph(settings, llm, embeddings, db, checkpointer=checkpointer)
        initial_state = {
            "run_id": run_id,
            "question": question,
            "source_mode": source_mode,
            "trace": [],
        }
        if library_path is not None:
            initial_state["library_path"] = str(library_path.expanduser().resolve())
        if year_from is not None:
            initial_state["year_from"] = year_from
        if year_to is not None:
            initial_state["year_to"] = year_to
        state = app.invoke(
            initial_state,
            {"configurable": {"thread_id": run_id}},
        )

    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))
    if state.get("claims"):
        db.save_claims(run_id, state["claims"], stage="raw")
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
    plan = db.load_plan(run_id)
    papers = db.load_papers(run_id)
    if settings.max_evidence_chunks:
        chunks = chunks[: settings.max_evidence_chunks]
    if not chunks:
        raise ValueError(f"No evidence chunks found for run: {run_id}")

    llm = build_chat_model(settings)
    state = {
        "run_id": run_id,
        "question": question,
        "plan": plan,
        "papers": papers,
        "evidence_chunks": chunks,
        "trace": [f"Resumed from {len(chunks)} saved evidence chunks."],
    }

    console.print(Panel.fit(question, title=f"Resume Run {run_id}"))
    state.update(
        read_evidence(
            llm,
            batch_size=settings.reader_batch_size,
            min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
        )(state)
    )
    if state.get("claims"):
        db.save_claims(run_id, state["claims"], stage="raw")

    state.update(
        verify_claims(
            llm,
            batch_size=settings.verifier_batch_size,
            min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
        )(state)
    )
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
    plan = db.load_plan(run_id)
    papers = db.load_papers(run_id)
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
        "plan": plan,
        "papers": papers,
        "evidence_chunks": chunks,
        "claims": claims,
        "trace": [f"Resumed from {len(claims)} saved raw claims and {len(chunks)} evidence chunks."],
    }

    console.print(Panel.fit(question, title=f"Verify Run {run_id}"))
    state.update(
        verify_claims(
            llm,
            batch_size=settings.verifier_batch_size,
            min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
        )(state)
    )
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
    parser = argparse.ArgumentParser(
        description="ScholarTrace: an evidence-grounded agent for CS literature reviews."
    )
    parser.add_argument("question", nargs="?", help="Research question to investigate.")
    parser.add_argument(
        "--year-from",
        type=int,
        help="Earliest arXiv upload year to include.",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        help="Latest arXiv upload year to include.",
    )
    parser.add_argument(
        "--source",
        choices=["arxiv", "library", "hybrid"],
        default="arxiv",
        help="Paper source: arXiv only, a local PDF library, or both.",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        help="A PDF file or directory used by the library and hybrid modes.",
    )
    parser.add_argument("--from-run", help="Resume reader/verifier/writer from saved evidence chunks.")
    parser.add_argument("--verify-from-run", help="Resume verifier/writer from saved raw claims.")
    args = parser.parse_args()
    if args.year_from is not None and args.year_to is not None and args.year_from > args.year_to:
        parser.error("--year-from must be less than or equal to --year-to")
    if (args.year_from is not None or args.year_to is not None) and not args.question:
        parser.error("--year-from/--year-to can only be used with a research question")
    if args.source in {"library", "hybrid"} and args.library_path is None:
        parser.error(f"--library-path is required when --source={args.source}")
    if args.source == "arxiv" and args.library_path is not None:
        parser.error("--library-path requires --source=library or --source=hybrid")
    if args.source == "library" and (args.year_from is not None or args.year_to is not None):
        parser.error("--year-from/--year-to only apply to arXiv and hybrid modes")
    if args.library_path is not None and not args.library_path.expanduser().exists():
        parser.error(f"library path does not exist: {args.library_path}")
    if (args.source != "arxiv" or args.library_path is not None) and not args.question:
        parser.error("--source/--library-path can only be used with a research question")
    selected_modes = sum(
        bool(mode)
        for mode in [
            args.from_run,
            args.verify_from_run,
            args.question,
        ]
    )
    if selected_modes > 1:
        parser.error(
            "use only one question, --from-run, or --verify-from-run"
        )
    if args.from_run:
        resume_from_chunks(args.from_run)
    elif args.verify_from_run:
        resume_from_claims(args.verify_from_run)
    elif args.question:
        run(
            args.question,
            year_from=args.year_from,
            year_to=args.year_to,
            source_mode=args.source,
            library_path=args.library_path,
        )
    else:
        parser.error("question is required unless --from-run is provided")


if __name__ == "__main__":
    main()
