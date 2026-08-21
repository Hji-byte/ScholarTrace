import argparse
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from langgraph.checkpoint.sqlite import SqliteSaver

from scholar_trace.agent.graph import build_graph
from scholar_trace.agent.reader import read_evidence
from scholar_trace.agent.verifier import verify_claims
from scholar_trace.agent.writer import write_report
from scholar_trace.config import get_settings
from scholar_trace.db.database import ResearchDatabase
from scholar_trace.llm import build_chat_model, build_embeddings


console = Console()

NODE_LABELS = {
    "planner": "Planner",
    "search": "Search",
    "ingest": "Ingest",
    "retrieve": "Retrieve",
    "reader": "Reader",
    "verifier": "Verifier",
    "writer": "Writer",
}

NODE_DESCRIPTIONS = {
    "planner": "Planning the research question...",
    "search": "Searching and reranking papers...",
    "ingest": "Downloading and indexing PDFs...",
    "retrieve": "Retrieving and reranking evidence...",
    "reader": "Extracting and consolidating claims...",
    "verifier": "Verifying claims against evidence...",
    "writer": "Writing the literature review...",
}


def count_label(count: int, singular: str, plural: str | None = None) -> str:
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def node_summary(node_name: str, update: dict[str, Any]) -> str:
    if node_name == "planner":
        plan = update.get("plan")
        if plan is not None:
            return ", ".join(
                [
                    count_label(len(plan.subquestions), "subquestion"),
                    count_label(len(plan.search_intents), "search intent"),
                ]
            )
    elif node_name == "search":
        failures = update.get("search_failure_count", 0)
        return ", ".join(
            [
                count_label(len(update.get("papers", [])), "paper candidate"),
                count_label(failures, "search failure"),
            ]
        )
    elif node_name == "ingest":
        return ", ".join(
            [
                count_label(len(update.get("papers", [])), "paper"),
                f"{count_label(update.get('chunks_indexed', 0), 'chunk')} indexed",
            ]
        )
    elif node_name == "retrieve":
        return f"{count_label(len(update.get('evidence_chunks', [])), 'evidence chunk')} selected"
    elif node_name == "reader":
        claims = update.get("claims", [])
        coverage = update.get("claim_coverage", {})
        covered = (
            coverage.get("all_subquestions_covered")
            if isinstance(coverage, dict)
            else None
        )
        suffix = ", all subquestions covered" if covered else ""
        return f"{count_label(len(claims), 'claim')}{suffix}"
    elif node_name == "verifier":
        accepted = update.get("supported_claim_count", len(update.get("verified_claims", [])))
        rejected = update.get("rejected_claim_count", 0)
        return f"{accepted} accepted, {rejected} rejected"
    elif node_name == "writer":
        markdown = update.get("report_markdown", "")
        references = len(re.findall(r"(?m)^\[\d+\]", markdown))
        return count_label(references, "cited paper")
    return "completed"


def progress_display() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )


def run_graph_with_progress(
    app,
    initial_state: dict[str, Any],
    config: dict[str, Any],
    node_order: list[str],
) -> dict[str, Any]:
    state = dict(initial_state)
    active_index = 0
    with progress_display() as progress:
        task_id = progress.add_task(
            f"[cyan]{NODE_LABELS[node_order[0]]}[/cyan]  "
            f"{NODE_DESCRIPTIONS[node_order[0]]}",
            total=None,
        )
        try:
            for event in app.stream(initial_state, config, stream_mode="updates"):
                for node_name, update in event.items():
                    if node_name == "__interrupt__" or not isinstance(update, dict):
                        continue
                    state.update(update)
                    label = NODE_LABELS.get(node_name, node_name)
                    progress.update(
                        task_id,
                        description=(
                            f"[green]OK {label}[/green]  {node_summary(node_name, update)}"
                        ),
                    )
                    progress.stop_task(task_id)
                    if node_name in node_order:
                        active_index = node_order.index(node_name) + 1
                    if active_index < len(node_order):
                        next_node = node_order[active_index]
                        task_id = progress.add_task(
                            f"[cyan]{NODE_LABELS[next_node]}[/cyan]  "
                            f"{NODE_DESCRIPTIONS[next_node]}",
                            total=None,
                        )
        except Exception:
            failed_node = node_order[min(active_index, len(node_order) - 1)]
            progress.update(
                task_id,
                description=f"[red]FAILED {NODE_LABELS[failed_node]}[/red]",
            )
            progress.stop_task(task_id)
            console.print(
                f"[red]Run {initial_state.get('run_id', '(unknown)')} failed in "
                f"{NODE_LABELS[failed_node]}.[/red]"
            )
            raise
    return state


def run_steps_with_progress(
    state: dict[str, Any],
    steps: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]],
) -> dict[str, Any]:
    with progress_display() as progress:
        for node_name, execute in steps:
            task_id = progress.add_task(
                f"[cyan]{NODE_LABELS[node_name]}[/cyan]  {NODE_DESCRIPTIONS[node_name]}",
                total=None,
            )
            try:
                update = execute(state)
            except Exception:
                progress.update(
                    task_id,
                    description=f"[red]FAILED {NODE_LABELS[node_name]}[/red]",
                )
                progress.stop_task(task_id)
                console.print(
                    f"[red]Run {state.get('run_id', '(unknown)')} failed in "
                    f"{NODE_LABELS[node_name]}.[/red]"
                )
                raise
            state.update(update)
            progress.update(
                task_id,
                description=(
                    f"[green]OK {NODE_LABELS[node_name]}[/green]  "
                    f"{node_summary(node_name, update)}"
                ),
            )
            progress.stop_task(task_id)
    return state


def print_run_summary(state: dict[str, Any], report_path: Path) -> None:
    console.print("\n[bold green]Research complete[/bold green]")
    console.print(
        f"Papers: {len(state.get('papers', []))}  |  "
        f"Evidence chunks: {len(state.get('evidence_chunks', []))}  |  "
        f"Claims: {len(state.get('claims', []))}  |  "
        f"Verified claims: {len(state.get('verified_claims', []))}"
    )
    console.print(f"[green]Report saved:[/green] {report_path}")


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
        graph_config = {"configurable": {"thread_id": run_id}}
        state = run_graph_with_progress(
            app,
            initial_state,
            graph_config,
            ["planner", "search", "ingest", "retrieve", "reader", "verifier", "writer"],
        )

    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))
    if state.get("claims"):
        db.save_claims(run_id, state["claims"], stage="raw")
    if state.get("verified_claims"):
        db.save_claims(run_id, state["verified_claims"], stage="verified")

    print_run_summary(state, report_path)
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
    read = read_evidence(
        llm,
        batch_size=settings.reader_batch_size,
        min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
    )
    verify = verify_claims(
        llm,
        batch_size=settings.verifier_batch_size,
        min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
    )
    write = write_report(llm)

    def read_and_save(current_state: dict[str, Any]) -> dict[str, Any]:
        update = read(current_state)
        if update.get("claims"):
            db.save_claims(run_id, update["claims"], stage="raw")
        return update

    def verify_and_save(current_state: dict[str, Any]) -> dict[str, Any]:
        update = verify(current_state)
        if update.get("verified_claims"):
            db.save_claims(run_id, update["verified_claims"], stage="verified")
        return update

    state = run_steps_with_progress(
        state,
        [
            ("reader", read_and_save),
            ("verifier", verify_and_save),
            ("writer", write),
        ],
    )

    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))
    print_run_summary(state, report_path)
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
    verify = verify_claims(
        llm,
        batch_size=settings.verifier_batch_size,
        min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
    )
    write = write_report(llm)

    def verify_and_save(current_state: dict[str, Any]) -> dict[str, Any]:
        update = verify(current_state)
        if update.get("verified_claims"):
            db.save_claims(run_id, update["verified_claims"], stage="verified")
        return update

    state = run_steps_with_progress(
        state,
        [("verifier", verify_and_save), ("writer", write)],
    )
    report_path = save_report(settings.output_dir, question, state["report_markdown"])
    db.set_report_path(run_id, str(report_path))

    print_run_summary(state, report_path)
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
