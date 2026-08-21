from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from scholar_trace.agent.planner import plan_research
from scholar_trace.agent.reader import read_evidence
from scholar_trace.agent.search import search_node
from scholar_trace.agent.verifier import verify_claims
from scholar_trace.agent.writer import write_report
from scholar_trace.config import Settings
from scholar_trace.db.database import ResearchDatabase
from scholar_trace.retrieval.chunking import chunk_documents
from scholar_trace.retrieval.hybrid import retrieve_hybrid_evidence
from scholar_trace.retrieval.vector_store import ChromaEvidenceStore
from scholar_trace.schema import ResearchState
from scholar_trace.tools.pdf_loader import load_paper_documents


def ingest_node(settings: Settings, embeddings: Embeddings):
    def node(state: ResearchState) -> ResearchState:
        documents = []
        ingested_papers = []
        skipped_papers = 0
        for paper in state.get("papers", []):
            if len(ingested_papers) >= settings.max_papers:
                break
            is_local = paper.source == "local" or bool(paper.local_pdf_path)
            paper_documents = load_paper_documents(
                paper,
                pdf_dir=Path("data/pdfs"),
                use_pdf=settings.use_pdf or is_local,
                pdf_failure_policy=settings.pdf_failure_policy,
            )
            if not paper_documents:
                skipped_papers += 1
                continue
            documents.extend(paper_documents)
            ingested_papers.append(paper)
        source_mode = state.get("source_mode", "arxiv")
        if source_mode == "library" and not ingested_papers:
            raise RuntimeError("No usable PDFs were found in the local library.")
        if (
            source_mode != "library"
            and settings.use_pdf
            and settings.pdf_failure_policy == "skip"
            and settings.min_pdf_papers
        ):
            if len(ingested_papers) < settings.min_pdf_papers:
                raise RuntimeError(
                    f"Only {len(ingested_papers)} PDF-backed papers were ingested; "
                    f"at least {settings.min_pdf_papers} are required."
                )
        chunks = chunk_documents(documents)
        store = ChromaEvidenceStore(settings.chroma_persist_dir, embeddings)
        count = store.add_documents(chunks)
        trace = state.get("trace", []) + [
            f"Ingested {len(ingested_papers)} papers. Indexed {count} chunks in Chroma."
        ]
        if skipped_papers:
            trace.append(f"Skipped {skipped_papers} papers without usable PDF evidence.")
        return {"papers": ingested_papers, "chunks_indexed": count, "trace": trace}

    return node


def retrieve_node(settings: Settings, embeddings: Embeddings, db: ResearchDatabase):
    def node(state: ResearchState) -> ResearchState:
        plan = state["plan"]
        queries = [state["question"], *plan.subquestions]
        if settings.retrieval_strategy == "hybrid_rrf_qwen3_rerank":
            result = retrieve_hybrid_evidence(
                state["question"], queries, settings, embeddings
            )
            db.save_chunks(state["run_id"], result.chunks)
            trace = state.get("trace", []) + [
                f"Retrieved {result.unique_candidate_count} unique candidates from "
                f"{result.ranked_list_count} Dense/BM25 lists, then selected "
                f"{len(result.chunks)} chunks with Qwen3-Rerank (no instruct)."
            ]
            return {
                "evidence_chunks": result.chunks,
                "evidence_reranker_tokens": result.reranker_tokens,
                "trace": trace,
            }

        store = ChromaEvidenceStore(settings.chroma_persist_dir, embeddings)
        all_chunks = []
        seen: set[str] = set()
        for query in queries:
            for chunk in store.similarity_search(query, k=settings.max_chunks_per_query):
                if chunk.chunk_id and chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    all_chunks.append(chunk)
                    if settings.max_evidence_chunks and len(all_chunks) >= settings.max_evidence_chunks:
                        break
            if settings.max_evidence_chunks and len(all_chunks) >= settings.max_evidence_chunks:
                break
        db.save_chunks(state["run_id"], all_chunks)
        trace = state.get("trace", []) + [
            f"Retrieved {len(all_chunks)} unique evidence chunks."
        ]
        return {"evidence_chunks": all_chunks, "trace": trace}

    return node


def reader_node(
    settings: Settings,
    llm: BaseChatModel,
    db: ResearchDatabase,
):
    read = read_evidence(
        llm,
        batch_size=settings.reader_batch_size,
        min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
    )

    def node(state: ResearchState) -> ResearchState:
        result = read(state)
        db.save_claims(state["run_id"], result.get("claims", []), stage="raw")
        return result

    return node


def build_graph(
    settings: Settings,
    llm: BaseChatModel,
    embeddings: Embeddings,
    db: ResearchDatabase,
    checkpointer: BaseCheckpointSaver | None = None,
):
    graph = StateGraph(ResearchState)
    graph.add_node("planner", plan_research(llm))
    graph.add_node("search", search_node(settings, db))
    graph.add_node("ingest", ingest_node(settings, embeddings))
    graph.add_node("retrieve", retrieve_node(settings, embeddings, db))
    graph.add_node("reader", reader_node(settings, llm, db))
    graph.add_node(
        "verifier",
        verify_claims(
            llm,
            batch_size=settings.verifier_batch_size,
            min_claims_per_subquestion=settings.reader_min_claims_per_subquestion,
        ),
    )
    graph.add_node("writer", write_report(llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "reader")
    graph.add_edge("reader", "verifier")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)
