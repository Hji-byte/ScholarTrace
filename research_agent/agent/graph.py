from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from research_agent.agent.planner import plan_research
from research_agent.agent.reader import read_evidence
from research_agent.agent.search import search_node
from research_agent.agent.verifier import verify_claims
from research_agent.agent.writer import write_report
from research_agent.config import Settings
from research_agent.db.database import ResearchDatabase
from research_agent.retrieval.chunking import chunk_documents
from research_agent.retrieval.vector_store import ChromaEvidenceStore
from research_agent.schema import ResearchState
from research_agent.tools.pdf_loader import load_paper_documents


def ingest_node(settings: Settings, embeddings: Embeddings):
    def node(state: ResearchState) -> ResearchState:
        documents = []
        ingested_papers = []
        skipped_papers = 0
        for paper in state.get("papers", []):
            paper_documents = load_paper_documents(
                paper,
                pdf_dir=Path("data/pdfs"),
                use_pdf=settings.use_pdf,
                pdf_failure_policy=settings.pdf_failure_policy,
            )
            if not paper_documents:
                skipped_papers += 1
                continue
            documents.extend(paper_documents)
            ingested_papers.append(paper)
            if len(ingested_papers) >= settings.max_papers:
                break
        chunks = chunk_documents(documents)
        store = ChromaEvidenceStore(settings.chroma_persist_dir, embeddings)
        count = store.add_documents(chunks)
        trace = state.get("trace", []) + [
            f"Ingested {len(ingested_papers)} papers. Indexed {count} chunks in Chroma."
        ]
        if skipped_papers:
            trace.append(f"Skipped {skipped_papers} papers without usable PDF evidence.")
        if settings.use_pdf and settings.pdf_failure_policy == "skip" and settings.min_pdf_papers:
            if len(ingested_papers) < settings.min_pdf_papers:
                trace.append(
                    f"Warning: only {len(ingested_papers)} PDF-backed papers were ingested; "
                    f"minimum target is {settings.min_pdf_papers}."
                )
        return {"papers": ingested_papers, "chunks_indexed": count, "trace": trace}

    return node


def retrieve_node(settings: Settings, embeddings: Embeddings, db: ResearchDatabase):
    def node(state: ResearchState) -> ResearchState:
        store = ChromaEvidenceStore(settings.chroma_persist_dir, embeddings)
        plan = state["plan"]
        all_chunks = []
        seen: set[str] = set()
        for query in [state["question"], *plan.subquestions]:
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
    graph.add_node("reader", read_evidence(llm, batch_size=settings.reader_batch_size))
    graph.add_node("verifier", verify_claims(llm, batch_size=settings.verifier_batch_size))
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
