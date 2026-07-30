from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent.graph import build_graph
from research_agent.agent.search import search_node
from research_agent.config import Settings
from research_agent.db.database import ResearchDatabase
from research_agent.schema import Paper, ResearchPlan, SearchIntent


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text) % 7)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, float(len(text) % 7)]


def make_plan(*terms: str) -> ResearchPlan:
    return ResearchPlan(
        search_intents=[
            SearchIntent(purpose=term, must_groups=[[term]])
            for term in terms
        ]
    )


def test_graph_smoke_with_mocked_llm_and_search(tmp_path: Path, monkeypatch):
    def fake_search(query: str, max_results: int = 8):
        return [
            Paper(
                paper_id="paper-1",
                title="Speculative Decoding for Faster Inference",
                authors=["A. Researcher"],
                year=2023,
                abstract="Speculative decoding uses a draft model to reduce LLM inference latency.",
                url="https://example.com/paper-1",
                pdf_url="",
            )
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)

    llm = FakeListChatModel(
        responses=[
            '{"subquestions":["How does speculative decoding work?"],"search_intents":[{"purpose":"speculative decoding methods","must_groups":[["speculative decoding"],["LLM inference"]]}]}',
            '{"claims":[{"claim":"Speculative decoding can reduce inference latency by using a draft model.","category":"Speculative Decoding","evidence_chunk_ids":["paper-1-chunk-0000"]}]}',
            '{"verified_claims":[{"claim":"Speculative decoding can reduce inference latency by using a draft model.","category":"Speculative Decoding","evidence_chunk_ids":["paper-1-chunk-0000"],"supported":true,"reason":"The cited chunk states this directly."}]}',
            "# Literature Review\n\nSpeculative decoding uses a draft model [paper-1-chunk-0000].",
        ]
    )
    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        sqlite_db_path=tmp_path / "runs.db",
        max_papers=1,
        max_chunks_per_query=2,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("test-run", "How can LLM inference be accelerated?")
    app = build_graph(settings, llm, TinyEmbeddings(), db)

    state = app.invoke(
        {
            "run_id": "test-run",
            "question": "How can LLM inference be accelerated?",
            "trace": [],
        }
    )

    assert "Literature Review" in state["report_markdown"]
    assert state["chunks_indexed"] >= 1
    assert state["verified_claims"][0].supported is True
    assert any("Generated final Markdown report" in item for item in state["trace"])


def test_graph_persists_state_with_checkpointer(tmp_path: Path, monkeypatch):
    def fake_search(query: str, max_results: int = 8):
        return [
            Paper(
                paper_id="paper-1",
                title="Speculative Decoding for Faster Inference",
                authors=["A. Researcher"],
                year=2023,
                abstract="Speculative decoding uses a draft model to reduce LLM inference latency.",
                url="https://example.com/paper-1",
            )
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)

    llm = FakeListChatModel(
        responses=[
            '{"subquestions":["How does speculative decoding work?"],"search_intents":[{"purpose":"speculative decoding methods","must_groups":[["speculative decoding"],["LLM inference"]]}]}',
            '{"claims":[{"claim":"Speculative decoding can reduce latency.","category":"Speculative Decoding","evidence_chunk_ids":["paper-1-chunk-0000"]}]}',
            '{"verified_claims":[{"claim":"Speculative decoding can reduce latency.","category":"Speculative Decoding","evidence_chunk_ids":["paper-1-chunk-0000"],"supported":true,"reason":"Supported by the cited chunk."}]}',
            "# Literature Review\n\nSpeculative decoding can reduce latency [paper-1-chunk-0000].",
        ]
    )
    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        sqlite_db_path=tmp_path / "runs.db",
        max_papers=1,
        max_chunks_per_query=2,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("checkpoint-run", "How can LLM inference be accelerated?")
    checkpointer = InMemorySaver()
    app = build_graph(settings, llm, TinyEmbeddings(), db, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "checkpoint-run"}}

    app.invoke(
        {
            "run_id": "checkpoint-run",
            "question": "How can LLM inference be accelerated?",
            "trace": [],
        },
        config,
    )

    snapshot = app.get_state(config)
    assert "report_markdown" in snapshot.values
    assert snapshot.next == ()


def test_search_node_requests_configured_results_per_query(tmp_path: Path, monkeypatch):
    seen_max_results = []

    def fake_search(query: str, max_results: int = 8):
        seen_max_results.append(max_results)
        return [
            Paper(
                paper_id=f"paper-{query}",
                title=f"Paper for {query}",
                abstract="This paper should be ingested.",
                pdf_url="https://example.com/paper.pdf",
            ),
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(
        sqlite_db_path=tmp_path / "runs.db",
        search_results_per_query=3,
        max_papers=2,
        arxiv_delay_seconds=0,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("search-run", "Which evidence is usable?")
    node = search_node(settings, db)

    state = node(
        {
            "run_id": "search-run",
            "plan": make_plan("query one", "query two"),
            "trace": [],
        }
    )

    assert seen_max_results == [3, 3]
    assert len(state["papers"]) == 2


def test_search_node_can_keep_more_pdf_candidates_than_max_papers(tmp_path: Path, monkeypatch):
    seen_max_results = []

    def fake_search(query: str, max_results: int = 8):
        seen_max_results.append(max_results)
        offset = 0 if "query one" in query else 10
        return [
            Paper(
                paper_id=f"paper-{offset + index}",
                title=f"Paper {offset + index}",
                year=2024 - offset - index,
                abstract="Candidate paper.",
            )
            for index in range(max_results)
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(
        sqlite_db_path=tmp_path / "runs.db",
        search_results_per_query=2,
        max_papers=2,
        pdf_candidate_limit=3,
        arxiv_delay_seconds=0,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("search-run", "Which evidence is usable?")
    node = search_node(settings, db)

    state = node(
        {
            "run_id": "search-run",
            "plan": make_plan("query one", "query two"),
            "trace": [],
        }
    )

    assert seen_max_results == [2, 2]
    assert len(state["papers"]) == 3


def test_search_node_uses_reciprocal_rank_fusion_across_queries(tmp_path: Path, monkeypatch):
    def fake_search(query: str, max_results: int = 8):
        papers = [
            Paper(paper_id=f"{query}-top", title=f"Top paper for {query}"),
            Paper(paper_id=f"{query}-2", title=f"Second paper for {query}"),
            Paper(paper_id=f"{query}-3", title=f"Third paper for {query}"),
            Paper(paper_id=f"{query}-4", title=f"Fourth paper for {query}"),
            Paper(paper_id="shared", title="Shared Evaluation Paper"),
        ]
        return papers[:max_results]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(
        sqlite_db_path=tmp_path / "runs.db",
        max_papers=5,
        arxiv_delay_seconds=0,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("rrf-run", "How should RAG systems be evaluated?")
    node = search_node(settings, db)

    state = node(
        {
            "run_id": "rrf-run",
            "plan": make_plan("query one", "query two", "query three"),
            "trace": [],
        }
    )

    assert state["papers"][0].paper_id == "shared"
    assert state["papers"][0].rank == 1
    assert state["papers"][0].rrf_score is not None
    assert "using RRF across 3 arXiv ranked result lists" in state["trace"][0]


def test_search_node_can_rerank_all_deduplicated_candidates_without_rrf(
    tmp_path: Path, monkeypatch
):
    def fake_search(query: str, max_results: int = 8):
        suffix = "one" if "query one" in query else "two"
        return [
            Paper(paper_id="shared", title="Shared Paper", abstract="Shared"),
            Paper(
                paper_id=suffix,
                title=f"Paper {suffix}",
                abstract=f"Abstract {suffix}",
            ),
        ]

    received = {}

    def fake_rerank(question, papers, settings):
        received["question"] = question
        received["papers"] = papers
        ranked = list(reversed(papers))
        return [
            paper.model_copy(update={"rank": rank, "reranker_score": 1 / rank})
            for rank, paper in enumerate(ranked, start=1)
        ], 77

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    monkeypatch.setattr("research_agent.agent.search.rerank_with_qwen3", fake_rerank)
    settings = Settings(
        sqlite_db_path=tmp_path / "runs.db",
        paper_ranking_strategy="qwen3_rerank",
        max_papers=3,
        arxiv_delay_seconds=0,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("rerank-run", "Which papers answer the question?")

    state = search_node(settings, db)(
        {
            "run_id": "rerank-run",
            "question": "Which papers answer the question?",
            "plan": make_plan("query one", "query two"),
            "trace": [],
        }
    )

    assert received["question"] == "Which papers answer the question?"
    assert len(received["papers"]) == 3
    shared = next(paper for paper in received["papers"] if paper.paper_id == "shared")
    assert shared.matched_query_indexes == [0, 1]
    assert shared.best_source_rank == 1
    assert shared.search_occurrence_count == 2
    assert state["reranker_tokens"] == 77
    assert all(paper.rrf_score is None for paper in state["papers"])
    assert "using Qwen3-Rerank over 3 deduplicated candidates" in state["trace"][0]


def test_search_node_applies_structured_year_filter(tmp_path: Path, monkeypatch):
    seen_year_range = []

    def fake_search(provider_query: str, max_results: int = 8):
        seen_year_range.append(provider_query)
        return [
            Paper(paper_id="old", title="Old Paper", year=2017),
            Paper(paper_id="in-range", title="In Range Paper", year=2021),
            Paper(paper_id="unknown", title="Unknown Year Paper"),
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(sqlite_db_path=tmp_path / "runs.db", max_papers=10)
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("year-run", "Recent work")

    state = search_node(settings, db)(
        {
            "run_id": "year-run",
            "plan": make_plan("recent work"),
            "year_from": 2018,
            "year_to": 2025,
            "trace": [],
        }
    )

    assert [paper.paper_id for paper in state["papers"]] == ["in-range"]
    assert seen_year_range == [
        '(all:"recent work") AND submittedDate:[201801010000 TO 202512312359]'
    ]
    assert state["papers"][0].rank == 1
    assert state["papers"][0].rrf_score == 1 / 61


def test_search_node_saves_structured_intent_and_actual_arxiv_query(tmp_path: Path, monkeypatch):
    seen = []

    def fake_search(provider_query: str, max_results: int = 8):
        seen.append(provider_query)
        return [Paper(paper_id="paper", title="Serverless Cold Start", year=2024)]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(sqlite_db_path=tmp_path / "runs.db", max_papers=1)
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("intent-run", "Serverless cold starts")
    intent = SearchIntent(
        purpose="serverless cold-start mitigation",
        must_groups=[
            ["serverless computing", "function as a service"],
            ["cold start", "startup latency"],
        ],
    )

    state = search_node(settings, db)(
        {
            "run_id": "intent-run",
            "plan": ResearchPlan(search_intents=[intent]),
            "trace": [],
        }
    )

    assert seen == [
        '(all:"serverless computing" OR all:"function as a service") AND '
        '(all:"cold start" OR all:"startup latency")'
    ]
    assert state["paper_search_results"][0]["search_intent"] == intent.model_dump()
    assert state["paper_search_results"][0]["provider_query"] == seen[0]


def test_search_node_throttles_consecutive_arxiv_requests(tmp_path: Path, monkeypatch):
    sleeps = []

    monkeypatch.setattr("research_agent.agent.search.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "research_agent.agent.search.search_arxiv",
        lambda query, max_results=8: [
            Paper(paper_id=query, title=f"Paper for {query}", year=2024)
        ],
    )
    settings = Settings(
        sqlite_db_path=tmp_path / "runs.db",
        max_papers=2,
        arxiv_delay_seconds=3.0,
    )
    db = ResearchDatabase(settings.sqlite_db_path)
    db.create_run("throttle-run", "Throttle arXiv")

    search_node(settings, db)(
        {
            "run_id": "throttle-run",
            "plan": make_plan("one", "two"),
            "trace": [],
        }
    )

    assert sleeps == [3.0]
