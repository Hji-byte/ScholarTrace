import time
from dataclasses import dataclass, field

from research_agent.config import Settings
from research_agent.db.database import ResearchDatabase
from research_agent.reranking.qwen3 import rerank_with_qwen3
from research_agent.schema import Paper, ResearchState
from research_agent.search_providers.arxiv_adapter import compile_arxiv_query
from research_agent.tools.arxiv_search import search_arxiv

RRF_K = 60


@dataclass
class RankedPaper:
    paper: Paper
    rrf_score: float = 0.0
    query_indexes: set[int] = field(default_factory=set)
    best_rank: int = 2**31 - 1


def paper_dedupe_key(paper: Paper) -> str:
    return " ".join(paper.title.lower().split())


def add_ranked_results(
    rankings: dict[str, RankedPaper],
    papers: list[Paper],
    query_index: int,
) -> None:
    for rank, paper in enumerate(papers, start=1):
        key = paper_dedupe_key(paper)
        if not key:
            continue
        if key not in rankings:
            rankings[key] = RankedPaper(paper=paper)

        ranked_paper = rankings[key]
        ranked_paper.rrf_score += 1 / (RRF_K + rank)
        ranked_paper.query_indexes.add(query_index)
        ranked_paper.best_rank = min(ranked_paper.best_rank, rank)


def rank_papers(rankings: dict[str, RankedPaper]) -> list[Paper]:
    ranked = sorted(
        rankings.values(),
        key=lambda item: (
            item.rrf_score,
            len(item.query_indexes),
            -item.best_rank,
            item.paper.year or 0,
            1 if item.paper.abstract else 0,
        ),
        reverse=True,
    )
    return [
        item.paper.model_copy(
            update={"rank": rank, "rrf_score": item.rrf_score}
        )
        for rank, item in enumerate(ranked, start=1)
    ]


def candidate_papers(rankings: dict[str, RankedPaper]) -> list[Paper]:
    """Return every deduplicated candidate without applying a fusion ranking."""
    return [
        item.paper.model_copy(
            update={
                "rrf_score": None,
                "reranker_score": None,
                "matched_query_indexes": sorted(item.query_indexes),
                "best_source_rank": item.best_rank,
                "search_occurrence_count": len(item.query_indexes),
            }
        )
        for item in rankings.values()
    ]


def filter_papers_by_year(
    papers: list[Paper],
    year_from: int | None,
    year_to: int | None,
) -> list[Paper]:
    if year_from is None and year_to is None:
        return papers
    return [
        paper
        for paper in papers
        if paper.year is not None
        and (year_from is None or paper.year >= year_from)
        and (year_to is None or paper.year <= year_to)
    ]


def search_node(settings: Settings, db: ResearchDatabase):
    def node(state: ResearchState) -> ResearchState:
        rankings: dict[str, RankedPaper] = {}
        paper_search_results: list[dict] = []
        ranked_list_count = 0
        search_failure_count = 0
        plan = state["plan"]
        paper_limit = settings.pdf_candidate_limit or settings.max_papers

        for query_index, intent in enumerate(plan.search_intents):
            if query_index > 0 and settings.arxiv_delay_seconds > 0:
                time.sleep(settings.arxiv_delay_seconds)
            provider_query: str | None = None
            try:
                provider_query = compile_arxiv_query(
                    intent,
                    year_from=state.get("year_from"),
                    year_to=state.get("year_to"),
                )
                db.log(
                    state["run_id"],
                    "arxiv_search",
                    {"provider_query": provider_query},
                )
                arxiv_papers = search_arxiv(
                    provider_query,
                    max_results=settings.search_results_per_query,
                )
                arxiv_papers = filter_papers_by_year(
                    arxiv_papers,
                    state.get("year_from"),
                    state.get("year_to"),
                )
                paper_search_results.append(
                    {
                        "query_index": query_index,
                        "search_intent": intent.model_dump(mode="json"),
                        "provider_query": provider_query,
                        "source": "arxiv",
                        "status": "ok",
                        "papers": [
                            {
                                "source_rank": rank,
                                **paper.model_dump(
                                    mode="json",
                                    exclude={
                                        "rank",
                                        "rrf_score",
                                        "reranker_score",
                                        "matched_query_indexes",
                                        "best_source_rank",
                                        "search_occurrence_count",
                                    },
                                ),
                            }
                            for rank, paper in enumerate(arxiv_papers, start=1)
                        ],
                    }
                )
                add_ranked_results(rankings, arxiv_papers, query_index)
                ranked_list_count += 1
            except Exception as exc:
                search_failure_count += 1
                paper_search_results.append(
                    {
                        "query_index": query_index,
                        "search_intent": intent.model_dump(mode="json"),
                        "provider_query": provider_query,
                        "source": "arxiv",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "papers": [],
                    }
                )
                db.log(
                    state["run_id"],
                    "arxiv_search_failed",
                    {"provider_query": provider_query, "error": str(exc)},
                )

        reranker_tokens: int | None = None
        if settings.paper_ranking_strategy == "qwen3_rerank":
            candidates = candidate_papers(rankings)
            papers, reranker_tokens = rerank_with_qwen3(
                state["question"], candidates, settings
            )
            ranking_description = (
                f"Qwen3-Rerank over {len(candidates)} deduplicated candidates from "
                f"{ranked_list_count} arXiv ranked result lists"
            )
        else:
            papers = rank_papers(rankings)
            ranking_description = (
                f"RRF across {ranked_list_count} arXiv ranked result lists"
            )
        papers = papers[:paper_limit]
        db.save_papers(state["run_id"], papers)
        trace = state.get("trace", []) + [
            f"Selected {len(papers)} papers using {ranking_description}."
        ]
        return {
            "papers": papers,
            "paper_search_results": paper_search_results,
            "ranked_list_count": ranked_list_count,
            "search_failure_count": search_failure_count,
            "reranker_tokens": reranker_tokens,
            "trace": trace,
        }

    return node
