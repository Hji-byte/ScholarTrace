from research_agent.config import Settings
from research_agent.reranking.qwen3 import (
    paper_rerank_document,
    qwen3_rerank_url,
    rerank_with_qwen3,
)
from research_agent.schema import Paper


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ],
            "usage": {"total_tokens": 123},
        }


class FakeSession:
    def __init__(self):
        self.call = None

    def post(self, url, **kwargs):
        self.call = {"url": url, **kwargs}
        return FakeResponse()


def test_qwen3_rerank_url_uses_workspace_endpoint():
    settings = Settings(dashscope_workspace_id="ws-123")

    assert qwen3_rerank_url(settings) == (
        "https://ws-123.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
    )


def test_qwen3_reranker_sends_all_candidates_once_and_maps_indexes():
    settings = Settings(
        dashscope_api_key="secret",
        dashscope_workspace_id="ws-123",
    )
    papers = [
        Paper(paper_id="a", title="Paper A", abstract="Abstract A"),
        Paper(paper_id="b", title="Paper B", abstract="Abstract B"),
    ]
    session = FakeSession()

    ranked, tokens = rerank_with_qwen3(
        "Which paper is relevant?", papers, settings, session=session
    )

    assert [paper.paper_id for paper in ranked] == ["b", "a"]
    assert [paper.rank for paper in ranked] == [1, 2]
    assert [paper.reranker_score for paper in ranked] == [0.9, 0.2]
    assert tokens == 123
    assert session.call["json"]["top_n"] == 2
    assert session.call["json"]["documents"] == [
        "Title: Paper A\n\nAbstract: Abstract A",
        "Title: Paper B\n\nAbstract: Abstract B",
    ]
    assert session.call["headers"]["Authorization"] == "Bearer secret"
    assert "instruct" not in session.call["json"]


def test_paper_rerank_document_works_without_abstract():
    paper = Paper(paper_id="a", title="  A   title  ")

    assert paper_rerank_document(paper) == "Title: A title"
