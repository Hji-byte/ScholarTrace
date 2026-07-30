import pytest


@pytest.fixture(autouse=True)
def isolate_paper_ranking_strategy(monkeypatch):
    """Keep unit tests from inheriting a live reranker setting from .env."""
    monkeypatch.setenv("PAPER_RANKING_STRATEGY", "rrf")
