import json

from langchain_core.embeddings import Embeddings

from research_agent.dense_rrf_evaluation import (
    QueryEmbeddingCache,
    fuse_dense_rankings,
    select_ingest_summary,
)


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 1.0]


def _item(chunk_id: str, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "dense_distance": rank / 10,
        "chunk": {
            "chunk_id": chunk_id,
            "paper_id": f"paper-{chunk_id}",
            "title": chunk_id,
            "text": "text",
            "url": "",
            "page": 1,
            "section": None,
            "metadata": {},
        },
    }


def test_fuse_dense_rankings_rewards_chunks_seen_by_multiple_queries():
    query_results = [
        {"query_index": 0, "query": "q0", "chunks": [_item("a", 1), _item("b", 2)]},
        {"query_index": 1, "query": "q1", "chunks": [_item("c", 1), _item("b", 2)]},
    ]
    ranked = fuse_dense_rankings(query_results, rrf_k=60, top_k=2)
    assert ranked[0]["chunk_id"] == "b"
    assert ranked[0]["matched_query_count"] == 2
    assert sum(item["selected"] for item in ranked) == 2


def test_query_embedding_cache_reuses_saved_vector(tmp_path):
    path = tmp_path / "cache.jsonl"
    embeddings = CountingEmbeddings()
    cache = QueryEmbeddingCache(path, "model")
    first, first_hit = cache.get_or_create("q1", "query", embeddings)
    second, second_hit = cache.get_or_create("q1", "query", embeddings)
    assert first == second
    assert not first_hit and second_hit
    assert embeddings.calls == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_select_ingest_summary_uses_first_sorted_run(tmp_path):
    later = tmp_path / "pdf-ingest-csq007-20260727-145353-summary.json"
    earlier = tmp_path / "pdf-ingest-csq007-20260727-144940-summary.json"
    later.write_text(json.dumps({}), encoding="utf-8")
    earlier.write_text(json.dumps({}), encoding="utf-8")
    assert select_ingest_summary(tmp_path, "csq007") == earlier
