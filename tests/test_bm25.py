from langchain_core.documents import Document

from research_agent.bm25_rrf_evaluation import fuse_bm25_rankings
from research_agent.retrieval.bm25 import BM25Index, tokenize_for_bm25


def test_tokenizer_preserves_common_cs_compound_terms():
    assert tokenize_for_bm25("KV-cache, text-to-SQL, C++ and RAG") == [
        "kv-cache",
        "text-to-sql",
        "c++",
        "and",
        "rag",
    ]


def test_bm25_ranks_matching_document_first():
    documents = [
        Document(page_content="dense embedding semantic retrieval", metadata={"chunk_id": "dense"}),
        Document(page_content="sparse lexical bm25 retrieval", metadata={"chunk_id": "bm25"}),
    ]
    hits = BM25Index(documents).search("bm25 lexical retrieval", k=2)
    assert hits[0].document.metadata["chunk_id"] == "bm25"
    assert hits[0].score > hits[1].score


def test_bm25_uses_cs_tokenizer_without_adding_query_vocabulary():
    documents = [
        Document(page_content="KV-cache and text-to-SQL", metadata={"chunk_id": "cs"}),
    ]
    index = BM25Index(documents)
    vocabulary_before = dict(index.tokenizer.get_vocab_dict())

    hits = index.search("C++ with KV-cache", k=1)

    assert hits[0].document.metadata["chunk_id"] == "cs"
    assert "kv-cache" in vocabulary_before
    assert "text-to-sql" in vocabulary_before
    assert "c++" not in index.tokenizer.get_vocab_dict()


def test_bm25_does_not_return_zero_score_documents():
    documents = [
        Document(page_content="dense semantic retrieval", metadata={"chunk_id": "dense"}),
        Document(page_content="computer vision segmentation", metadata={"chunk_id": "vision"}),
    ]
    assert BM25Index(documents).search("quantum cryptography", k=2) == []


def test_bm25_rrf_rewards_multi_query_hits():
    def item(chunk_id: str, rank: int, score: float) -> dict:
        return {
            "rank": rank,
            "chunk_id": chunk_id,
            "bm25_score": score,
            "chunk": {
                "chunk_id": chunk_id,
                "paper_id": chunk_id,
                "title": chunk_id,
                "text": "text",
                "url": "",
                "page": 1,
                "section": None,
                "metadata": {},
            },
        }

    rankings = [
        {"query_index": 0, "query": "q0", "chunks": [item("a", 1, 3), item("b", 2, 2)]},
        {"query_index": 1, "query": "q1", "chunks": [item("c", 1, 3), item("b", 2, 2)]},
    ]
    fused = fuse_bm25_rankings(rankings, rrf_k=60, top_k=2)
    assert fused[0]["chunk_id"] == "b"
    assert fused[0]["matched_query_count"] == 2
    assert sum(item["selected"] for item in fused) == 2
