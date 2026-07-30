import sqlite3

from research_agent.db.database import ResearchDatabase
from research_agent.schema import Claim, EvidenceChunk, Paper, VerifiedClaim


def test_database_loads_run_question_and_chunks(tmp_path):
    db = ResearchDatabase(tmp_path / "runs.db")
    db.create_run("run-1", "What is RAG evaluation?")
    db.save_chunks(
        "run-1",
        [
            EvidenceChunk(
                chunk_id="chunk-1",
                paper_id="paper-1",
                title="A Paper",
                text="Evidence text.",
                url="https://example.com",
            )
        ],
    )

    chunks = db.load_chunks("run-1")

    assert db.get_run_question("run-1") == "What is RAG evaluation?"
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"


def test_database_saves_claims_by_stage(tmp_path):
    db = ResearchDatabase(tmp_path / "runs.db")
    db.create_run("run-1", "What is RAG evaluation?")
    db.save_claims(
        "run-1",
        [Claim(claim="Raw claim", category="General", evidence_chunk_ids=["c1"])],
        stage="raw",
    )
    db.save_claims(
        "run-1",
        [
            VerifiedClaim(
                claim="Verified claim",
                category="General",
                evidence_chunk_ids=["c1"],
                supported=True,
            )
        ],
        stage="verified",
    )

    raw_claims = db.load_claims("run-1", stage="raw")

    assert len(raw_claims) == 1
    assert raw_claims[0].claim == "Raw claim"


def test_database_keeps_the_same_paper_for_multiple_runs(tmp_path):
    db = ResearchDatabase(tmp_path / "runs.db")
    paper = Paper(
        paper_id="paper-1",
        source_id="arXiv:2309.15217",
        title="Ragas",
        rank=1,
        rrf_score=0.25,
    )
    db.create_run("run-1", "Question one")
    db.create_run("run-2", "Question two")

    db.save_papers("run-1", [paper])
    db.save_papers("run-2", [paper.model_copy(update={"rank": 2})])

    assert db.load_papers("run-1")[0].rank == 1
    assert db.load_papers("run-2")[0].rank == 2


def test_database_migrates_legacy_global_paper_primary_key(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table papers (
            paper_id text primary key,
            run_id text,
            payload text not null
        );
        insert into papers(paper_id, run_id, payload)
        values ('paper-1', 'run-1', '{"paper_id":"paper-1","title":"Legacy"}');
        """
    )
    conn.commit()
    conn.close()

    db = ResearchDatabase(path)

    assert db.load_papers("run-1")[0].title == "Legacy"
    primary_key = [
        row[1]
        for row in sorted(db.conn.execute("pragma table_info(papers)"), key=lambda row: row[5])
        if row[5]
    ]
    assert primary_key == ["run_id", "paper_id"]
