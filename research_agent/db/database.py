import json
import sqlite3
from pathlib import Path
from typing import Any

from research_agent.schema import Claim, EvidenceChunk, Paper, VerifiedClaim


SCHEMA = """
create table if not exists runs (
    run_id text primary key,
    question text not null,
    report_path text,
    created_at text default current_timestamp
);
create table if not exists papers (
    run_id text not null,
    paper_id text not null,
    rank integer not null,
    rrf_score real,
    payload text not null,
    primary key (run_id, paper_id)
);
create table if not exists chunks (
    chunk_id text primary key,
    run_id text,
    paper_id text,
    payload text not null
);
create table if not exists claims (
    id integer primary key autoincrement,
    run_id text,
    stage text default 'verified',
    payload text not null
);
create table if not exists tool_calls (
    id integer primary key autoincrement,
    run_id text,
    message text not null,
    created_at text default current_timestamp
);
"""


class ResearchDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        columns = {
            row[1]
            for row in self.conn.execute("pragma table_info(claims)").fetchall()
        }
        if "stage" not in columns:
            self.conn.execute("alter table claims add column stage text default 'verified'")

        paper_columns = self.conn.execute("pragma table_info(papers)").fetchall()
        paper_primary_key = [
            row[1]
            for row in sorted(paper_columns, key=lambda row: row[5])
            if row[5]
        ]
        if paper_primary_key == ["paper_id"]:
            self.conn.execute("alter table papers rename to papers_legacy")
            self.conn.execute(
                """
                create table papers (
                    run_id text not null,
                    paper_id text not null,
                    rank integer not null,
                    rrf_score real,
                    payload text not null,
                    primary key (run_id, paper_id)
                )
                """
            )
            self.conn.execute(
                """
                insert or ignore into papers(run_id, paper_id, rank, rrf_score, payload)
                select run_id, paper_id, 0, null, payload
                from papers_legacy
                where run_id is not null
                """
            )
            self.conn.execute("drop table papers_legacy")

    def create_run(self, run_id: str, question: str) -> None:
        self.conn.execute(
            "insert or replace into runs(run_id, question) values (?, ?)",
            (run_id, question),
        )
        self.conn.commit()

    def set_report_path(self, run_id: str, report_path: str) -> None:
        self.conn.execute(
            "update runs set report_path = ? where run_id = ?",
            (report_path, run_id),
        )
        self.conn.commit()

    def get_run_question(self, run_id: str) -> str:
        row = self.conn.execute(
            "select question from runs where run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        return str(row[0])

    def save_papers(self, run_id: str, papers: list[Paper]) -> None:
        self.conn.execute("delete from papers where run_id = ?", (run_id,))
        self.conn.executemany(
            """
            insert into papers(run_id, paper_id, rank, rrf_score, payload)
            values (?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    paper.paper_id,
                    paper.rank or rank,
                    paper.rrf_score,
                    paper.model_dump_json(),
                )
                for rank, paper in enumerate(papers, start=1)
            ],
        )
        self.conn.commit()

    def load_papers(self, run_id: str) -> list[Paper]:
        rows = self.conn.execute(
            "select payload, rank, rrf_score from papers where run_id = ? order by rank",
            (run_id,),
        ).fetchall()
        return [
            Paper.model_validate_json(payload).model_copy(
                update={"rank": rank, "rrf_score": rrf_score}
            )
            for payload, rank, rrf_score in rows
        ]

    def save_chunks(self, run_id: str, chunks: list[EvidenceChunk]) -> None:
        self.conn.executemany(
            "insert or replace into chunks(chunk_id, run_id, paper_id, payload) values (?, ?, ?, ?)",
            [(chunk.chunk_id, run_id, chunk.paper_id, chunk.model_dump_json()) for chunk in chunks],
        )
        self.conn.commit()

    def load_chunks(self, run_id: str) -> list[EvidenceChunk]:
        rows = self.conn.execute(
            "select payload from chunks where run_id = ? order by chunk_id",
            (run_id,),
        ).fetchall()
        return [EvidenceChunk.model_validate_json(row[0]) for row in rows]

    def save_claims(
        self,
        run_id: str,
        claims: list[Claim] | list[VerifiedClaim],
        stage: str = "verified",
    ) -> None:
        self.conn.execute(
            "delete from claims where run_id = ? and stage = ?",
            (run_id, stage),
        )
        self.conn.executemany(
            "insert into claims(run_id, stage, payload) values (?, ?, ?)",
            [(run_id, stage, claim.model_dump_json()) for claim in claims],
        )
        self.conn.commit()

    def load_claims(self, run_id: str, stage: str = "raw") -> list[Claim]:
        rows = self.conn.execute(
            "select payload from claims where run_id = ? and stage = ? order by id",
            (run_id, stage),
        ).fetchall()
        return [Claim.model_validate_json(row[0]) for row in rows]

    def log(self, run_id: str, message: str, payload: dict[str, Any] | None = None) -> None:
        full_message = message if payload is None else f"{message}: {json.dumps(payload, ensure_ascii=False)}"
        self.conn.execute(
            "insert into tool_calls(run_id, message) values (?, ?)",
            (run_id, full_message),
        )
        self.conn.commit()
