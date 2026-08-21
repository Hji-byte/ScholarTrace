from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    paper_id: str
    source_id: str = ""
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    local_pdf_path: str = ""
    venue: str = ""
    citation_count: int | None = None
    source: str = "arxiv"
    rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    matched_query_indexes: list[int] = Field(default_factory=list)
    best_source_rank: int | None = None
    search_occurrence_count: int = 0


class EvidenceChunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    text: str
    url: str = ""
    page: int | None = None
    section: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    claim: str
    category: str = "General"
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    subquestion_ids: list[int] = Field(default_factory=list)


class VerifiedClaim(Claim):
    verified_evidence_chunk_ids: list[str] = Field(default_factory=list)
    supported: bool = False
    reason: str = ""


class SearchIntent(BaseModel):
    """Search concepts grouped as alternatives (OR) and requirements (AND)."""

    purpose: str
    must_groups: list[list[str]] = Field(min_length=1)

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("purpose must not be empty")
        return value

    @field_validator("must_groups")
    @classmethod
    def validate_must_groups(cls, groups: list[list[str]]) -> list[list[str]]:
        normalized_groups: list[list[str]] = []
        for group in groups:
            normalized = list(
                dict.fromkeys(" ".join(term.split()) for term in group if term.strip())
            )
            if not normalized:
                raise ValueError("must_groups must not contain empty groups")
            normalized_groups.append(normalized)
        return normalized_groups


class ResearchPlan(BaseModel):
    subquestions: list[str] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(min_length=1)


class ResearchState(TypedDict, total=False):
    run_id: str
    question: str
    year_from: int
    year_to: int
    source_mode: str
    library_path: str
    plan: ResearchPlan
    papers: list[Paper]
    paper_search_results: list[dict[str, Any]]
    ranked_list_count: int
    search_failure_count: int
    paper_reranker_tokens: int | None
    evidence_reranker_tokens: int | None
    chunks_indexed: int
    evidence_chunks: list[EvidenceChunk]
    claims: list[Claim]
    claim_coverage: dict[str, Any]
    verified_claims: list[VerifiedClaim]
    verified_claim_coverage: dict[str, Any]
    supported_claim_count: int
    rejected_claim_count: int
    report_markdown: str
    report_path: str
    trace: list[str]
