from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scholar_trace.agent.llm_retry import invoke_with_retry
from scholar_trace.tools.json_utils import extract_json_object
from scholar_trace.schema import ResearchState, VerifiedClaim


SYSTEM = """You are the claim verifier in a computer science literature-review agent.
Your job is to decide whether each candidate claim is supported by its cited evidence chunks.

Return strict JSON only, with this exact shape:
{"verifications":[{"claim_id":1,"verified_evidence_chunk_ids":["..."],"supported":true,"reason":"..."}]}

For each claim:
1. Read the claim and the evidence chunks listed in its cited evidence chunk ids.
2. Identify which of those chunks provide evidence for the claim and decide whether they fully support the claim as written.
3. If the identified chunks fully support the claim, set supported to true and put only their ids in verified_evidence_chunk_ids.
4. If they do not, set supported to false and return an empty verified_evidence_chunk_ids list.

Constraints:
- Return exactly one verification for every claim_id in the current batch.
- Use only the provided evidence chunks, not outside knowledge.
- Use only ids from the claim's cited evidence chunk ids. Do not add or rewrite ids.
- The reason should be one short sentence explaining what the evidence supports or what support is missing."""


class VerificationDecision(BaseModel):
    claim_id: int
    verified_evidence_chunk_ids: list[str] = Field(default_factory=list)
    supported: bool
    reason: str


def claim_batches(claims: list, batch_size: int):
    size = max(1, batch_size)
    for start in range(0, len(claims), size):
        yield claims[start : start + size]


def parse_verified_batch(text: str, batch: list) -> list[VerifiedClaim]:
    data = extract_json_object(text)
    decisions = [
        VerificationDecision.model_validate(item)
        for item in data.get("verifications", [])
    ]
    expected_claim_ids = set(range(1, len(batch) + 1))
    returned_claim_ids = [decision.claim_id for decision in decisions]
    if len(decisions) != len(batch) or set(returned_claim_ids) != expected_claim_ids:
        raise ValueError(
            "Verifier must return every input claim_id exactly once; "
            f"expected {sorted(expected_claim_ids)}, got {returned_claim_ids}"
        )

    verified: list[VerifiedClaim] = []
    for decision in sorted(decisions, key=lambda item: item.claim_id):
        source_claim = batch[decision.claim_id - 1]
        verified_ids = decision.verified_evidence_chunk_ids
        unknown_verified_ids = set(verified_ids) - set(source_claim.evidence_chunk_ids)
        if unknown_verified_ids:
            raise ValueError(
                "Verifier returned verified evidence ids outside the original "
                f"claim: {sorted(unknown_verified_ids)}"
            )
        if decision.supported and not verified_ids:
            raise ValueError(
                "Verifier marked a claim supported without verified evidence"
            )
        if not decision.supported and verified_ids:
            raise ValueError(
                "Verifier returned verified evidence for an unsupported claim"
            )
        verified.append(
            VerifiedClaim(
                **source_claim.model_dump(),
                verified_evidence_chunk_ids=verified_ids,
                supported=decision.supported,
                reason=decision.reason,
            )
        )
    return verified


def build_verified_claim_coverage(
    subquestions: list[str],
    claims: list[VerifiedClaim],
    min_claims_per_subquestion: int,
) -> dict:
    minimum = max(1, min_claims_per_subquestion)
    items = []
    supported = [claim for claim in claims if claim.supported]
    for subquestion_id, subquestion in enumerate(subquestions, start=1):
        matched = [
            claim for claim in supported if subquestion_id in claim.subquestion_ids
        ]
        chunk_ids = {
            chunk_id
            for claim in matched
            for chunk_id in claim.verified_evidence_chunk_ids
        }
        items.append(
            {
                "subquestion_id": subquestion_id,
                "subquestion": subquestion,
                "claim_count": len(matched),
                "evidence_chunk_count": len(chunk_ids),
                "covered": len(matched) >= minimum,
            }
        )
    return {
        "minimum_claims_per_subquestion": minimum,
        "all_subquestions_covered": all(item["covered"] for item in items),
        "subquestions": items,
    }


def verify_claims(
    llm: BaseChatModel,
    batch_size: int = 8,
    min_claims_per_subquestion: int = 2,
    validation_attempts: int = 2,
):
    def node(state: ResearchState) -> ResearchState:
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in state.get("evidence_chunks", [])
        }
        verified: list[VerifiedClaim] = []
        claims_list = state.get("claims", [])
        batches = list(claim_batches(claims_list, batch_size))
        for batch in batches:
            for claim in batch:
                missing_ids = set(claim.evidence_chunk_ids) - set(chunks_by_id)
                if missing_ids:
                    raise ValueError(
                        "Claim cites evidence chunks that are not available: "
                        f"{sorted(missing_ids)}"
                    )
            cited_chunk_ids = {
                chunk_id
                for claim in batch
                for chunk_id in claim.evidence_chunk_ids
            }
            evidence = "\n\n".join(
                f"Chunk id: {chunk.chunk_id}\nTitle: {chunk.title}\nText: {chunk.text}"
                for chunk_id, chunk in chunks_by_id.items()
                if chunk_id in cited_chunk_ids
            )
            claims = "\n\n".join(
                f"Claim id: {claim_id}\n"
                f"Claim: {claim.claim}\n"
                f"Cited evidence chunk ids: {claim.evidence_chunk_ids}"
                for claim_id, claim in enumerate(batch, start=1)
            )
            messages = [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=f"Claims:\n{claims}\n\nEvidence:\n{evidence}"),
            ]
            attempts = max(1, validation_attempts)
            for attempt in range(1, attempts + 1):
                response = invoke_with_retry(llm, messages)
                try:
                    verified.extend(
                        parse_verified_batch(str(response.content), batch)
                    )
                    break
                except (ValueError, TypeError) as exc:
                    if attempt >= attempts:
                        raise
                    messages.extend(
                        [
                            response,
                            HumanMessage(
                                content=(
                                    f"The previous JSON was invalid: {exc}. "
                                    "Return the complete corrected JSON for every "
                                    "claim_id in this batch."
                                )
                            ),
                        ]
                    )
        accepted = sum(1 for claim in verified if claim.supported)
        rejected = len(verified) - accepted
        subquestions = list(state["plan"].subquestions) if state.get("plan") else []
        coverage = build_verified_claim_coverage(
            subquestions,
            verified,
            min_claims_per_subquestion,
        )
        uncovered = [
            item["subquestion_id"]
            for item in coverage["subquestions"]
            if not item["covered"]
        ]
        trace = state.get("trace", []) + [
            f"Verified {len(claims_list)} claims in {len(batches)} batches.",
            f"Verified claims: {accepted} accepted, {rejected} rejected.",
            (
                "All subquestions retain verified claim coverage."
                if not uncovered
                else f"Subquestions below verified claim coverage: {uncovered}."
            ),
        ]
        return {
            "verified_claims": verified,
            "verified_claim_coverage": coverage,
            "supported_claim_count": accepted,
            "rejected_claim_count": rejected,
            "trace": trace,
        }

    return node
