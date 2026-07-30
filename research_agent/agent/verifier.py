from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agent.llm_retry import invoke_with_retry
from research_agent.tools.json_utils import extract_json_object
from research_agent.schema import ResearchState, VerifiedClaim


SYSTEM = """You are the claim verifier in a computer science literature-review agent.
Your job is to decide whether each candidate claim is supported by its cited evidence chunks.

Return strict JSON only, with this exact shape:
{"verified_claims":[{"claim":"...", "category":"...", "evidence_chunk_ids":["..."], "supported":true, "reason":"..."}]}

Rules:
- Verify every candidate claim in the current batch; do not drop claims from the output.
- Use only the provided evidence chunks, not outside knowledge.
- A claim is supported only if its cited chunk ids exist and the cited text directly supports the full claim.
- Reject claims whose cited chunk ids are missing, irrelevant, too vague, contradictory, or only partially support the claim.
- Reject claims that add methods, results, comparisons, datasets, dates, or numeric values not stated in the cited chunks.
- If a claim is mostly right but too broad, mark it unsupported and explain the missing support.
- Keep the original claim text, category, and evidence_chunk_ids unchanged.
- The reason should be one short sentence explaining the decision."""


def claim_batches(claims: list, batch_size: int):
    size = max(1, batch_size)
    for start in range(0, len(claims), size):
        yield claims[start : start + size]


def verify_claims(llm: BaseChatModel, batch_size: int = 8):
    def node(state: ResearchState) -> ResearchState:
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in state.get("evidence_chunks", [])
        }
        verified: list[VerifiedClaim] = []
        claims_list = state.get("claims", [])
        batches = list(claim_batches(claims_list, batch_size))
        for batch in batches:
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
            claims = "\n".join(
                f"- {claim.claim} | category={claim.category} | evidence={claim.evidence_chunk_ids}"
                for claim in batch
            )
            response = invoke_with_retry(
                llm,
                [
                    SystemMessage(content=SYSTEM),
                    HumanMessage(content=f"Claims:\n{claims}\n\nEvidence:\n{evidence}"),
                ]
            )
            data = extract_json_object(str(response.content))
            verified.extend(
                VerifiedClaim.model_validate(item)
                for item in data.get("verified_claims", [])
            )
        accepted = sum(1 for claim in verified if claim.supported)
        rejected = len(verified) - accepted
        trace = state.get("trace", []) + [
            f"Verified {len(claims_list)} claims in {len(batches)} batches.",
            f"Verified claims: {accepted} accepted, {rejected} rejected.",
        ]
        return {
            "verified_claims": verified,
            "supported_claim_count": accepted,
            "rejected_claim_count": rejected,
            "trace": trace,
        }

    return node
