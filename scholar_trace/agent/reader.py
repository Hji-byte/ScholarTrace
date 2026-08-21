from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_trace.agent.llm_retry import invoke_with_retry
from scholar_trace.tools.json_utils import extract_json_object
from scholar_trace.schema import Claim, ResearchState


SYSTEM = """You are the evidence reader in a computer science literature-review agent.
Your job is to extract precise, evidence-backed claims from the provided chunks.

Return strict JSON only, with this exact shape:
{"claims":[{"claim":"...", "category":"...", "evidence_chunk_ids":["..."]}]}

Field meanings:
- claim: one atomic statement that is directly supported by the cited chunks.
- category: a short topic label used later to group claims in the report.
- evidence_chunk_ids: the chunk ids that directly support the claim.

Rules:
- Each claim must be directly supported by one or more provided chunks.
- Use only chunk ids that appear in the evidence context.
- Make each claim atomic: one specific method, result, limitation, comparison, or trend per claim.
- Prefer claims about mechanisms, method categories, evaluation findings, tradeoffs, limitations, and open problems.
- Do not invent citations, paper titles, benchmark results, dates, or numeric values that are not in the chunks.
- Do not make broad field-level claims unless the chunks explicitly support them.
- If multiple chunks support the same idea, include all relevant chunk ids.
- If the evidence is thin, return fewer claims rather than padding.
- Choose concise, evidence-driven categories that would make good literature-review subsections for the current research question."""


CONSOLIDATION_SYSTEM = """You are the global claim organizer in a computer science literature-review agent.
Consolidate the candidate claims and map them to the research subquestions.

Return strict JSON only, with this exact shape:
{"claims":[{"claim":"...","category":"...","evidence_chunk_ids":["..."],"subquestion_ids":[1]}]}

Rules:
- Merge claims only when they express the same atomic proposition.
- When merging equivalent claims, preserve one precise statement and union all directly supporting evidence_chunk_ids.
- Do not merge merely related claims or create a broader statement than the candidates support.
- Do not add facts, numbers, methods, comparisons, or evidence IDs absent from the candidates.
- The user message lists the subquestions with integer labels such as 1., 2., and 3.
- Set subquestion_ids to the integer labels of all subquestions that the claim directly helps answer.
- A claim may map to multiple subquestions, or to an empty list if it helps answer only the overall question.
- Return fewer claims only when candidates are semantically duplicated or do not help answer the research question or any subquestion."""


def parse_claims(text: str) -> list[Claim]:
    data = extract_json_object(text)
    return [Claim.model_validate(item) for item in data.get("claims", [])]


def chunk_batches(chunks: list, batch_size: int):
    size = max(1, batch_size)
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def consolidate_claims(
    llm: BaseChatModel,
    question: str,
    subquestions: list[str],
    claims: list[Claim],
    valid_chunk_ids: set[str],
    validation_attempts: int = 2,
) -> list[Claim]:
    if not claims:
        return []
    subquestion_text = "\n".join(
        f"{index}. {subquestion}" for index, subquestion in enumerate(subquestions, start=1)
    ) or "(No explicit subquestions; use an empty subquestion_ids list.)"
    candidate_text = "\n\n".join(
        (
            f"Candidate claim {index}:\n"
            f"Claim: {claim.claim}\n"
            f"Category: {claim.category}\n"
            f"Evidence chunk ids: {', '.join(claim.evidence_chunk_ids)}"
        )
        for index, claim in enumerate(claims, start=1)
    )
    messages = [
        SystemMessage(content=CONSOLIDATION_SYSTEM),
        HumanMessage(
            content=(
                f"Research question: {question}\n\n"
                f"Numbered subquestions:\n{subquestion_text}\n\n"
                f"Candidate claims:\n{candidate_text}"
            )
        ),
    ]
    allowed_subquestions = set(range(1, len(subquestions) + 1))
    attempts = max(1, validation_attempts)
    for attempt in range(1, attempts + 1):
        response = invoke_with_retry(llm, messages)
        try:
            consolidated = parse_claims(str(response.content))
            for claim in consolidated:
                if not claim.evidence_chunk_ids:
                    raise ValueError("Consolidated claim has no evidence chunk ids")
                unknown_chunks = set(claim.evidence_chunk_ids) - valid_chunk_ids
                if unknown_chunks:
                    raise ValueError(
                        "Consolidated claim cites unknown chunks: "
                        f"{sorted(unknown_chunks)}"
                    )
                unknown_subquestions = (
                    set(claim.subquestion_ids) - allowed_subquestions
                )
                if unknown_subquestions:
                    raise ValueError(
                        "Consolidated claim uses unknown subquestion ids: "
                        f"{sorted(unknown_subquestions)}"
                    )
            return consolidated
        except (ValueError, TypeError) as exc:
            if attempt >= attempts:
                raise
            messages.extend(
                [
                    response,
                    HumanMessage(
                        content=(
                            f"The previous JSON was invalid: {exc}. "
                            "Return the complete corrected JSON. Copy evidence_chunk_ids "
                            "exactly from the candidate claims and use only the supplied "
                            "subquestion integer labels."
                        )
                    ),
                ]
            )
    raise RuntimeError("Claim consolidation exhausted validation attempts")


def build_claim_coverage(
    subquestions: list[str],
    claims: list[Claim],
    min_claims_per_subquestion: int,
) -> dict:
    minimum = max(1, min_claims_per_subquestion)
    items = []
    for subquestion_id, subquestion in enumerate(subquestions, start=1):
        matched = [
            claim for claim in claims if subquestion_id in claim.subquestion_ids
        ]
        chunk_ids = {
            chunk_id for claim in matched for chunk_id in claim.evidence_chunk_ids
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


def chunks_for_subquestion(chunks: list, subquestion_id: int) -> list:
    """Return retained evidence chunks originally hit by a subquestion query."""
    matched = []
    for chunk in chunks:
        query_hits = chunk.metadata.get("query_hits", [])
        if any(
            isinstance(hit, dict) and hit.get("query_index") == subquestion_id
            for hit in query_hits
        ):
            matched.append(chunk)
    return matched


def extract_additional_claims(
    llm: BaseChatModel,
    question: str,
    subquestion_id: int,
    subquestion: str,
    chunks: list,
    existing_claims: list[Claim],
    batch_size: int,
) -> list[Claim]:
    additional: list[Claim] = []
    existing_text = "\n".join(
        f"- {claim.claim}" for claim in existing_claims
    ) or "(No claims are currently mapped to this subquestion.)"
    batches = list(chunk_batches(chunks, batch_size))
    for index, batch in enumerate(batches, start=1):
        context = "\n\n".join(
            f"Chunk id: {chunk.chunk_id}\nTitle: {chunk.title}\nText: {chunk.text}"
            for chunk in batch
        )
        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(
                    content=(
                        f"Original research question: {question}\n\n"
                        f"Target subquestion {subquestion_id}: {subquestion}\n\n"
                        f"Existing claims for this subquestion:\n{existing_text}\n\n"
                        "Extract additional claims from the provided evidence chunks that "
                        "directly help answer the target subquestion and are not already "
                        "represented by the existing claims.\n\n"
                        "Return only the new claims extracted in this request. Do not "
                        "return or copy any of the existing claims.\n\n"
                        "If the provided evidence chunks do not directly support any "
                        "additional claim, return {\"claims\": []}.\n\n"
                        f"Evidence batch {index} of {len(batches)}:\n{context}"
                    )
                ),
            ],
        )
        additional.extend(parse_claims(str(response.content)))
    return additional


def supplement_claim_coverage(
    llm: BaseChatModel,
    question: str,
    subquestions: list[str],
    chunks: list,
    claims: list[Claim],
    batch_size: int,
    min_claims_per_subquestion: int,
) -> tuple[list[Claim], dict, list[Claim], list[int]]:
    coverage = build_claim_coverage(
        subquestions,
        claims,
        min_claims_per_subquestion,
    )
    initial_uncovered = [
        item for item in coverage["subquestions"] if not item["covered"]
    ]
    additional_claims: list[Claim] = []
    retried_subquestions: list[int] = []
    for item in initial_uncovered:
        subquestion_id = item["subquestion_id"]
        relevant_chunks = chunks_for_subquestion(chunks, subquestion_id)
        if not relevant_chunks:
            continue
        retried_subquestions.append(subquestion_id)
        existing_claims = [
            claim for claim in claims if subquestion_id in claim.subquestion_ids
        ]
        additional_claims.extend(
            extract_additional_claims(
                llm,
                question,
                subquestion_id,
                item["subquestion"],
                relevant_chunks,
                existing_claims,
                batch_size,
            )
        )
    if additional_claims:
        claims = consolidate_claims(
            llm,
            question,
            subquestions,
            [*claims, *additional_claims],
            {chunk.chunk_id for chunk in chunks},
        )
        coverage = build_claim_coverage(
            subquestions,
            claims,
            min_claims_per_subquestion,
        )
    return claims, coverage, additional_claims, retried_subquestions


def read_evidence(
    llm: BaseChatModel,
    batch_size: int = 10,
    min_claims_per_subquestion: int = 2,
):
    def node(state: ResearchState) -> ResearchState:
        chunks = state.get("evidence_chunks", [])
        claims: list[Claim] = []
        batches = list(chunk_batches(chunks, batch_size))
        for index, batch in enumerate(batches, start=1):
            context = "\n\n".join(
                f"Chunk id: {chunk.chunk_id}\nTitle: {chunk.title}\nText: {chunk.text}"
                for chunk in batch
            )
            response = invoke_with_retry(
                llm,
                [
                    SystemMessage(content=SYSTEM),
                    HumanMessage(
                        content=(
                            f"Question: {state['question']}\n\n"
                            "Extract 5 to 12 claims that help answer the research question.\n\n"
                            f"Evidence batch {index} of {len(batches)}:\n{context}"
                        )
                    ),
                ]
            )
            claims.extend(parse_claims(str(response.content)))
        batch_claims = claims
        subquestions = list(state.get("plan").subquestions) if state.get("plan") else []
        claims = consolidate_claims(
            llm,
            state["question"],
            subquestions,
            batch_claims,
            {chunk.chunk_id for chunk in chunks},
        )
        claims, coverage, additional_claims, retried_subquestions = (
            supplement_claim_coverage(
                llm,
                state["question"],
                subquestions,
                chunks,
                claims,
                batch_size,
                min_claims_per_subquestion,
            )
        )
        uncovered = [
            item["subquestion_id"]
            for item in coverage["subquestions"]
            if not item["covered"]
        ]
        trace = state.get("trace", []) + [
            f"Read {len(chunks)} evidence chunks in {len(batches)} batches.",
            f"Extracted {len(batch_claims)} batch claims and consolidated them into "
            f"{len(claims)} global claims.",
            (
                f"Targeted re-extraction for subquestions {retried_subquestions} produced "
                f"{len(additional_claims)} additional candidate claims."
                if retried_subquestions
                else "No subquestions had retained evidence for targeted re-extraction."
            ),
            (
                "All subquestions meet claim coverage."
                if not uncovered
                else f"Subquestions below claim coverage target: {uncovered}."
            ),
        ]
        return {"claims": claims, "claim_coverage": coverage, "trace": trace}

    return node
