from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agent.llm_retry import invoke_with_retry
from research_agent.tools.json_utils import extract_json_object
from research_agent.schema import Claim, ResearchState


SYSTEM = """You are the evidence reader in a computer science literature-review agent.
Your job is to extract precise, evidence-backed claims from the provided chunks.

Return strict JSON only, with this exact shape:
{"claims":[{"claim":"...", "category":"...", "evidence_chunk_ids":["..."]}]}

Field meanings:
- claim: one atomic statement that is directly supported by the cited chunks.
- category: a short topic label used later to group claims in the report.
- evidence_chunk_ids: the chunk ids that directly support the claim.

Rules:
- Extract 5 to 12 claims that help answer the research question.
- Each claim must be directly supported by one or more provided chunks.
- Use only chunk ids that appear in the evidence context.
- Make each claim atomic: one specific method, result, limitation, comparison, or trend per claim.
- Prefer claims about mechanisms, method categories, evaluation findings, tradeoffs, limitations, and open problems.
- Do not invent citations, paper titles, benchmark results, dates, or numeric values that are not in the chunks.
- Do not make broad field-level claims unless the chunks explicitly support them.
- If multiple chunks support the same idea, include all relevant chunk ids.
- If the evidence is thin, return fewer claims rather than padding.
- Choose concise, evidence-driven categories that would make good literature-review subsections for the current research question."""


def parse_claims(text: str) -> list[Claim]:
    data = extract_json_object(text)
    return [Claim.model_validate(item) for item in data.get("claims", [])]


def chunk_batches(chunks: list, batch_size: int):
    size = max(1, batch_size)
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def dedupe_claims(claims: list[Claim]) -> list[Claim]:
    deduped: list[Claim] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for claim in claims:
        key = (
            " ".join(claim.claim.lower().split()),
            tuple(sorted(claim.evidence_chunk_ids)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def read_evidence(llm: BaseChatModel, batch_size: int = 10):
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
                            f"Evidence batch {index} of {len(batches)}:\n{context}"
                        )
                    ),
                ]
            )
            claims.extend(parse_claims(str(response.content)))
        claims = dedupe_claims(claims)
        trace = state.get("trace", []) + [
            f"Read {len(chunks)} evidence chunks in {len(batches)} batches.",
            f"Extracted {len(claims)} candidate claims.",
        ]
        return {"claims": claims, "trace": trace}

    return node
