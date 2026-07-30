from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agent.llm_retry import invoke_with_retry
from research_agent.schema import ResearchState


SYSTEM = """You are the report writer for a computer science literature-review agent.
Write a concise, evidence-grounded Markdown literature review that answers the research question.

Use only the supported claims and reference chunks provided by the user message.
Do not use rejected claims, unsupported claims, outside knowledge, or uncited background facts.

Required structure:
# Literature Review
## Summary
## Scope and Evidence Base
## Key Findings
## Method Categories
## Evidence-backed Discussion
## Limitations and Open Problems
## References

Writing rules:
- Every substantive statement about a method, result, trend, limitation, or comparison must cite at least one chunk id in square brackets, such as [chunk-1].
- Use Scope and Evidence Base to describe the retrieval scope, number of supported claims, and any evidence limitations.
- Use Key Findings for the most important evidence-backed takeaways.
- Organize the Method Categories section around the claim categories when possible.
- Synthesize across claims instead of listing them mechanically, but do not merge claims in a way that creates a broader unsupported statement.
- Keep the tone careful and academic.
- If the supported-claim count is low, write a shorter report and explicitly say that the evidence base is limited.
- If there are no supported claims, do not fabricate a review; say that the retrieved evidence was insufficient and provide only a brief References section if references were provided.
- Do not mention the internal reader or verifier agents.
- The References section must list only chunks provided in Reference chunks, preserving their chunk ids."""


def write_report(llm: BaseChatModel):
    def node(state: ResearchState) -> ResearchState:
        supported = [claim for claim in state.get("verified_claims", []) if claim.supported]
        cited_chunk_ids = {
            chunk_id
            for claim in supported
            for chunk_id in claim.evidence_chunk_ids
        }
        chunks = {
            chunk.chunk_id: chunk
            for chunk in state.get("evidence_chunks", [])
            if chunk.chunk_id in cited_chunk_ids
        }
        claims_text = "\n\n".join(
            (
                f"Supported claim {index}:\n"
                f"Claim: {claim.claim}\n"
                f"Category: {claim.category}\n"
                f"Evidence chunk ids: {', '.join(claim.evidence_chunk_ids)}"
            )
            for index, claim in enumerate(supported, start=1)
        )
        refs = "\n\n".join(
            (
                f"Chunk id: {chunk.chunk_id}\n"
                f"Title: {chunk.title}\n"
                f"URL: {chunk.url}"
            )
            for chunk in chunks.values()
        )
        supported_count = state.get("supported_claim_count", len(supported))
        rejected_count = state.get("rejected_claim_count", 0)
        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"Claim verification summary: {supported_count} supported, "
                        f"{rejected_count} rejected.\n\n"
                        f"Supported claims:\n{claims_text}\n\n"
                        f"Reference chunks:\n{refs}"
                    )
                ),
            ]
        )
        trace = state.get("trace", []) + ["Generated final Markdown report."]
        return {"report_markdown": str(response.content), "trace": trace}

    return node
