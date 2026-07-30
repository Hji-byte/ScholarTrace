from langchain_core.messages import AIMessage

from research_agent.agent.writer import SYSTEM, write_report
from research_agent.schema import EvidenceChunk, VerifiedClaim


class RecordingChatModel:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="# Report")


def test_writer_includes_claim_verification_summary():
    llm = RecordingChatModel()
    node = write_report(llm)
    state = {
        "question": "How can inference be accelerated?",
        "verified_claims": [
            VerifiedClaim(
                claim="Speculative decoding uses a draft model.",
                category="Speculative Decoding",
                evidence_chunk_ids=["chunk-1"],
                supported=True,
            )
        ],
        "evidence_chunks": [
            EvidenceChunk(
                chunk_id="chunk-1",
                paper_id="paper-1",
                title="Speculative Decoding",
                text="Speculative decoding uses a draft model.",
                url="https://example.com/chunk-1",
            ),
            EvidenceChunk(
                chunk_id="chunk-2",
                paper_id="paper-2",
                title="Uncited Evidence",
                text="This chunk was retrieved but not cited.",
                url="https://example.com/chunk-2",
            )
        ],
        "papers": [],
        "supported_claim_count": 1,
        "rejected_claim_count": 3,
        "trace": [],
    }

    result = node(state)

    assert result["report_markdown"] == "# Report"
    assert llm.messages
    human_message = llm.messages[-1]
    assert "Claim verification summary: 1 supported, 3 rejected." in human_message.content
    assert "Supported claim 1:" in human_message.content
    assert "Claim: Speculative decoding uses a draft model." in human_message.content
    assert "Category: Speculative Decoding" in human_message.content
    assert "Evidence chunk ids: chunk-1" in human_message.content
    assert "Known papers:" not in human_message.content
    assert "Reference chunk 1:" not in human_message.content
    assert "Chunk id: chunk-1" in human_message.content
    assert "Title: Speculative Decoding" in human_message.content
    assert "URL: https://example.com/chunk-1" in human_message.content
    assert "chunk-1" in human_message.content
    assert "chunk-2" not in human_message.content


def test_writer_prompt_requires_cited_supported_claims():
    assert "Use only the supported claims" in SYSTEM
    assert "## Scope and Evidence Base" in SYSTEM
    assert "## Key Findings" in SYSTEM
    assert "Every substantive statement" in SYSTEM
    assert "If there are no supported claims" in SYSTEM
