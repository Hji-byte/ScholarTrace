from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_agent.agent.verifier import verify_claims
from research_agent.schema import Claim, EvidenceChunk


class RecordingChatModel:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return type(
            "Response",
            (),
            {
                "content": (
                    '{"verified_claims":[{"claim":"Supported claim","category":"General",'
                    '"evidence_chunk_ids":["real"],"supported":true,'
                    '"reason":"The cited chunk states it."}]}'
                )
            },
        )()


def test_verify_claims_parses_llm_verification_response():
    llm = FakeListChatModel(
        responses=[
            '{"verified_claims":[{"claim":"Supported claim","category":"General","evidence_chunk_ids":["real"],"supported":true,"reason":"The cited chunk states it."}]}'
        ]
    )
    node = verify_claims(llm)
    state = {
        "claims": [Claim(claim="Supported claim", evidence_chunk_ids=["real"])],
        "evidence_chunks": [
            EvidenceChunk(chunk_id="real", paper_id="p", title="T", text="Supported claim")
        ],
        "trace": [],
    }

    result = node(state)

    assert result["verified_claims"][0].supported is True
    assert result["supported_claim_count"] == 1
    assert result["rejected_claim_count"] == 0
    assert "Verified claims: 1 accepted, 0 rejected." in result["trace"]


def test_verify_claims_lists_claims_before_evidence():
    llm = RecordingChatModel()
    node = verify_claims(llm)
    state = {
        "claims": [Claim(claim="Supported claim", evidence_chunk_ids=["real"])],
        "evidence_chunks": [
            EvidenceChunk(chunk_id="real", paper_id="p", title="T", text="Supported claim")
        ],
        "trace": [],
    }

    node(state)

    human_message = llm.messages[-1]
    assert human_message.content.index("Claims:") < human_message.content.index("Evidence:")


def test_verify_claims_only_includes_cited_evidence_chunks():
    llm = RecordingChatModel()
    node = verify_claims(llm)
    state = {
        "claims": [Claim(claim="Supported claim", evidence_chunk_ids=["real"])],
        "evidence_chunks": [
            EvidenceChunk(chunk_id="real", paper_id="p", title="T", text="Supported claim"),
            EvidenceChunk(chunk_id="unused", paper_id="p", title="T", text="Unused evidence"),
        ],
        "trace": [],
    }

    node(state)

    human_message = llm.messages[-1]
    assert "Chunk id: real" in human_message.content
    assert "Chunk id: unused" not in human_message.content
