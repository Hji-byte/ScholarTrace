from research_agent.agent.reader import parse_claims
from research_agent.agent.reader import read_evidence
from research_agent.schema import EvidenceChunk


class RecordingChatModel:
    def __init__(self):
        self.messages = []
        self.calls = []

    def invoke(self, messages):
        self.messages = messages
        self.calls.append(messages)
        return type("Response", (), {"content": '{"claims":[]}'})()


def test_parse_claims_from_json_response():
    claims = parse_claims(
        """
        {"claims":[
          {
            "claim":"Speculative decoding uses a draft model to reduce latency.",
            "category":"Speculative Decoding",
            "evidence_chunk_ids":["chunk-1"]
          }
        ]}
        """
    )

    assert len(claims) == 1
    assert claims[0].category == "Speculative Decoding"
    assert claims[0].evidence_chunk_ids == ["chunk-1"]


def test_reader_context_labels_chunk_id_explicitly():
    llm = RecordingChatModel()
    node = read_evidence(llm)

    node(
        {
            "question": "How can inference be accelerated?",
            "evidence_chunks": [
                EvidenceChunk(
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    title="Speculative Decoding",
                    text="Uses a draft model.",
                )
            ],
            "trace": [],
        }
    )

    human_message = llm.messages[-1]
    assert "Chunk id: chunk-1" in human_message.content
    assert "Title: Speculative Decoding" in human_message.content


def test_reader_batches_evidence_chunks():
    llm = RecordingChatModel()
    node = read_evidence(llm, batch_size=2)

    node(
        {
            "question": "How can inference be accelerated?",
            "evidence_chunks": [
                EvidenceChunk(
                    chunk_id=f"chunk-{index}",
                    paper_id="paper-1",
                    title="Speculative Decoding",
                    text=f"Text {index}.",
                )
                for index in range(5)
            ],
            "trace": [],
        }
    )

    assert len(llm.calls) == 3
    assert "Evidence batch 1 of 3" in llm.calls[0][-1].content
    assert "Chunk id: chunk-0" in llm.calls[0][-1].content
    assert "Chunk id: chunk-2" in llm.calls[1][-1].content
    assert "Chunk id: chunk-4" in llm.calls[2][-1].content
