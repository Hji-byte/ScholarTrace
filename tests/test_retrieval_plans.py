import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_agent.config import Settings
from research_agent.retrieval_plans import generate_retrieval_plans


def test_generate_retrieval_plans_writes_full_plan_and_resumes(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "retrieval_plans.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "csq001",
                "domain": "nlp",
                "topic": "rag",
                "question_type": "survey",
                "difficulty": "medium",
                "question": "How is RAG evaluated?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = (
        '{"subquestions":["Which retrieval metrics are used?"],'
        '"search_intents":[{"purpose":"RAG evaluation",'
        '"must_groups":[["RAG"],["evaluation"]]}]}'
    )
    settings = Settings(dashscope_api_key="test", qwen_chat_model="test-planner")

    generate_retrieval_plans(
        settings,
        questions_path,
        output_path,
        llm=FakeListChatModel(responses=[response]),
    )
    first_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_record["status"] == "ok"
    assert first_record["plan"]["subquestions"] == [
        "Which retrieval metrics are used?"
    ]
    assert first_record["plan"]["search_intents"][0]["purpose"] == "RAG evaluation"
    assert first_record["planner_model"] == "test-planner"

    generate_retrieval_plans(
        settings,
        questions_path,
        output_path,
        llm=FakeListChatModel(responses=[]),
    )
    resumed_records = output_path.read_text(encoding="utf-8").splitlines()
    assert len(resumed_records) == 1


def test_generate_retrieval_plans_preserves_other_records_in_selected_runs(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "retrieval_plans.jsonl"
    questions_path.write_text(
        "\n".join(
            json.dumps({"question_id": question_id, "question": question})
            for question_id, question in [
                ("csq001", "Question one?"),
                ("csq002", "Question two?"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = [
        '{"subquestions":["Subquestion one?"],"search_intents":['
        '{"purpose":"one","must_groups":[["one"]]}]}',
        '{"subquestions":["Subquestion two?"],"search_intents":['
        '{"purpose":"two","must_groups":[["two"]]}]}',
    ]
    settings = Settings(dashscope_api_key="test")

    for question_id, response in zip(["csq001", "csq002"], responses, strict=True):
        generate_retrieval_plans(
            settings,
            questions_path,
            output_path,
            question_ids=[question_id],
            llm=FakeListChatModel(responses=[response]),
        )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["question_id"] for record in records] == ["csq001", "csq002"]
