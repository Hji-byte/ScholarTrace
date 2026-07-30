import json
import re
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_agent.config import Settings
from research_agent.paper_evaluation import (
    evaluate_ranked_papers,
    normalize_source_id,
    normalize_title,
    run_paper_evaluation,
    select_questions_by_id,
)
from research_agent.schema import Paper
from research_agent.tools.arxiv_search import arxiv_source_id


def test_v2_questions_move_years_out_of_natural_language():
    root = Path(__file__).resolve().parents[1]
    v1 = [json.loads(line) for line in (root / "evaluation/datasets/cs_questions_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    v2 = [json.loads(line) for line in (root / "evaluation/datasets/cs_questions_v2.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(v1) == len(v2) == 30
    assert [record["question_id"] for record in v1] == [record["question_id"] for record in v2]
    assert all(not re.search(r"\b(?:19|20)\d{2}\b", record["question"]) for record in v2)
    for old, new in zip(v1, v2, strict=True):
        assert old["year_to"] == new["year_to"]
        assert old["domain"] == new["domain"]
        assert old["topic"] == new["topic"]


def test_select_questions_by_id_preserves_dataset_order_and_rejects_unknown_ids():
    questions = [
        {"question_id": "csq001"},
        {"question_id": "csq002"},
        {"question_id": "csq003"},
    ]

    selected = select_questions_by_id(questions, ["csq003", "csq001", "csq003"])

    assert [question["question_id"] for question in selected] == ["csq001", "csq003"]
    try:
        select_questions_by_id(questions, ["csq999"])
    except ValueError as exc:
        assert str(exc) == "Unknown question ids: csq999"
    else:
        raise AssertionError("unknown question IDs should be rejected")


def test_arxiv_gold_v2_is_unique_complete_and_within_question_ranges():
    root = Path(__file__).resolve().parents[1]
    questions = {
        record["question_id"]: record
        for record in (
            json.loads(line)
            for line in (root / "evaluation/datasets/cs_questions_v2.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    gold_records = [
        json.loads(line)
        for line in (root / "evaluation/datasets/key_papers_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    source_ids = []
    for record in gold_records:
        question = questions[record["question_id"]]
        assert len(record["key_papers"]) == 4
        assert sum(bool(paper["required"]) for paper in record["key_papers"]) == 3
        for paper in record["key_papers"]:
            assert re.fullmatch(r"arXiv:\d{4}\.\d+", paper["source_id"])
            assert question["year_from"] <= paper["year"] <= question["year_to"]
            source_ids.append(paper["source_id"])

    assert len(gold_records) == 30
    assert len(source_ids) == len(set(source_ids)) == 120
    sequencer = next(
        paper
        for record in gold_records
        for paper in record["key_papers"]
        if paper["source_id"] == "arXiv:1901.01808"
    )
    assert sequencer["year"] == 2018


def test_arxiv_source_id_is_stable_across_url_forms_and_versions():
    assert arxiv_source_id("https://arxiv.org/abs/2309.15217v2") == "arXiv:2309.15217"
    assert arxiv_source_id("https://arxiv.org/pdf/2309.15217.pdf") == "arXiv:2309.15217"
    assert normalize_source_id(" arXiv:2309.15217v3 ") == "arxiv:2309.15217"


def test_normalized_title_ignores_case_punctuation_and_accents():
    assert normalize_title("FActScore: Fine-grained Évaluation") == normalize_title(
        "factscore — fine grained evaluation"
    )


def test_evaluate_ranked_papers_computes_strict_broad_recall_and_mrr():
    candidates = [
        Paper(
            paper_id="supplemental",
            source_id="arXiv:1111.11111",
            title="Supplemental Paper",
            rank=1,
        ),
        Paper(
            paper_id="required-hit",
            source_id="arXiv:2222.22222",
            title="Required Hit",
            rank=2,
        ),
    ]
    gold = [
        {"source_id": "arXiv:2222.22222", "title": "Required Hit", "required": True},
        {"source_id": "arXiv:3333.33333", "title": "Missing", "required": True},
        {"source_id": "arXiv:1111.11111", "title": "Supplemental Paper", "required": False},
    ]

    metrics, matches = evaluate_ranked_papers(candidates, gold)

    assert metrics["strict_recall@5"] == 0.5
    assert metrics["broad_recall@5"] == 2 / 3
    assert metrics["mrr"] == 0.5
    assert matches[0]["match_method"] == "source_id"
    assert matches[1]["matched"] is False


def test_paper_evaluation_runner_writes_results_and_summary(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "csq001",
                "domain": "nlp",
                "topic": "rag",
                "question_type": "survey",
                "difficulty": "medium",
                "year_from": 2020,
                "year_to": 2025,
                "question": "How is RAG evaluated?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(
            {
                "question_id": "csq001",
                "key_papers": [
                    {
                        "source_id": "arXiv:2309.15217",
                        "title": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
                        "required": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "research_agent.paper_evaluation.build_chat_model",
        lambda settings: FakeListChatModel(
            responses=[
                '{"subquestions":["Which RAG evaluation metrics are used?"],"search_intents":['
                '{"purpose":"RAG evaluation","must_groups":[["RAG"],["evaluation"]]}'
                ']}'
            ]
        ),
    )

    def fake_search(
        provider_query: str,
        max_results: int = 10,
    ):
        assert provider_query == (
            '(all:"RAG") AND (all:"evaluation") AND '
            'submittedDate:[202001010000 TO 202512312359]'
        )
        assert max_results == 15
        return [
            Paper(
                paper_id="ragas",
                source_id="arXiv:2309.15217",
                title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
                year=2023,
            )
        ] + [
            Paper(
                paper_id=f"candidate-{index}",
                source_id=f"arXiv:2301.{index:05d}",
                title=f"Candidate Paper {index}",
                year=2023,
            )
            for index in range(1, 20)
        ]

    monkeypatch.setattr("research_agent.agent.search.search_arxiv", fake_search)
    settings = Settings(
        dashscope_api_key="test",
        sqlite_db_path=tmp_path / "runs.db",
        experiment_id="test eval",
    )

    results_path, raw_search_path, summary_path = run_paper_evaluation(
        settings,
        questions_path=questions_path,
        gold_path=gold_path,
        output_dir=tmp_path / "results",
    )

    result = json.loads(results_path.read_text(encoding="utf-8"))
    raw_result = json.loads(raw_search_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["metrics"]["strict_recall@20"] == 1.0
    assert result["papers"][0]["rrf_score"] is not None
    assert raw_result["question_id"] == "csq001"
    assert raw_result["provider_query"] == (
        '(all:"RAG") AND (all:"evaluation") AND '
        'submittedDate:[202001010000 TO 202512312359]'
    )
    assert raw_result["search_intent"]["purpose"] == "RAG evaluation"
    assert len(raw_result["papers"]) == 20
    assert raw_result["papers"][0]["source_rank"] == 1
    assert "rank" not in raw_result["papers"][0]
    assert "rrf_score" not in raw_result["papers"][0]
    assert summary["successful_count"] == 1
    assert summary["raw_search_path"] == str(raw_search_path)
    assert result["plan"]["subquestions"] == [
        "Which RAG evaluation metrics are used?"
    ]
    assert result["plan"]["search_intents"] == result["search_intents"]
    assert result["search_intents"][0]["purpose"] == "RAG evaluation"
    assert summary["configuration"]["year_filter"] == (
        "arxiv_submitted_date_query_with_local_audit"
    )
    assert summary["configuration"]["year_semantics"] == "arxiv_first_submission_year"
    assert summary["configuration"]["rrf_k"] == 60
    assert summary["configuration"]["arxiv_sort_by"] == "relevance"
    assert summary["configuration"]["arxiv_sort_order"] == "descending"
    assert summary["configuration"]["arxiv_timeout_seconds"] == 20
    assert summary["configuration"]["arxiv_max_retries"] == 3
