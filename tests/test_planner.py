from research_agent.agent.planner import SYSTEM, parse_plan


def test_planner_prompt_is_domain_neutral():
    prompt = SYSTEM.lower()

    assert "derive subquestions and search intents only from" in prompt
    assert "create 3 to 5 focused subquestions" in prompt
    assert "create 3 to 5 search intents" in prompt
    assert "substantively different evidence need implied by the question" in prompt
    assert "solely because it is common in literature reviews" in prompt
    assert "minimum scope information" in prompt
    assert "different task, domain, modality, or research object" in prompt
    assert "not stated or clearly implied by the question" in prompt
    for benchmark_specific_term in (
        "serverless computing",
        "malicious package",
        "cold start",
        "insertions/deletions",
    ):
        assert benchmark_specific_term not in prompt

    for provider_term in ("provider", "arxiv", "semantic scholar", "adapter"):
        assert provider_term not in prompt


def test_parse_plan_from_json_block():
    plan = parse_plan(
        """```json
        {"subquestions":["a","b"],"search_intents":[
          {"purpose":"methods","must_groups":[["serverless computing","function as a service"],["cold start","startup latency"]]},
          {"purpose":"mitigations","must_groups":[["serverless computing"],["prewarming","snapshotting"]]}
        ]}
        ```"""
    )
    assert plan.subquestions == ["a", "b"]
    assert len(plan.search_intents) == 2
    assert plan.search_intents[0].must_groups[0] == [
        "serverless computing",
        "function as a service",
    ]
