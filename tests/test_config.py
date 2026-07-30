from research_agent.config import Settings


def test_use_pdf_setting_can_be_enabled_directly():
    settings = Settings(use_pdf=True)

    assert settings.use_pdf is True


def test_use_pdf_setting_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("USE_PDF", "true")

    settings = Settings()

    assert settings.use_pdf is True


def test_pdf_experiment_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("SEARCH_RESULTS_PER_QUERY", "12")
    monkeypatch.setenv("PDF_FAILURE_POLICY", "skip")
    monkeypatch.setenv("PDF_CANDIDATE_LIMIT", "15")
    monkeypatch.setenv("MIN_PDF_PAPERS", "6")

    settings = Settings()

    assert settings.search_results_per_query == 12
    assert settings.pdf_failure_policy == "skip"
    assert settings.pdf_candidate_limit == 15
    assert settings.min_pdf_papers == 6


def test_reranker_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("PAPER_RANKING_STRATEGY", "qwen3_rerank")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-example")
    monkeypatch.setenv("QWEN_RERANK_MODEL", "qwen3-rerank")

    settings = Settings()

    assert settings.paper_ranking_strategy == "qwen3_rerank"
    assert settings.dashscope_workspace_id == "ws-example"
    assert settings.qwen_rerank_model == "qwen3-rerank"
