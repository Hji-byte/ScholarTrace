from research_agent.config import Settings
from research_agent.llm import build_chat_model, build_embeddings, dashscope_base_url


def test_dashscope_base_url_prefers_new_setting():
    settings = Settings(
        _env_file=None,
        dashscope_base_url="https://new.example/v1",
    )

    assert dashscope_base_url(settings) == "https://new.example/v1"


def test_dashscope_base_url_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert dashscope_base_url(settings) == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_build_chat_model_uses_init_chat_model(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return "chat-model"

    monkeypatch.setattr("research_agent.llm.init_chat_model", fake_init_chat_model)

    settings = Settings(
        _env_file=None,
        dashscope_api_key="test-key",
        dashscope_base_url="https://dashscope.example/v1",
        qwen_chat_model="qwen3.7-plus",
    )

    model = build_chat_model(settings)

    assert model == "chat-model"
    assert captured == {
        "model": "qwen3.7-plus",
        "model_provider": "openai",
        "base_url": "https://dashscope.example/v1",
        "api_key": "test-key",
        "temperature": 0.2,
    }


def test_build_embeddings_uses_dashscope_embeddings(monkeypatch):
    captured = {}

    class FakeDashScopeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("research_agent.llm.DashScopeEmbeddings", FakeDashScopeEmbeddings)

    settings = Settings(
        _env_file=None,
        dashscope_api_key="test-key",
        qwen_embedding_model="qwen3.7-text-embedding",
    )

    embeddings = build_embeddings(settings)

    assert isinstance(embeddings, FakeDashScopeEmbeddings)
    assert captured == {
        "model": "qwen3.7-text-embedding",
        "dashscope_api_key": "test-key",
    }
