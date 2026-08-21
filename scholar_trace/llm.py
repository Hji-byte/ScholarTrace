from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel

from scholar_trace.config import Settings


DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def dashscope_base_url(settings: Settings) -> str:
    return settings.dashscope_base_url or DASHSCOPE_DEFAULT_BASE_URL


def build_chat_model(settings: Settings) -> BaseChatModel:
    if not settings.dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and fill it in."
        )
    return init_chat_model(
        model=settings.qwen_chat_model,
        model_provider="openai",
        base_url=dashscope_base_url(settings),
        api_key=settings.dashscope_api_key,
    )


def build_embeddings(settings: Settings) -> DashScopeEmbeddings:
    if not settings.dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and fill it in."
        )
    return DashScopeEmbeddings(
        model=settings.qwen_embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
    )
