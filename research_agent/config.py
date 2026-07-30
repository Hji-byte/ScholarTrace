from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    dashscope_api_key: str = ""
    dashscope_base_url: str = ""
    qwen_chat_model: str = "qwen3.7-plus"
    qwen_embedding_model: str = "qwen3.7-text-embedding"
    paper_ranking_strategy: Literal["rrf", "qwen3_rerank"] = "rrf"
    dashscope_workspace_id: str = ""
    qwen_rerank_base_url: str = ""
    qwen_rerank_model: str = "qwen3-rerank"
    qwen_rerank_timeout_seconds: float = 60.0
    search_results_per_query: int = 15
    max_papers: int = 10
    pdf_candidate_limit: int = 0
    max_chunks_per_query: int = 12
    max_evidence_chunks: int = 30
    reader_batch_size: int = 10
    verifier_batch_size: int = 8
    use_pdf: bool = False
    pdf_failure_policy: str = "fallback_abstract"
    min_pdf_papers: int = 0
    arxiv_delay_seconds: float = 3.0
    experiment_id: str = ""
    chroma_persist_dir: Path = Path("data/chroma")
    sqlite_db_path: Path = Path("data/research_agent.db")
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    output_dir: Path = Path("outputs/reports")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
