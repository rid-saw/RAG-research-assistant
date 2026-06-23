from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT_ENV), extra="ignore")

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"

    reranker_model: str = "BAAI/bge-reranker-base"

    chroma_persist_dir: str = "./data/chroma"

    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    candidates_per_search: int = 50
    rerank_top_k: int = 8

    # LLM provider switch — "groq" or "gemini" (both via OpenAI-compatible API).
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 1024

    groq_api_key: str | None = None
    google_api_key: str | None = None

    tavily_api_key: str | None = None
    web_search_results: int = 5


settings = Settings()
