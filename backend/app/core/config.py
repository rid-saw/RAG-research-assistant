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

    # LLM provider — any OpenAI-compatible API: "openrouter" | "groq" | "gemini" | "openai".
    llm_provider: str = "openrouter"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 1024

    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None

    tavily_api_key: str | None = None
    web_search_results: int = 5


settings = Settings()
