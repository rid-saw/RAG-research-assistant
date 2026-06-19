from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"

    reranker_model: str = "BAAI/bge-reranker-base"

    chroma_persist_dir: str = "./data/chroma"

    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    candidates_per_search: int = 50
    rerank_top_k: int = 8

    google_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 1024


settings = Settings()
