from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, max_length=512)
