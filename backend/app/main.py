from fastapi import FastAPI

from app.api import libraries, search
from app.core.config import settings

app = FastAPI(title="Universal RAG", version="0.1.0")

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(libraries.router, prefix="/api", tags=["libraries"])


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }
