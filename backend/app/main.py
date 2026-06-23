from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, libraries, search
from app.core.chroma import get_chroma_client
from app.core.config import settings

# Pre-warm the shared chromadb client at import time so the first
# concurrent batch of requests doesn't race on its lazy creation.
get_chroma_client()

app = FastAPI(title="Universal RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:4173",  # Vite preview
        "http://localhost:3000",  # alt React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(libraries.router, prefix="/api", tags=["libraries"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }
