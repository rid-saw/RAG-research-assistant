from functools import lru_cache

from langchain_core.documents import Document
from tavily import TavilyClient

from app.core.config import settings


class WebSearchConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    if not settings.tavily_api_key:
        raise WebSearchConfigError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=settings.tavily_api_key)


def search_web(query: str) -> list[Document]:
    response = _client().search(query, max_results=settings.web_search_results)
    docs: list[Document] = []
    for r in response.get("results", []):
        docs.append(
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("url", ""),
                    "title": r.get("title", ""),
                    "page": None,
                    "chunk_id": None,
                },
            )
        )
    return docs
