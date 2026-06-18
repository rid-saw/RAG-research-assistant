from fastapi import APIRouter

from app.schemas.search import SearchHit, SearchRequest, SearchResponse
from app.services.retriever import get_retriever

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    retriever = get_retriever(request.collection)
    docs = retriever.search(request.query, top_k=request.top_k)

    hits = [
        SearchHit(
            content=doc.page_content,
            source=doc.metadata.get("source"),
            page=doc.metadata.get("page"),
            chunk_id=doc.metadata.get("chunk_id"),
        )
        for doc in docs
    ]

    return SearchResponse(
        query=request.query,
        collection=request.collection,
        hits=hits,
    )
