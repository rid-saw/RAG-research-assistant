from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse
from app.services.llm import LLMConfigError
from app.services.orchestrator import run as run_pipeline
from app.services.web_search import WebSearchConfigError

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_pipeline(
            query=request.query,
            collection=request.collection,
            allow_web_search=request.allow_web_search,
            top_k=request.top_k,
        )
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except WebSearchConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))

    citations = [
        ChatCitation(
            source=doc.metadata.get("source"),
            title=doc.metadata.get("title"),
            page=doc.metadata.get("page"),
            chunk_id=doc.metadata.get("chunk_id"),
            snippet=doc.page_content[:200],
        )
        for doc in result["citations"]
    ]

    return ChatResponse(
        query=request.query,
        collection=request.collection,
        status=result["status"],
        answer=result["answer"],
        citations=citations,
        reason=result["reason"],
        rewritten_query=result["rewritten_query"],
    )
