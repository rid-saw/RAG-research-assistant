import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse
from app.services.llm import LLMConfigError
from app.services.orchestrator import run as run_pipeline
from app.services.orchestrator import run_stream as run_pipeline_stream
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


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream of the corrective RAG pipeline.

    Emits, in order:
      - one or more {"event":"stage","stage":...} events
      - many {"event":"token","text":...} events (during generation)
      - one terminal {"event":"done", status, citations, reason, rewritten_query}
      - on config errors: one {"event":"error","detail":...}
    """

    def generator():
        try:
            for evt in run_pipeline_stream(
                query=request.query,
                collection=request.collection,
                allow_web_search=request.allow_web_search,
                top_k=request.top_k,
            ):
                yield _sse(evt)
        except LLMConfigError as e:
            yield _sse({"event": "error", "detail": str(e)})
        except WebSearchConfigError as e:
            yield _sse({"event": "error", "detail": str(e)})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )
