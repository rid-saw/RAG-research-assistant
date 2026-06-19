from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse
from app.services.llm import LLMConfigError, generate
from app.services.retriever import get_retriever

router = APIRouter()

ANSWER_PROMPT = """You are a research assistant. Answer the user's question using ONLY the provided context passages. Cite each claim with the marker [{{source}} p.{{page}}] inline.

If the context does not contain enough information to answer, say so explicitly. Do not invent facts.

Context passages:
{context}

Question: {query}

Answer:"""


def _format_context(docs) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Passage {i} | {source} p.{page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    retriever = get_retriever(request.collection)
    docs = retriever.search(request.query, top_k=request.top_k)

    if not docs:
        return ChatResponse(
            query=request.query,
            collection=request.collection,
            answer="No documents in this library yet. Upload some PDFs first.",
            citations=[],
        )

    prompt = ANSWER_PROMPT.format(
        context=_format_context(docs),
        query=request.query,
    )

    try:
        answer = generate(prompt)
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))

    citations = [
        ChatCitation(
            source=doc.metadata.get("source"),
            page=doc.metadata.get("page"),
            chunk_id=doc.metadata.get("chunk_id"),
            snippet=doc.page_content[:200],
        )
        for doc in docs
    ]

    return ChatResponse(
        query=request.query,
        collection=request.collection,
        answer=answer,
        citations=citations,
    )
