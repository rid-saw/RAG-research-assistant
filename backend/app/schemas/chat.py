from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field("default")
    top_k: int | None = Field(None, ge=1, le=50)
    allow_web_search: bool = Field(False, description="If true, fall back to web when corpus is insufficient")


class ChatCitation(BaseModel):
    source: str | None = None
    title: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    snippet: str


ChatStatus = Literal["answered", "needs_web_search", "answered_from_web", "refused"]


class ChatResponse(BaseModel):
    query: str
    collection: str
    status: ChatStatus
    answer: str
    citations: list[ChatCitation]
    reason: str | None = None
    rewritten_query: str | None = None
