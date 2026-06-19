from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field("default")
    top_k: int | None = Field(None, ge=1, le=50)


class ChatCitation(BaseModel):
    source: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    snippet: str


class ChatResponse(BaseModel):
    query: str
    collection: str
    answer: str
    citations: list[ChatCitation]
