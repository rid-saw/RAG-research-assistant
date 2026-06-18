from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's search query")
    collection: str = Field("default", description="Which library to search")
    top_k: int | None = Field(None, ge=1, le=50, description="Override default result count")


class SearchHit(BaseModel):
    content: str
    source: str | None = None
    page: int | None = None
    chunk_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    collection: str
    hits: list[SearchHit]
