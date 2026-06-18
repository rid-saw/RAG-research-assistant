from pydantic import BaseModel, Field


class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")


class LibrarySummary(BaseModel):
    name: str
    document_count: int


class ListLibrariesResponse(BaseModel):
    libraries: list[LibrarySummary]


class IngestResponse(BaseModel):
    library: str
    filename: str
    pages: int
    chunks_added: int
