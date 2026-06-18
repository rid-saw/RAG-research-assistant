import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.library import (
    CreateLibraryRequest,
    IngestResponse,
    LibrarySummary,
    ListLibrariesResponse,
)
from app.services.ingest import ingest_pdf, list_libraries
from app.services.retriever import get_retriever

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/libraries", response_model=LibrarySummary, status_code=201)
def create_library(request: CreateLibraryRequest) -> LibrarySummary:
    retriever = get_retriever(request.name)
    return LibrarySummary(name=request.name, document_count=len(retriever._documents))


@router.get("/libraries", response_model=ListLibrariesResponse)
def get_libraries() -> ListLibrariesResponse:
    libs = list_libraries()
    return ListLibrariesResponse(
        libraries=[LibrarySummary(name=n, document_count=c) for n, c in libs]
    )


@router.post(
    "/libraries/{name}/documents",
    response_model=IngestResponse,
    status_code=201,
)
async def upload_document(name: str, file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds 10MB limit")
            tmp.write(chunk)

    try:
        pages, chunks_added = ingest_pdf(name, tmp_path, file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        library=name,
        filename=file.filename,
        pages=pages,
        chunks_added=chunks_added,
    )
