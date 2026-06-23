import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.chroma import get_chroma_client
from app.services.retriever import get_retriever


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )


def ingest_pdf(library: str, pdf_path: Path, filename: str) -> tuple[int, int]:
    pages = PyPDFLoader(str(pdf_path)).load()
    chunks = _splitter().split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    retriever = get_retriever(library)
    retriever.add_documents(chunks)

    return len(pages), len(chunks)


def list_libraries() -> list[tuple[str, int]]:
    client = get_chroma_client()
    out: list[tuple[str, int]] = []
    for col in client.list_collections():
        out.append((col.name, col.count()))
    return out


def list_library_files(library: str) -> list[tuple[str, int]]:
    """Return [(filename, chunk_count), ...] for the given library."""
    client = get_chroma_client()
    try:
        col = client.get_collection(name=library)
    except Exception:
        return []
    data = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data.get("metadatas", []) or []:
        src = (meta or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return sorted(counts.items(), key=lambda x: x[0].lower())


def delete_library(library: str) -> bool:
    """Drop the entire chroma collection. Returns True if removed."""
    from app.services.retriever import invalidate_retriever

    client = get_chroma_client()
    try:
        client.delete_collection(name=library)
    except Exception:
        invalidate_retriever(library)
        return False
    invalidate_retriever(library)
    return True


def delete_file(library: str, filename: str) -> int:
    """Delete every chunk in `library` whose source metadata matches filename.

    Returns the number of chunks deleted.
    """
    from app.services.retriever import invalidate_retriever

    client = get_chroma_client()
    try:
        col = client.get_collection(name=library)
    except Exception:
        return 0

    matches = col.get(where={"source": filename}, include=[])
    ids = matches.get("ids", []) or []
    if not ids:
        return 0

    col.delete(ids=ids)
    invalidate_retriever(library)
    return len(ids)
