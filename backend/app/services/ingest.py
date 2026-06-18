import uuid
from pathlib import Path

import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import settings
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
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    out: list[tuple[str, int]] = []
    for col in client.list_collections():
        out.append((col.name, col.count()))
    return out
