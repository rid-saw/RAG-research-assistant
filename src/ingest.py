# src/ingest.py
"""
Document ingestion and chunking for RAG system.
Loads PDFs, splits them into chunks, and prepares them for embedding.
"""

import os
import json
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_paper_titles(arxiv_json_path: str = "arxiv_papers.json") -> Dict[str, str]:
    """
    Load paper titles from arxiv_papers.json.

    Args:
        arxiv_json_path: Path to arxiv_papers.json

    Returns:
        Dictionary mapping arxiv_id to paper title
    """
    try:
        with open(arxiv_json_path, 'r') as f:
            data = json.load(f)

        # Create mapping: arxiv_id -> title
        title_map = {paper["arxiv_id"]: paper["title"] for paper in data["papers"]}
        return title_map
    except Exception as e:
        print(f"Warning: Could not load paper titles from {arxiv_json_path}: {e}")
        return {}


def load_pdfs(data_folder: str = "data") -> List[Document]:
    """
    Load all PDFs from the data folder.

    Args:
        data_folder: Path to folder containing PDFs

    Returns:
        List of Document objects with page content and metadata
    """
    documents = []

    # Load paper titles
    arxiv_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arxiv_papers.json")
    title_map = load_paper_titles(arxiv_json_path)

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_folder}")

    print(f"Found {len(pdf_files)} PDF(s) to process...")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"  Loading: {pdf_file}")

        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()

            # Extract arxiv_id from filename (e.g., "1706.03762.pdf" -> "1706.03762")
            arxiv_id = pdf_file.replace(".pdf", "")
            paper_title = title_map.get(arxiv_id, "Unknown Paper")

            # Add filename and title to metadata for citation
            for doc in pdf_docs:
                doc.metadata["source_file"] = pdf_file
                doc.metadata["paper_title"] = paper_title

            documents.extend(pdf_docs)
            print(f"    → Loaded {len(pdf_docs)} pages")

        except Exception as e:
            print(f"    → Error loading {pdf_file}: {e}")

    print(f"Total pages loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between chunks to maintain context
        
    Returns:
        List of chunked Document objects
    """
    print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def process_documents(data_folder: str = "data") -> List[Document]:
    """
    Main function to load and process all documents.
    
    Args:
        data_folder: Path to folder containing PDFs
        
    Returns:
        List of processed and chunked Document objects
    """
    # Load PDFs
    documents = load_pdfs(data_folder)
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    return chunks


# For testing
if __name__ == "__main__":
    chunks = process_documents()
    
    print("\n--- Sample Chunk ---")
    print(f"Content: {chunks[0].page_content[:300]}...")
    print(f"Metadata: {chunks[0].metadata}")