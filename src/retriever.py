# src/retriever.py
"""
Retrieval system with hybrid search (semantic + keyword) and CrossEncoder reranking.
Creates vector store and implements search functionality.
"""

import os
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np


class HybridRetriever:
    """
    Hybrid retriever combining semantic search (embeddings) and 
    keyword search (BM25) for better retrieval performance.
    """
    
    def __init__(
        self,
        chunks: List[Document],
        embedding_model: str = "models/fine-tuned-embeddings",
        persist_directory: str = "chroma_db",
        use_fine_tuned: bool = True,
        fine_tuned_path: str = "models/fine-tuned-embeddings",
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the hybrid retriever with optional reranking.

        Args:
            chunks: List of document chunks
            embedding_model: HuggingFace model for embeddings
            persist_directory: Where to store the vector database
            use_fine_tuned: Whether to use fine-tuned embeddings
            fine_tuned_path: Path to fine-tuned model
            use_reranker: Whether to use CrossEncoder reranking
            reranker_model: CrossEncoder model for reranking
        """
        self.chunks = chunks
        self.persist_directory = persist_directory
        self.use_reranker = use_reranker

        # Check if fine-tuned model exists
        if use_fine_tuned and os.path.exists(fine_tuned_path):
            print(f"\nUsing FINE-TUNED embeddings from {fine_tuned_path}")
            model_path = fine_tuned_path
        else:
            print(f"\nUsing base embeddings: {embedding_model}")
            model_path = embedding_model

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Build vector store (semantic search)
        self._build_vector_store()

        # Build BM25 index (keyword search)
        self._build_bm25_index()

        # Initialize CrossEncoder reranker
        if use_reranker:
            print(f"Loading reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model, max_length=512)
            print("Reranker ready")
        else:
            self.reranker = None
    
    def _build_vector_store(self):
        """Build or load the Chroma vector store."""
        print("Building vector store...")

        if not self.chunks:
            raise ValueError("Cannot build vector store: no document chunks provided")

        # Filter out empty chunks
        self.valid_chunks = [chunk for chunk in self.chunks if chunk.page_content.strip()]

        if not self.valid_chunks:
            raise ValueError(f"All {len(self.chunks)} chunks have empty content")

        print(f"Processing {len(self.valid_chunks)} valid chunks (filtered from {len(self.chunks)} total)")

        self.vector_store = Chroma.from_documents(
            documents=self.valid_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"Vector store ready with {len(self.valid_chunks)} chunks")
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search."""
        print("Building BM25 index...")

        # Use valid chunks (already filtered)
        chunks_to_use = getattr(self, 'valid_chunks', self.chunks)

        # Tokenize documents for BM25
        tokenized_docs = [
            chunk.page_content.lower().split()
            for chunk in chunks_to_use
        ]

        self.bm25 = BM25Okapi(tokenized_docs)
        print("BM25 index ready")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search using embeddings (semantic similarity).
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search using BM25 (keyword matching).

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Use valid chunks (already filtered)
        chunks_to_use = getattr(self, 'valid_chunks', self.chunks)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = [
            (chunks_to_use[i], scores[i])
            for i in top_indices
        ]

        return results
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Document]:
        """
        Combine semantic and keyword search for better results.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search scores
            keyword_weight: Weight for keyword search scores
            
        Returns:
            List of top k documents
        """
        # Get more results from each method, then combine
        semantic_results = self.semantic_search(query, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Normalize and combine scores
        doc_scores = {}
        
        # Process semantic results (lower distance = better, so invert)
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results) + 0.01
            for doc, score in semantic_results:
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                normalized_score = 1 - (score / max_semantic)  # Invert distance
                doc_scores[chunk_id] = {
                    "doc": doc,
                    "semantic": normalized_score * semantic_weight,
                    "keyword": 0
                }
        
        # Process keyword results
        if keyword_results:
            max_keyword = max(score for _, score in keyword_results) + 0.01
            for doc, score in keyword_results:
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                normalized_score = score / max_keyword
                if chunk_id in doc_scores:
                    doc_scores[chunk_id]["keyword"] = normalized_score * keyword_weight
                else:
                    doc_scores[chunk_id] = {
                        "doc": doc,
                        "semantic": 0,
                        "keyword": normalized_score * keyword_weight
                    }
        
        # Calculate final scores and sort
        ranked_results = [
            (data["doc"], data["semantic"] + data["keyword"])
            for chunk_id, data in doc_scores.items()
        ]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, score in ranked_results[:k]]

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents using CrossEncoder for more accurate relevance scoring.

        The CrossEncoder sees query and passage together, allowing it to
        understand relationships that bi-encoders miss.

        Args:
            query: Search query
            documents: List of candidate documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        if not documents or not self.reranker:
            return documents[:top_k]

        # Create query-document pairs for CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Main retrieval method - uses hybrid search with optional reranking.

        Pipeline:
        1. Hybrid search (semantic + BM25) retrieves k*3 candidates
        2. CrossEncoder reranks candidates (if enabled)
        3. Return top k documents

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant documents
        """
        if self.use_reranker and self.reranker:
            # Get more candidates for reranking
            candidates = self.hybrid_search(query, k=k * 3)
            # Rerank and return top k
            return self.rerank(query, candidates, top_k=k)
        else:
            return self.hybrid_search(query, k=k)


def create_retriever(chunks: List[Document]) -> HybridRetriever:
    """
    Factory function to create a retriever.
    
    Args:
        chunks: Document chunks to index
        
    Returns:
        Configured HybridRetriever instance
    """
    return HybridRetriever(chunks)


# For testing
if __name__ == "__main__":
    from ingest import process_documents
    
    # Load and process documents
    chunks = process_documents()
    
    # Create retriever
    retriever = create_retriever(chunks)
    
    # Test search
    test_query = "What is attention mechanism?"
    print(f"\n--- Testing search for: '{test_query}' ---\n")
    
    results = retriever.get_relevant_documents(test_query, k=3)
    
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"  Content: {doc.page_content[:200]}...")
        print()