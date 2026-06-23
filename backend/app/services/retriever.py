from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from app.core.chroma import get_chroma_client
from app.core.config import settings
from app.core.embeddings import get_embeddings
from app.services.reranker import get_reranker


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, collection_name: str = "default") -> None:
        self.collection_name = collection_name
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            client=get_chroma_client(),
        )
        self._documents: List[Document] = []
        self._bm25: BM25Okapi | None = None
        self._hydrate_from_store()

    def _hydrate_from_store(self) -> None:
        data = self.vector_store.get()
        if not data.get("ids"):
            self._documents = []
            self._bm25 = None
            return
        self._documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self._rebuild_bm25()

    def _chroma_count(self) -> int:
        return self.vector_store._collection.count()

    def _sync_if_stale(self) -> None:
        """If chroma has more (or fewer) docs than we cached, rehydrate.

        Catches the case where the collection was modified in another process
        or by a different code path while this retriever instance was alive.
        """
        try:
            disk = self._chroma_count()
        except Exception:
            return
        if disk != len(self._documents):
            self._hydrate_from_store()

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "in_memory_docs": len(self._documents),
            "chroma_count": self._chroma_count(),
            "bm25_ready": self._bm25 is not None,
        }

    def _rebuild_bm25(self) -> None:
        if not self._documents:
            self._bm25 = None
            return
        tokenized = [_tokenize(d.page_content) for d in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def _doc_key(self, doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is not None:
            return str(chunk_id)
        return f"{doc.metadata.get('source', '')}::{hash(doc.page_content)}"

    def add_documents(self, docs: List[Document]) -> None:
        valid = [d for d in docs if d.page_content.strip()]
        if not valid:
            return
        self.vector_store.add_documents(valid)
        self._documents.extend(valid)
        self._rebuild_bm25()

    def semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k)

    def keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._bm25 or not self._documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self._documents[i], float(scores[i])) for i in top_indices]

    def hybrid_search(self, query: str, k: int) -> List[Document]:
        semantic = self.semantic_search(query, k=k * 2)
        keyword = self.keyword_search(query, k=k * 2)

        fused: dict[str, dict] = {}

        if semantic:
            max_s = max(score for _, score in semantic) + 1e-6
            for doc, score in semantic:
                key = self._doc_key(doc)
                entry = fused.setdefault(key, {"doc": doc, "semantic": 0.0, "keyword": 0.0})
                entry["semantic"] = (1 - score / max_s) * settings.semantic_weight

        if keyword:
            max_k = max(score for _, score in keyword) + 1e-6
            for doc, score in keyword:
                key = self._doc_key(doc)
                entry = fused.setdefault(key, {"doc": doc, "semantic": 0.0, "keyword": 0.0})
                entry["keyword"] = (score / max_k) * settings.keyword_weight

        ranked = sorted(
            fused.values(),
            key=lambda v: v["semantic"] + v["keyword"],
            reverse=True,
        )
        return [v["doc"] for v in ranked[:k]]

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = get_reranker().predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def search(self, query: str, top_k: int | None = None) -> List[Document]:
        self._sync_if_stale()
        top_k = top_k or settings.rerank_top_k
        candidates = self.hybrid_search(query, k=settings.candidates_per_search)
        return self.rerank(query, candidates, top_k=top_k)


_retriever_cache: dict[str, HybridRetriever] = {}


def get_retriever(collection_name: str = "default") -> HybridRetriever:
    if collection_name not in _retriever_cache:
        _retriever_cache[collection_name] = HybridRetriever(collection_name)
    return _retriever_cache[collection_name]
