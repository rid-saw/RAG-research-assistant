from langchain.schema import Document

from app.services.grader import grade_chunks
from app.services.llm import generate
from app.services.query_rewriter import rewrite_for_web
from app.services.retriever import get_retriever
from app.services.web_search import search_web

ANSWER_PROMPT = """You are a research assistant. Answer the user's question using ONLY the provided context passages. Cite each claim with the marker [source p.page] inline using the metadata above each passage.

If the context does not contain enough information, say so explicitly. Do not invent facts.

Context passages:
{context}

Question: {query}

Answer:"""


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        marker = f"{source} p.{page}" if page is not None else source
        blocks.append(f"[Passage {i} | {marker}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _answer_with(query: str, docs: list[Document]) -> str:
    prompt = ANSWER_PROMPT.format(context=_format_context(docs), query=query)
    return generate(prompt)


def run(query: str, collection: str, allow_web_search: bool, top_k: int | None = None) -> dict:
    retriever = get_retriever(collection)
    chunks = retriever.search(query, top_k=top_k)

    sufficient, reason = grade_chunks(query, chunks)

    if sufficient:
        return {
            "status": "answered",
            "answer": _answer_with(query, chunks),
            "citations": chunks,
            "reason": None,
            "rewritten_query": None,
        }

    if not allow_web_search:
        return {
            "status": "needs_web_search",
            "answer": "",
            "citations": [],
            "reason": reason,
            "rewritten_query": None,
        }

    rewritten = rewrite_for_web(query)
    web_docs = search_web(rewritten)

    if not web_docs:
        return {
            "status": "needs_web_search",
            "answer": "",
            "citations": [],
            "reason": "Web search returned no results.",
            "rewritten_query": rewritten,
        }

    return {
        "status": "answered_from_web",
        "answer": _answer_with(query, web_docs),
        "citations": web_docs,
        "reason": reason,
        "rewritten_query": rewritten,
    }
