from collections.abc import Iterator

from langchain_core.documents import Document

from app.services.grader import grade_chunks
from app.services.llm import generate, generate_stream
from app.services.query_rewriter import rewrite_for_web
from app.services.retriever import get_retriever
from app.services.web_search import search_web

INSUFFICIENT_TOKEN = "INSUFFICIENT_CONTEXT"

REFUSAL_MESSAGE = (
    "I couldn't find a reliable answer in the available sources. "
    "Try rephrasing your question, or add more documents to this library."
)

ANSWER_PROMPT = """You are a research assistant. Answer the user's question using ONLY the provided context passages.

Citations: cite each claim with bracketed numbers like [1], [2], [3] that correspond to the numbered passages below. Place the bracket right after the claim it supports. Multiple sources per claim are fine: [1][3].

If the context does not contain enough information to give a grounded, specific answer, reply with EXACTLY this token and nothing else: INSUFFICIENT_CONTEXT

Do not invent facts. Do not pad. Do not apologize.

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


def _is_refusal(answer: str) -> bool:
    return INSUFFICIENT_TOKEN in answer.strip()


def run(query: str, collection: str, allow_web_search: bool, top_k: int | None = None) -> dict:
    retriever = get_retriever(collection)
    chunks = retriever.search(query, top_k=top_k)

    sufficient, reason = grade_chunks(query, chunks)

    if not sufficient:
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
                "status": "refused",
                "answer": REFUSAL_MESSAGE,
                "citations": [],
                "reason": "Web search returned no results.",
                "rewritten_query": rewritten,
            }

        answer = _answer_with(query, web_docs)
        if _is_refusal(answer):
            return {
                "status": "refused",
                "answer": REFUSAL_MESSAGE,
                "citations": [],
                "reason": "Web results were not specific enough to ground an answer.",
                "rewritten_query": rewritten,
            }

        return {
            "status": "answered_from_web",
            "answer": answer,
            "citations": web_docs,
            "reason": reason,
            "rewritten_query": rewritten,
        }

    answer = _answer_with(query, chunks)
    if _is_refusal(answer):
        return {
            "status": "refused",
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "reason": "Retrieved passages did not contain a specific answer.",
            "rewritten_query": None,
        }

    return {
        "status": "answered",
        "answer": answer,
        "citations": chunks,
        "reason": None,
        "rewritten_query": None,
    }


def _citations_payload(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source"),
            "title": d.metadata.get("title"),
            "page": d.metadata.get("page"),
            "chunk_id": d.metadata.get("chunk_id"),
            "snippet": d.page_content[:200],
        }
        for d in docs
    ]


def run_stream(
    query: str,
    collection: str,
    allow_web_search: bool,
    top_k: int | None = None,
) -> Iterator[dict]:
    """Yield pipeline events: stage updates, token chunks, and a final done event.

    Event shapes:
      {"event": "stage", "stage": "retrieving"|"grading"|"generating"|"web_search"}
      {"event": "token", "text": "..."}
      {"event": "done", "status": "...", "citations": [...], "reason": ..., "rewritten_query": ...}
    """
    yield {"event": "stage", "stage": "retrieving"}
    retriever = get_retriever(collection)
    chunks = retriever.search(query, top_k=top_k)

    yield {"event": "stage", "stage": "grading"}
    sufficient, reason = grade_chunks(query, chunks)

    if not sufficient:
        if not allow_web_search:
            yield {
                "event": "done",
                "status": "needs_web_search",
                "citations": [],
                "reason": reason,
                "rewritten_query": None,
            }
            return

        yield {"event": "stage", "stage": "web_search"}
        rewritten = rewrite_for_web(query)
        web_docs = search_web(rewritten)

        if not web_docs:
            yield {
                "event": "done",
                "status": "refused",
                "citations": [],
                "reason": "Web search returned no results.",
                "rewritten_query": rewritten,
            }
            return

        yield {"event": "stage", "stage": "generating"}
        prompt = ANSWER_PROMPT.format(context=_format_context(web_docs), query=query)
        full = ""
        for chunk in generate_stream(prompt):
            full += chunk
            yield {"event": "token", "text": chunk}

        if _is_refusal(full):
            yield {
                "event": "done",
                "status": "refused",
                "citations": [],
                "reason": "Web results were not specific enough to ground an answer.",
                "rewritten_query": rewritten,
            }
            return

        yield {
            "event": "done",
            "status": "answered_from_web",
            "citations": _citations_payload(web_docs),
            "reason": reason,
            "rewritten_query": rewritten,
        }
        return

    yield {"event": "stage", "stage": "generating"}
    prompt = ANSWER_PROMPT.format(context=_format_context(chunks), query=query)
    full = ""
    for chunk in generate_stream(prompt):
        full += chunk
        yield {"event": "token", "text": chunk}

    if _is_refusal(full):
        yield {
            "event": "done",
            "status": "refused",
            "citations": [],
            "reason": "Retrieved passages did not contain a specific answer.",
            "rewritten_query": None,
        }
        return

    yield {
        "event": "done",
        "status": "answered",
        "citations": _citations_payload(chunks),
        "reason": None,
        "rewritten_query": None,
    }
