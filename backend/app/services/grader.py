from langchain_core.documents import Document

from app.services.llm import generate_json

GRADER_PROMPT = """You are evaluating whether the retrieved passages are sufficient to faithfully answer the user's question.

Be strict: if the passages only tangentially touch the topic, mark as insufficient. The user can fall back to web search, so it's better to flag uncertainty than to hallucinate.

Question: {query}

Retrieved passages:
{passages}

Reply with JSON:
- "sufficient": true if the passages contain enough information to give a grounded, specific answer; false otherwise
- "reason": one short sentence explaining your verdict (will be shown to the user if insufficient)"""


GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["sufficient", "reason"],
}


def grade_chunks(query: str, chunks: list[Document]) -> tuple[bool, str]:
    if not chunks:
        return False, "No relevant passages found in this library."

    passages = "\n\n".join(
        f"[Passage {i}]\n{c.page_content}" for i, c in enumerate(chunks, start=1)
    )
    prompt = GRADER_PROMPT.format(query=query, passages=passages)
    result = generate_json(prompt, GRADER_SCHEMA)
    return bool(result.get("sufficient", False)), str(result.get("reason", ""))
