from langchain_core.documents import Document

from app.services.llm import generate_json

GRADER_PROMPT = """You are deciding whether the retrieved passages contain enough information to answer the user's question.

Default to "sufficient" if the passages discuss the question's topic or contain partial information that could be combined into an answer. Only mark "insufficient" if the passages are clearly off-topic, contradict the question's premise, or contain none of the information needed to answer.

Question: {query}

Retrieved passages:
{passages}

Reply with JSON:
- "sufficient": true if the passages discuss the topic and contain answer-relevant information (full or partial); false only if clearly off-topic
- "reason": one short sentence explaining your verdict (shown to the user if insufficient)"""


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
