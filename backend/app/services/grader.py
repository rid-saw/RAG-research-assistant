from langchain_core.documents import Document

from app.services.llm import generate_json

GRADER_PROMPT = """You decide whether the retrieved passages from the user's library can answer their question.

The user uploaded these papers and is asking about them. Treat the library as the intended context: if a term in the question matches a concept, method, model, or entity discussed in the passages, that is what the user means — even if the same word has a more common meaning elsewhere. Do not second-guess the user's intent based on dictionary definitions or external knowledge.

Mark "sufficient" if the passages discuss the question's subject, define a term it asks about, or contain partial information that could be combined into an answer.

Mark "insufficient" only if the passages genuinely do not address the question in any form — for example, the user asks about a topic the passages never mention.

Question: {query}

Retrieved passages:
{passages}

Reply with JSON:
- "sufficient": true if the passages address the question's subject (full or partial answer); false only if they don't address it at all
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
