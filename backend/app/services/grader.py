from langchain_core.documents import Document

from app.services.llm import generate_json

GRADER_PROMPT = """You decide whether the retrieved passages from the user's library can answer their question.

THE LIBRARY IS THE INTENDED CONTEXT. The user uploaded these papers because they want to know what's in them. If a term in the question matches a concept, method, model, system, or entity discussed in the passages, that IS what the user is asking about. You must NOT refuse on the grounds that the word has a different common meaning in everyday usage.

Examples of correct behavior:

Example 1 (sufficient — term defined in passages):
  Question: "what is hydrant?"
  Passages define Hydrant as a hybrid ML model for time series classification.
  Verdict: sufficient=true. The library defines the term; that is the user's question.
  (Refusing because "hydrant" usually means a water device would be WRONG.)

Example 2 (sufficient — partial info):
  Question: "how does attention scale with sequence length?"
  Passages mention quadratic complexity but don't give a full derivation.
  Verdict: sufficient=true. Partial information can be combined into an answer.

Example 3 (insufficient — truly off-topic):
  Question: "what is the climate of Madagascar?"
  Passages discuss neural network architectures.
  Verdict: sufficient=false. Passages do not address the topic in any form.

Now grade the actual query:

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
