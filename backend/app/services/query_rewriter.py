from app.services.llm import generate

REWRITE_PROMPT = """Rewrite the user's question into a concise web search query (under 15 words). Strip conversational filler, add specific terms a search engine would match on.

Question: {query}

Search query:"""


def rewrite_for_web(query: str) -> str:
    return generate(REWRITE_PROMPT.format(query=query), temperature=0.0).strip()
