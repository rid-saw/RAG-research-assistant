from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings


class LLMConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not settings.google_api_key:
        raise LLMConfigError("GOOGLE_API_KEY is not set")
    return genai.Client(api_key=settings.google_api_key)


def generate(prompt: str, temperature: float | None = None) -> str:
    response = _client().models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
    )
    return response.text or ""


def generate_json(prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    import json

    response = _client().models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(response.text or "{}")
