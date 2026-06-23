import json
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


class LLMConfigError(RuntimeError):
    pass


PROVIDER_BASES = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    provider = settings.llm_provider
    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMConfigError("GROQ_API_KEY is not set")
        return OpenAI(api_key=settings.groq_api_key, base_url=PROVIDER_BASES["groq"])
    if provider == "gemini":
        if not settings.google_api_key:
            raise LLMConfigError("GOOGLE_API_KEY is not set")
        return OpenAI(api_key=settings.google_api_key, base_url=PROVIDER_BASES["gemini"])
    raise LLMConfigError(f"Unknown LLM provider: {provider}")


def generate(prompt: str, temperature: float | None = None) -> str:
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
    )
    return response.choices[0].message.content or ""


def generate_json(prompt: str, schema: dict, temperature: float = 0.0) -> dict:
    full_prompt = (
        f"{prompt}\n\n"
        f"Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
    )
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=settings.llm_max_output_tokens,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)
