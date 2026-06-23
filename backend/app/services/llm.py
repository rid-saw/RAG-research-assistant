import json
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


class LLMConfigError(RuntimeError):
    pass


PROVIDER_CONFIG = {
    "mistral": ("mistral_api_key", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    "gemini": (
        "google_api_key",
        "GOOGLE_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "openai": ("openai_api_key", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "groq": ("groq_api_key", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    provider = settings.llm_provider
    if provider not in PROVIDER_CONFIG:
        raise LLMConfigError(f"Unknown LLM provider: {provider}")
    attr, env_name, base_url = PROVIDER_CONFIG[provider]
    key = getattr(settings, attr)
    if not key:
        raise LLMConfigError(f"{env_name} is not set")
    return OpenAI(api_key=key, base_url=base_url)


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
