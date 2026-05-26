"""Helpers for normalizing provider/model identifiers used by API models."""

from __future__ import annotations

from typing import Optional, Tuple


_PROVIDER_ALIASES = {
    "google": "google_genai",
    "google_genai": "google_genai",
    "google-genai": "google_genai",
    "genai": "google_genai",
    "gemini": "google_genai",
    "google_vertexai": "google_vertexai",
    "google-vertexai": "google_vertexai",
    "vertexai": "google_vertexai",
    "vertex_ai": "google_vertexai",
}


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    """Normalize provider aliases accepted by CLI/config."""
    if provider is None:
        return None
    cleaned = provider.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    return _PROVIDER_ALIASES.get(lowered, lowered)


def normalize_model_name(model_name: str) -> str:
    """Normalize model names while preserving provider-specific model ids."""
    cleaned = model_name.strip()
    if cleaned.startswith("models/gemini-"):
        return cleaned.removeprefix("models/")
    return cleaned


def parse_api_model_id(api_model: str) -> Tuple[Optional[str], str]:
    """Parse API model ids in provider/model, provider:model, or bare model format.

    Existing evaluation scripts use ``provider/model`` (for example
    ``anthropic/claude-...``). LangChain also supports ``provider:model``. For
    bare Gemini ids, prefer the Gemini Developer API integration
    (``google_genai``) instead of LangChain's default Vertex inference.
    """
    raw = (api_model or "").strip()
    if not raw:
        raise ValueError("api_model must be a non-empty string")

    provider: Optional[str] = None
    model_name = raw

    if ":" in raw:
        maybe_provider, maybe_model = raw.split(":", maxsplit=1)
        if maybe_provider and maybe_model:
            provider = normalize_provider(maybe_provider)
            model_name = maybe_model
    elif "/" in raw and not raw.startswith("models/gemini-"):
        maybe_provider, maybe_model = raw.split("/", maxsplit=1)
        if maybe_provider and maybe_model:
            provider = normalize_provider(maybe_provider)
            model_name = maybe_model
    elif raw.startswith("gemini-") or raw.startswith("models/gemini-"):
        provider = "google_genai"
        model_name = raw

    return provider, normalize_model_name(model_name)


def get_api_model_provider(api_model: Optional[str]) -> Optional[str]:
    """Return the normalized provider for an API model id, if known."""
    if not api_model:
        return None
    try:
        provider, _ = parse_api_model_id(api_model)
    except ValueError:
        return None
    return provider


def is_google_genai_model(api_model: Optional[str]) -> bool:
    """Whether the model id routes through LangChain's Google GenAI package."""
    return get_api_model_provider(api_model) == "google_genai"


def is_google_model(api_model: Optional[str]) -> bool:
    """Whether the model id routes through a Google Gemini provider."""
    return get_api_model_provider(api_model) in {"google_genai", "google_vertexai"}


def is_openai_model(api_model: Optional[str]) -> bool:
    """Whether the model id routes through OpenAI-compatible first-party provider."""
    return get_api_model_provider(api_model) == "openai"
