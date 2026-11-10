"""Shared utilities for the template tools: state access, text embedding, the per-image captions, and RUN_LOG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool

from ..visual_tools._common import _decode_base64_image, _extract_image_from_messages, _message_image_urls, _normalize_to_data_url


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_set(state: Any, key: str, value: Any) -> None:
    try:
        if isinstance(state, dict):
            state[key] = value
        else:
            setattr(state, key, value)
    except Exception:
        pass


def _get_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: List[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            t = v.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
            for c in v.values():
                if isinstance(c, (dict, list)):
                    walk(c)
        elif isinstance(v, list):
            for c in v:
                walk(c)

    walk(content)
    return "\n".join(parts)


def _readable_class(class_name: Optional[str]) -> str:
    return (class_name or "object").replace("_", " ").replace("-", " ").strip() or "object"


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


async def _embed_texts(ctx: Any, texts: List[str]) -> np.ndarray:
    emb = ctx.embedding_model
    if emb is None:
        raise RuntimeError("Embedding model is not available (set --embedding_model_path).")
    vecs = await emb.aembed_documents(texts)
    return _l2norm(np.asarray(vecs, dtype=np.float32))


def _raw_image_for_state(state: Any) -> Optional[str]:
    url = _state_get(state, "raw_image_url")
    if url:
        return _normalize_to_data_url(url) or url
    messages = _state_get(state, "messages", []) or []
    # First user-supplied image (raw), not a tool output.
    from langchain_core.messages import HumanMessage

    for msg in messages:
        if isinstance(msg, HumanMessage):
            urls = _message_image_urls(msg)
            if urls:
                return urls[0]
    return _extract_image_from_messages(messages)


_CAPTION_ATTEMPTS = 2


async def _get_image_captions(state: Any, runtime: Any) -> List[str]:
    """Three perspective captions (PROMPT4DESCRIPTION) for the raw image; cached per image."""
    import hashlib

    from langchain_core.messages import HumanMessage

    from .utils import _parse_json_from_text, fit_image_to_budget, safe_api_invoke

    ctx = runtime.context
    cached = _state_get(state, "image_captions")
    if cached:
        return list(cached)

    image_url = _raw_image_for_state(state)
    if not image_url:
        raise ValueError("No image available for caption generation.")
    key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    cache = getattr(ctx, "_caption_cache", None)
    if cache is None:
        cache = {}
        try:
            ctx._caption_cache = cache
        except Exception:
            pass
    if key in cache:
        captions = list(cache[key])
        _state_set(state, "image_captions", captions)
        return captions

    safe_url = fit_image_to_budget(
        image_url, max_bytes=4 * 1024 * 1024, jpeg_quality=85, scale_decay=0.9, decode_func=_decode_base64_image
    )
    class_name = _readable_class(_state_get(state, "class_name"))
    text = f"{ctx.prompt4description}\n\nObject class: {class_name}"
    msg = HumanMessage(
        content=[{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": safe_url}}]
    )
    model = ctx.image_description_model

    captions: List[str] = []
    for _ in range(_CAPTION_ATTEMPTS):
        resp = await safe_api_invoke(model, [msg], "image_description", ctx)
        parsed = _parse_json_from_text(_get_text(resp.content))
        raw = parsed.get("captions") if isinstance(parsed, dict) else None
        if isinstance(raw, list):
            captions = [c.strip() for c in raw if isinstance(c, str) and c.strip()]
        if captions:
            break
    if not captions:
        raise RuntimeError("Image description model returned no valid captions.")
    cache[key] = list(captions)
    _state_set(state, "image_captions", list(captions))
    return captions


# Filled by the two template tools; evaluate_agg_v12_target.py clears and copies it per image.
RUN_LOG: Dict[str, Any] = {}
