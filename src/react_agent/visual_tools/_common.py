"""Shared helpers for the visual tools: base64 / data-URL / PIL / OpenCV conversion and the tool result format."""

from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from langchain_core.tools import tool
from PIL import Image as PILImage


try:
    import cv2

    IMAGE_PROCESSING_AVAILABLE = True
except Exception:  # pragma: no cover - cv2 missing
    cv2 = None  # type: ignore[assignment]
    IMAGE_PROCESSING_AVAILABLE = False


# ============ Image helpers ============

_MAX_PIXELS = 1024 * 1024 * 1024
_DATA_URL_RE = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,", re.IGNORECASE)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_MAGIC_TO_MIME = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _sniff_mime(data: bytes) -> Optional[str]:
    for magic, mime in _MAGIC_TO_MIME:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _b64decode_lenient(s: str) -> Optional[bytes]:
    s = s.strip()
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return None


def to_pure_base64(image_str: Any) -> Optional[str]:
    """Return the raw base64 payload of a data URL / base64 string (None if not a string)."""
    if not isinstance(image_str, str):
        return None
    s = image_str.strip()
    if not s:
        return None
    if s.startswith("BYPASS:"):
        s = s[len("BYPASS:"):]
    if s.lower().startswith("data:"):
        if "," not in s:
            return None
        s = s.split(",", 1)[1]
    return re.sub(r"\s+", "", s)


def _encode_image_to_base64(image: PILImage.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image into a data URL (``data:image/png;base64,...``)."""
    if not isinstance(image, PILImage.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")
    fmt = fmt.upper()
    img = image
    if fmt in {"JPEG", "JPG"} and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "RGBA", "L", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if fmt == "JPG" else fmt)
    mime = "image/jpeg" if fmt in {"JPEG", "JPG"} else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _normalize_to_data_url(image_str: Any) -> Optional[str]:
    """Normalize a data URL / raw base64 / local path / http(s) URL to a usable image URL.

    Returns None when the input is not (recognizably) an image reference, so arbitrary
    text (e.g. a plain-text HumanMessage) is never mistaken for an image.
    """
    if not isinstance(image_str, str):
        return None
    s = image_str.strip()
    if not s:
        return None
    if s.startswith("BYPASS:"):
        s = s[len("BYPASS:"):].strip()

    if _DATA_URL_RE.match(s):
        payload = re.sub(r"\s+", "", s.split(",", 1)[1])
        data = _b64decode_lenient(payload)
        if not data:
            return None
        mime = s[5:].split(";", 1)[0]
        return f"data:{mime};base64,{payload}"
    if s.lower().startswith("data:"):
        return None
    if s.lower().startswith(("http://", "https://")):
        return s

    if s.lower().startswith("file://"):
        s = s[7:]
        if re.match(r"^/[A-Za-z]:", s):
            s = s[1:]
    if len(s) < 4096 and "\n" not in s:
        try:
            p = Path(s)
            if p.is_file():
                with PILImage.open(p) as img:
                    img.load()
                    return _encode_image_to_base64(img)
        except Exception:
            pass

    compact = re.sub(r"\s+", "", s)
    if len(compact) >= 100 and len(compact) == len(s.strip().replace("\n", "").replace("\r", "")) and _BASE64_RE.match(compact):
        if "-" in compact or "_" in compact:
            compact = compact.replace("-", "+").replace("_", "/")
        data = _b64decode_lenient(compact)
        if data:
            mime = _sniff_mime(data)
            if mime:
                return f"data:{mime};base64,{compact}"
    return None


def _decode_base64_image(data: Any) -> PILImage.Image:
    """Decode a data URL / base64 string into an RGB PIL image (EXIF-transposed)."""
    from PIL import ImageOps

    b64 = to_pure_base64(data)
    if not b64:
        raise ValueError("Empty or invalid base64 image data")
    raw = _b64decode_lenient(b64)
    if not raw:
        raise ValueError("Failed to base64-decode image data")
    old = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = _MAX_PIXELS
    try:
        img = PILImage.open(io.BytesIO(raw))
        img.load()
    finally:
        PILImage.MAX_IMAGE_PIXELS = old
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def _pil_to_cv2(image: PILImage.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV BGR uint8 array."""
    arr = np.array(image.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _cv2_to_pil(array: np.ndarray) -> PILImage.Image:
    """OpenCV BGR (or gray) array -> PIL RGB image."""
    if array.ndim == 2:
        return PILImage.fromarray(array).convert("RGB")
    return PILImage.fromarray(array[:, :, ::-1].copy())


def _message_image_urls(msg: Any) -> List[str]:
    """All image URLs (data URLs) found in a message, in order."""
    urls: List[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    norm = _normalize_to_data_url(url)
                    if norm:
                        urls.append(norm)
                elif item.get("type") == "image" and isinstance(item.get("image"), PILImage.Image):
                    try:
                        urls.append(_encode_image_to_base64(item["image"]))
                    except Exception:
                        pass
            elif isinstance(item, PILImage.Image):
                try:
                    urls.append(_encode_image_to_base64(item))
                except Exception:
                    pass
    elif isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("processed_image_base64", "image_base64", "image", "result_image", "output_image"):
                if key in parsed:
                    norm = _normalize_to_data_url(parsed[key])
                    if norm:
                        urls.append(norm)
                        break
        else:
            norm = _normalize_to_data_url(content)
            if norm:
                urls.append(norm)
    return urls


def _extract_image_from_messages(messages: Any) -> Optional[str]:
    """Most recent image in the message history (tool outputs first, then user images)."""
    for msg in reversed(list(messages or [])):
        urls = _message_image_urls(msg)
        if urls:
            return urls[-1]
    return None


def _tool_result(image: PILImage.Image, text: str, bypass: bool = False) -> str:
    data_url = _encode_image_to_base64(image, "PNG")
    return json.dumps(
        {
            "processed_image_base64": ("BYPASS:" + data_url) if bypass else data_url,
            "text": text,
        }
    )


def _load_tool_image(image_base64: Optional[str]) -> PILImage.Image:
    if not image_base64:
        raise ValueError("image_base64 is missing")
    return _decode_base64_image(image_base64)


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return max(lo, min(hi, f))
