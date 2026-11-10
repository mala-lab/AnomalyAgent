"""Tools for the AnomalyAgent (reconstructed implementation).

This module provides:
  * image helpers (base64 / data-URL / PIL / OpenCV conversion),
  * the five general visual tools of the paper (Appendix A.1) exposed as LangChain tools in ``TOOLS``,
  * the two "template tools" run before the planner:
      - ``_keyword_heuristic_decision``  (general template tool, Eq. 1-2)
      - ``counterfactual_atomic_candidate_tool`` (counterfactual tool, Eq. 3-6)

Image tool contract (see ``tools_node_with_image_injection`` in agent_v1_16.py):
  * ``image_base64`` is injected by the agent (pure base64 string, no data-URL prefix).
  * On success a tool returns a JSON string ``{"processed_image_base64": <data url>, "text": <msg>}``;
    the image value may be prefixed with ``BYPASS:`` when the image was left unchanged.
  * On failure a tool returns a string starting with ``Error:``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ============ General visual tools (paper Appendix A.1) ============

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


@tool
def image_denoising(image_base64: str = "", strength: float = 10.0) -> str:
    """Reduce noise or grain in the image (OpenCV non-local means denoising).

    Args:
        image_base64: Image to process. Optional: filled in automatically.
        strength: Filter strength h in [1, 30]. Default 10.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: OpenCV is not available for image denoising."
    try:
        img = _load_tool_image(image_base64)
        h = _clamp(strength, 1.0, 30.0, 10.0)
        out = cv2.fastNlMeansDenoisingColored(_pil_to_cv2(img), None, h, h, 7, 21)
        return _tool_result(_cv2_to_pil(out), f"Image denoising completed (non-local means, h={h:g}).")
    except Exception as e:
        return f"Error: image_denoising failed: {e}"


@tool
def image_deblurring(image_base64: str = "", amount: float = 1.5, sigma: float = 1.5) -> str:
    """Sharpen a blurry or out-of-focus image (unsharp masking).

    Args:
        image_base64: Image to process. Optional: filled in automatically.
        amount: Sharpening amount in [0.1, 5]. Default 1.5.
        sigma: Gaussian blur sigma in [0.3, 10]. Default 1.5.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: OpenCV is not available for image deblurring."
    try:
        img = _load_tool_image(image_base64)
        amt = _clamp(amount, 0.1, 5.0, 1.5)
        sig = _clamp(sigma, 0.3, 10.0, 1.5)
        bgr = _pil_to_cv2(img)
        blurred = cv2.GaussianBlur(bgr, (0, 0), sig)
        # unsharp masking: original + amount * (original - blurred)
        out = cv2.addWeighted(bgr, 1.0 + amt, blurred, -amt, 0)
        return _tool_result(_cv2_to_pil(out), f"Image deblurring completed (unsharp mask, amount={amt:g}, sigma={sig:g}).")
    except Exception as e:
        return f"Error: image_deblurring failed: {e}"


@tool
def image_brightness_enhancement(image_base64: str = "", clip_limit: float = 2.0) -> str:
    """Enhance brightness/contrast of dim or overexposed images (CLAHE on the LAB L channel).

    Args:
        image_base64: Image to process. Optional: filled in automatically.
        clip_limit: CLAHE clip limit in [0.5, 10]. Default 2.0.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: OpenCV is not available for brightness enhancement."
    try:
        img = _load_tool_image(image_base64)
        clip = _clamp(clip_limit, 0.5, 10.0, 2.0)
        lab = cv2.cvtColor(_pil_to_cv2(img), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return _tool_result(_cv2_to_pil(out), f"Brightness enhancement completed (CLAHE on LAB L-channel, clip={clip:g}).")
    except Exception as e:
        return f"Error: image_brightness_enhancement failed: {e}"


_SR_CACHE: Dict[str, Any] = {}


def _get_realesrgan_upsampler(version: str):
    """Build a Real-ESRGAN / Real-ESRNet upsampler, or return None when unavailable."""
    if version in _SR_CACHE:
        return _SR_CACHE[version]
    upsampler = None
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        env = "REALESRGAN_MODEL_PATH" if version == "gan" else "REALESRNET_MODEL_PATH"
        model_path = os.environ.get(env)
        if model_path and Path(model_path).is_file():
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            use_cuda = torch.cuda.is_available()
            upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=use_cuda,
                device=torch.device("cuda" if use_cuda else "cpu"),
            )
    except Exception as e:
        print(f"[tools] Real-ESRGAN unavailable ({type(e).__name__}: {e}); using Lanczos fallback.")
        upsampler = None
    _SR_CACHE[version] = upsampler
    return upsampler


@tool
def image_super_resolution(image_base64: str = "", scale: float = 2.0) -> str:
    """Enhance fine-grained details by upscaling (Real-ESRGAN; Lanczos fallback if unavailable).

    Args:
        image_base64: Image to process. Optional: filled in automatically.
        scale: Output upscaling factor in [1, 4]. Default 2. A value of 1 returns the image unchanged.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: OpenCV is not available for super-resolution."
    try:
        img = _load_tool_image(image_base64)
        sc = _clamp(scale, 1.0, 4.0, 2.0)
        if sc <= 1.0:
            return _tool_result(img, "Super-resolution skipped (scale=1, image unchanged).", bypass=True)

        w, h = img.size
        # Keep the output within a sane pixel budget.
        max_out_pixels = 4096 * 4096
        if (w * sc) * (h * sc) > max_out_pixels:
            sc = max(1.0, math.sqrt(max_out_pixels / (w * h)))
            if sc <= 1.0:
                return _tool_result(img, "Super-resolution skipped (image already very large).", bypass=True)

        version = os.environ.get("SUPER_RESOLUTION_VERSION", "nongan").lower()
        version = "gan" if version == "gan" else "nongan"
        upsampler = _get_realesrgan_upsampler(version)
        new_size = (max(1, int(round(w * sc))), max(1, int(round(h * sc))))
        if upsampler is not None:
            out, _ = upsampler.enhance(_pil_to_cv2(img), outscale=sc)
            result = _cv2_to_pil(out)
            if result.size != new_size:
                result = result.resize(new_size, PILImage.LANCZOS)
            method = "Real-ESRGAN" if version == "gan" else "Real-ESRNet"
        else:
            result = img.resize(new_size, PILImage.LANCZOS)
            method = "Lanczos interpolation (Real-ESRGAN not available)"
        return _tool_result(result, f"Super-resolution completed via {method}, scale={sc:g} ({w}x{h} -> {result.size[0]}x{result.size[1]}).")
    except Exception as e:
        return f"Error: image_super_resolution failed: {e}"


@tool
def image_zooming(
    image_base64: str = "",
    zoom_factor: float = 2.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> str:
    """Magnify a region of the image (crop around a center point and resize back to the original size).

    Args:
        image_base64: Image to process. Optional: filled in automatically.
        zoom_factor: Magnification in [1, 8]. Default 2. A value of 1 returns the image unchanged.
        center_x: Horizontal center of the region as a fraction of width in [0, 1]. Default 0.5.
        center_y: Vertical center of the region as a fraction of height in [0, 1]. Default 0.5.
    """
    try:
        img = _load_tool_image(image_base64)
        z = _clamp(zoom_factor, 1.0, 8.0, 2.0)
        if z <= 1.0:
            return _tool_result(img, "Zooming skipped (zoom_factor=1, image unchanged).", bypass=True)
        cx = _clamp(center_x, 0.0, 1.0, 0.5)
        cy = _clamp(center_y, 0.0, 1.0, 0.5)
        w, h = img.size
        cw, ch = max(1, int(round(w / z))), max(1, int(round(h / z)))
        x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
        y0 = int(round(min(max(cy * h - ch / 2, 0), h - ch)))
        crop = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), PILImage.LANCZOS)
        return _tool_result(
            crop,
            f"Zooming completed ({z:g}x on region x[{x0},{x0 + cw}] y[{y0},{y0 + ch}] of {w}x{h}).",
        )
    except Exception as e:
        return f"Error: image_zooming failed: {e}"


# ============ PromptOrder ============

class PromptOrder(str, Enum):
    """Placeholder. Imported by agent_v1_16 / agent_v1_27 but never referenced there."""

    DEFAULT = "default"


# ============ Template tools: shared utilities ============

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


# ============ General template tool (paper Eq. 1-2) ============
# WinCLIP-style fixed prompt ensemble: state words x template list (WinCLIP, Jeong et al., CVPR 2023).

_WINCLIP_NORMAL_STATES = [
    "{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect", "{} without damage",
]
_WINCLIP_ANOMALY_STATES = [
    "damaged {}", "{} with flaw", "{} with defect", "{} with damage",
]
_WINCLIP_TEMPLATES = [
    "a cropped photo of the {}.", "a cropped photo of a {}.", "a close-up photo of the {}.",
    "a close-up photo of a {}.", "a bright photo of the {}.", "a bright photo of a {}.",
    "a dark photo of the {}.", "a dark photo of a {}.", "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.", "a photo of the {}.", "a photo of a {}.",
    "a photo of a small {}.", "a photo of a large {}.", "a photo of the large {}.", "a photo of the small {}.",
]


def _winclip_prompts(class_name: str) -> Tuple[List[str], List[str]]:
    normal = [t.format(s.format(class_name)) for s in _WINCLIP_NORMAL_STATES for t in _WINCLIP_TEMPLATES]
    anomaly = [t.format(s.format(class_name)) for s in _WINCLIP_ANOMALY_STATES for t in _WINCLIP_TEMPLATES]
    return normal, anomaly


async def _class_prototypes(ctx: Any, class_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Normal / anomaly prototypes (mean of normalized prompt embeddings), cached per class."""
    cache = getattr(ctx, "_prototype_cache", None)
    if cache is None:
        cache = {}
        try:
            ctx._prototype_cache = cache
        except Exception:
            pass
    key = class_name.strip().lower()
    if key in cache:
        return cache[key]
    normal, anomaly = _winclip_prompts(class_name)
    embs = await _embed_texts(ctx, normal + anomaly)
    p_n = _l2norm(embs[: len(normal)].mean(axis=0))
    p_a = _l2norm(embs[len(normal):].mean(axis=0))
    cache[key] = (p_n, p_a)
    return p_n, p_a


# 평가 스크립트가 사진마다 읽어 가는 연속 점수 기록 (results.json 의 template_log)
RUN_LOG: Dict[str, Any] = {}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


async def _keyword_heuristic_decision(
    state: Any, runtime: Any, decision: bool = False
) -> Tuple[str, float, float, float, str]:
    """General template tool.

    Compares each of the three perspective captions with fixed normal / anomaly prototypes
    (embedding of the WinCLIP prompt ensemble).  For caption c_i:
        margin_i = cos(c_i, p_A) - cos(c_i, p_N)
        score_i  = sigmoid(margin_i / tau)           (anomaly probability)

    Returns (label, score1, score2, score3, report).  ``label`` is "anomalous" when the mean
    score exceeds 0.5.  With ``decision=True`` the report is a one-line suffix; otherwise it is
    the full report passed to the planner/reasoner as "General Template Analysis".
    """
    ctx = runtime.context
    class_name = _readable_class(_state_get(state, "class_name"))
    captions = await _get_image_captions(state, runtime)
    p_n, p_a = await _class_prototypes(ctx, class_name)
    cap_embs = await _embed_texts(ctx, captions)

    tau = float(getattr(ctx, "tau", 0.3)) or 0.3
    margins = [float(e @ p_a - e @ p_n) for e in cap_embs]
    scores = [_sigmoid(m / tau) for m in margins]
    # Always return exactly three scores (pad by repeating the mean if fewer captions were produced).
    mean_score = float(np.mean(scores))
    while len(scores) < 3:
        scores.append(mean_score)
        margins.append(float(np.mean(margins)))
    scores, margins = scores[:3], margins[:3]
    avg = float(np.mean(scores))
    label = "anomalous" if avg > 0.5 else "normal"
    if not decision:
        RUN_LOG["general"] = {"margins": margins, "scores": scores, "avg_score": avg, "avg_margin": float(np.mean(margins))}

    if decision:
        report = f"Heuristic decision from general template analysis: label={label}, avg_score={avg:.3f}."
        return label, scores[0], scores[1], scores[2], report

    lines = [f"Class: {class_name}", "Caption-vs-prototype comparison (positive margin = closer to anomaly prototype):"]
    for i, (c, m, s) in enumerate(zip(captions, margins, scores), 1):
        lines.append(f'  [{i}] "{c}" -> margin={m:+.3f}, anomaly_score={s:.3f}')
    lines.append(f"Mean anomaly score: {avg:.3f} -> template prediction: {label}.")
    return label, scores[0], scores[1], scores[2], "\n".join(lines)


# ============ Counterfactual atomic candidate tool (paper Eq. 3-6) ============

_candidates_file_cache: Dict[str, Dict[str, Any]] = {}


def _lookup_candidates(data: Dict[str, Any], class_name: str) -> Optional[Tuple[List[str], List[str]]]:
    want = class_name.strip().lower().replace("_", " ")
    for k, v in data.items():
        if isinstance(k, str) and k.strip().lower().replace("_", " ") == want and isinstance(v, dict):
            a = [c for c in v.get("anomaly_candidates", []) if isinstance(c, str) and c.strip()]
            n = [c for c in v.get("normal_candidates", []) if isinstance(c, str) and c.strip()]
            if a and n:
                return a, n
    return None


async def _get_candidates(ctx: Any, class_name_raw: str) -> Tuple[List[str], List[str]]:
    """Anomaly / normal candidates (E_A^C, E_N^C): from atomic_candidates_file, else generated by the LLM."""
    path = getattr(ctx, "atomic_candidates_file", None)
    if path and Path(path).is_file():
        try:
            mtime = Path(path).stat().st_mtime
            cached = _candidates_file_cache.get(path)
            if cached is None or cached["mtime"] != mtime:
                with open(path, "r", encoding="utf-8") as f:
                    cached = {"mtime": mtime, "data": json.load(f)}
                _candidates_file_cache[path] = cached
            found = _lookup_candidates(cached["data"], class_name_raw)
            if found:
                return found
        except Exception as e:
            print(f"[counterfactual] Failed to read candidates file {path}: {e}")

    import tempfile

    from .utils import generate_and_save_atomic_candidates

    with tempfile.TemporaryDirectory() as td:
        generated = await generate_and_save_atomic_candidates(
            [class_name_raw], ctx, output_file=os.path.join(td, "candidates.json")
        )
    found = _lookup_candidates(generated, class_name_raw)
    if not found:
        raise RuntimeError(f"Could not obtain atomic candidates for class '{class_name_raw}'.")
    return found


async def counterfactual_atomic_candidate_tool(state: Any, runtime: Any, k: int = 3) -> str:
    """Counterfactual atomic-candidate tool (paper Eq. 5).

    For each caption c_i (3 perspectives) the margin is
        m_i = max_{a in E_A^C} cos(c_i, a) - max_{n in E_N^C} cos(c_i, n)
    i.e. the best-matching anomaly candidate versus the best-matching normal candidate.
    Returns Report_CR and stores summary statistics in ``state.counterfactual_stats``.
    """
    ctx = runtime.context
    class_raw = _state_get(state, "class_name") or "object"
    class_name = _readable_class(class_raw)
    captions = await _get_image_captions(state, runtime)
    anomaly_c, normal_c = await _get_candidates(ctx, class_raw)

    cand_embs = await _embed_texts(ctx, anomaly_c + normal_c)
    ea, en = cand_embs[: len(anomaly_c)], cand_embs[len(anomaly_c):]
    cap_embs = await _embed_texts(ctx, captions)
    k = max(1, int(k))

    per = []
    for cap, ce in zip(captions, cap_embs):
        if not np.all(np.isfinite(ce)):
            continue
        sa, sn = ea @ ce, en @ ce
        top_a = np.argsort(-sa)[:k]
        top_n = np.argsort(-sn)[:k]
        max_a, max_n = float(sa.max()), float(sn.max())
        per.append(
            {
                "caption": cap,
                "margin": float(max_a - max_n),
                "max_sim_anom": max_a,
                "max_sim_norm": max_n,
                "evidence_strength": max(max_a, max_n),
                "top_anom": [(anomaly_c[i], float(sa[i])) for i in top_a],
                "top_norm": [(normal_c[i], float(sn[i])) for i in top_n],
            }
        )

    num_total = len(captions)
    if per:
        stats = {
            "num_valid": len(per),
            "num_total": num_total,
            "avg_margin": float(np.mean([p["margin"] for p in per])),
            "avg_evidence_strength": float(np.mean([p["evidence_strength"] for p in per])),
            "avg_max_sim_anom": float(np.mean([p["max_sim_anom"] for p in per])),
            "avg_max_sim_norm": float(np.mean([p["max_sim_norm"] for p in per])),
        }
    else:
        stats = {"num_valid": 0, "num_total": num_total, "avg_margin": None, "avg_evidence_strength": None,
                 "avg_max_sim_anom": None, "avg_max_sim_norm": None}
    _state_set(state, "counterfactual_stats", stats)
    RUN_LOG["counterfactual"] = {**stats, "margins": [p["margin"] for p in per]}
    RUN_LOG["captions"] = list(captions)

    lines = [
        f"Class: {class_name}",
        f"Counterfactual matching of {num_total} caption(s) against {len(anomaly_c)} anomaly / {len(normal_c)} normal atomic candidates "
        f"(positive margin = closer to anomaly candidates).",
    ]
    for i, p in enumerate(per, 1):
        lines.append(f'[{i}] "{p["caption"]}" -> margin={p["margin"]:+.3f} (max_sim anomaly={p["max_sim_anom"]:.3f}, normal={p["max_sim_norm"]:.3f})')
        lines.append("    top anomaly matches: " + "; ".join(f'"{c}" ({s:.2f})' for c, s in p["top_anom"]))
        lines.append("    top normal matches: " + "; ".join(f'"{c}" ({s:.2f})' for c, s in p["top_norm"]))
    if per:
        am = stats["avg_margin"]
        verdict = "leans anomalous" if am > 0 else "leans normal"
        lines.append(f"Summary: average margin={am:+.3f}, average evidence strength={stats['avg_evidence_strength']:.3f} -> {verdict}.")
    else:
        lines.append("Summary: no valid caption perspectives; counterfactual evidence unavailable.")
    return "\n".join(lines)


# ============ Tool registry ============

TOOLS = [
    image_denoising,
    image_deblurring,
    image_super_resolution,
    image_zooming,
    image_brightness_enhancement,
]
