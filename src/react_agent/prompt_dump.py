"""Opt-in dump of every successful API call (what each node received and returned).

Enabled only when the env var DUMP_PROMPTS_DIR is set; otherwise a no-op, so agent behavior is unchanged.
Output: <dir>/calls.jsonl (one record per call) and <dir>/images/<hash>.<ext> (images sent to the model).
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_SAMPLE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_sample", default=None)
_seq = 0


def _redact_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ("[IMAGE]" if k == "image_base64" else v) for k, v in (args or {}).items()}


def _part(part: Any, img_dir: Path) -> Dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if isinstance(part, dict):
        t = part.get("type")
        if t == "text":
            return {"type": "text", "text": part.get("text", "")}
        if t == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                try:
                    raw = base64.b64decode(url.split(",", 1)[1] + "==")
                    ext = url.split(";", 1)[0].split("/")[-1].replace("jpeg", "jpg")
                    name = f"{hashlib.sha1(raw).hexdigest()[:12]}.{ext}"
                    f = img_dir / name
                    if not f.exists():
                        f.write_bytes(raw)
                    return {"type": "image", "file": f"images/{name}", "bytes": len(raw)}
                except Exception:
                    pass
            return {"type": "image", "file": None, "url": str(url)[:200]}
        return {"type": t or "other", "text": json.dumps(part, ensure_ascii=False, default=str)[:4000]}
    return {"type": "text", "text": str(part)[:4000]}


def _message(m: Any, img_dir: Path) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if isinstance(m, dict):
        role, content = m.get("role", "?"), m.get("content")
    else:
        role, content = getattr(m, "type", type(m).__name__), getattr(m, "content", "")
        if getattr(m, "tool_calls", None):
            extra["tool_calls"] = [{"name": tc.get("name"), "args": _redact_args(tc.get("args"))} for tc in m.tool_calls]
    parts = content if isinstance(content, list) else [content]
    return {"role": role, "parts": [_part(p, img_dir) for p in parts], **extra}


def dump_api_call(node_name: str, payload: List[Any], response: Any, model_id: Optional[str]) -> None:
    global _seq
    root = os.environ.get("DUMP_PROMPTS_DIR")
    if not root:
        return
    try:
        root_p = Path(root)
        img_dir = root_p / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        _seq += 1
        resp_parts = response.content if isinstance(response.content, list) else [response.content]
        rec = {
            "seq": _seq,
            "sample": CURRENT_SAMPLE.get(),
            "node": node_name,
            "model": model_id,
            "messages": [_message(m, img_dir) for m in payload],
            "response": {
                "parts": [_part(p, img_dir) for p in resp_parts],
                "tool_calls": [
                    {"name": tc.get("name"), "args": _redact_args(tc.get("args"))}
                    for tc in (getattr(response, "tool_calls", None) or [])
                ],
                "usage": getattr(response, "usage_metadata", None),
            },
        }
        with open(root_p / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # logging must never break a run
        print(f"[dump] failed: {e}")
