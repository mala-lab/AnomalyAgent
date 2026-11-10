"""General template tool (paper Eq. 1-2): WinCLIP-style prompt prototypes vs. the image captions."""

from __future__ import annotations

import json
import math
from typing import Any, List, Tuple

import numpy as np
from langchain_core.tools import tool

from ._common import RUN_LOG, _embed_texts, _get_image_captions, _l2norm, _readable_class, _state_get



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
