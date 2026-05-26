from __future__ import annotations

from typing import Any

from PIL import Image


def resize_image(image: Image.Image, n_px: int) -> Image.Image:
    """Resize image to (n_px, n_px) with BICUBIC interpolation.

    This matches the resize behavior used in:
    `src/react_agent/clip_visual_encoder/anomalyclip_lib/model_load.py::_transform`.
    """
    if n_px <= 0:
        raise ValueError(f"n_px must be positive, got {n_px}")
    return image.resize((n_px, n_px), resample=Image.Resampling.BICUBIC)


ROI_WINDOW_SPECS = {
    0: (2, 2),
    1: (4, 4),
    2: (6, 6),
    3: (9, 9),
    4: (12, 12),
    5: (2, 8),
    6: (3, 12),
    7: (8, 2),
    8: (12, 3),
}


def _roi_bbox(size_id: int, r: int, c: int) -> tuple[int, int, int, int]:
    """Return ROI bbox as (r0, c0, r1, c1) in patch coordinates."""
    h, w = ROI_WINDOW_SPECS[size_id]
    return r, c, r + h, c + w


def _roi_area(size_id: int) -> int:
    h, w = ROI_WINDOW_SPECS[size_id]
    return h * w


def _iou(roi_a: dict, roi_b: dict) -> float:
    a_r0, a_c0, a_r1, a_c1 = _roi_bbox(roi_a["size_id"], roi_a["r"], roi_a["c"])
    b_r0, b_c0, b_r1, b_c1 = _roi_bbox(roi_b["size_id"], roi_b["r"], roi_b["c"])
    inter_r0 = max(a_r0, b_r0)
    inter_c0 = max(a_c0, b_c0)
    inter_r1 = min(a_r1, b_r1)
    inter_c1 = min(a_c1, b_c1)
    inter_h = max(0, inter_r1 - inter_r0)
    inter_w = max(0, inter_c1 - inter_c0)
    inter = inter_h * inter_w
    if inter == 0:
        return 0.0
    area_a = _roi_area(roi_a["size_id"])
    area_b = _roi_area(roi_b["size_id"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def normalize_rois_with_wrapper(
    rois: Any,
    *,
    max_count: int = 2,
    iou_threshold: float = 0.6,
    grid_size: int = 37,
) -> tuple[list[dict], bool, bool]:
    """
    Normalize ROI list from MLLM output.

    Rules:
    - Clip size_id/r/c into legal range. If clipping happens -> roi_clipped=True.
    - If multiple ROIs have IoU > threshold, remove the larger one.
      If same area, keep the first and remove the later one.
      If dedup happens -> roi_deduped=True.
    - Limit result length to max_count.
    """
    roi_clipped = False
    roi_deduped = False

    if not isinstance(rois, list):
        return [], roi_clipped, roi_deduped

    normalized: list[dict] = []
    for item in rois:
        if not isinstance(item, dict):
            continue
        try:
            size_id_raw = int(item.get("size_id"))
            r_raw = int(item.get("r"))
            c_raw = int(item.get("c"))
        except Exception:
            continue

        size_id = min(max(size_id_raw, 0), 8)
        if size_id != size_id_raw:
            roi_clipped = True
        h, w = ROI_WINDOW_SPECS[size_id]
        r = min(max(r_raw, 0), grid_size - h)
        c = min(max(c_raw, 0), grid_size - w)
        if r != r_raw or c != c_raw:
            roi_clipped = True

        roi_dict = {"size_id": size_id, "r": r, "c": c}
        roi_type = item.get("roi_type")
        if roi_type in ("detection", "prior"):
            roi_dict["roi_type"] = roi_type
        normalized.append(roi_dict)

    keep = [True] * len(normalized)
    for i in range(len(normalized)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(normalized)):
            if not keep[j]:
                continue
            if _iou(normalized[i], normalized[j]) <= iou_threshold:
                continue
            area_i = _roi_area(normalized[i]["size_id"])
            area_j = _roi_area(normalized[j]["size_id"])
            if area_i > area_j:
                keep[i] = False
            elif area_j > area_i:
                keep[j] = False
            else:
                # same area: keep first
                keep[j] = False
            roi_deduped = True
            if not keep[i]:
                break

    deduped = [roi for roi, k in zip(normalized, keep) if k]
    if len(deduped) > max_count:
        deduped = deduped[:max_count]
        roi_deduped = True

    return deduped, roi_clipped, roi_deduped
