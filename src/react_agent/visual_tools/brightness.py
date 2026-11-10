"""image_brightness_enhancement (paper Appendix A.1)."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ._common import IMAGE_PROCESSING_AVAILABLE, _clamp, _cv2_to_pil, _load_tool_image, _pil_to_cv2, _tool_result, cv2


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
