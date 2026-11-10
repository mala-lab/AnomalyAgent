"""image_denoising (paper Appendix A.1)."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ._common import IMAGE_PROCESSING_AVAILABLE, _clamp, _cv2_to_pil, _load_tool_image, _pil_to_cv2, _tool_result, cv2


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
