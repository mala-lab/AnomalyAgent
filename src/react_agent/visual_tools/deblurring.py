"""image_deblurring (paper Appendix A.1)."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ._common import IMAGE_PROCESSING_AVAILABLE, _clamp, _cv2_to_pil, _load_tool_image, _pil_to_cv2, _tool_result, cv2


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
