"""image_zooming (paper Appendix A.1)."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from PIL import Image as PILImage

from ._common import _clamp, _load_tool_image, _tool_result


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
