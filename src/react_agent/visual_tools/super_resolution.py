"""image_super_resolution (paper Appendix A.1), Real-ESRGAN when available."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from PIL import Image as PILImage

from ._common import IMAGE_PROCESSING_AVAILABLE, _clamp, _cv2_to_pil, _load_tool_image, _pil_to_cv2, _tool_result


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
