"""Tools for the v1.27 (MVTec LOCO) agent.

Re-exports the shared implementation from ``tools`` so that ``agent_v1_27`` and ``utils_v1_27``
import the same names.  The LOCO-specific "hard visual_check" counterfactual rules are NOT
implemented (not needed for evaluate_mvtec_v12); see ``generate_and_save_counterfactual_atomic_visual_checks``.
"""

from __future__ import annotations

from .tools import (  # noqa: F401
    IMAGE_PROCESSING_AVAILABLE,
    TOOLS,
    PromptOrder,
    _cv2_to_pil,
    _decode_base64_image,
    _encode_image_to_base64,
    _extract_image_from_messages,
    _keyword_heuristic_decision,
    _normalize_to_data_url,
    _pil_to_cv2,
    counterfactual_atomic_candidate_tool,
    image_brightness_enhancement,
    image_deblurring,
    image_denoising,
    image_super_resolution,
    image_zooming,
    to_pure_base64,
)


async def generate_and_save_counterfactual_atomic_visual_checks(*args, **kwargs):
    """Used only by evaluate_mvtecloco_v27.py with ``visual_check`` matching rules. Not implemented."""
    raise NotImplementedError(
        "visual_check counterfactual rules (MVTec LOCO) are not implemented in this reconstruction."
    )
