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

Layout (this module only re-exports, so the original imports ``from .tools import ...`` keep working):
  visual_tools/    the five general visual tools (+ shared image helpers)
  template_tools/  the general template tool and the counterfactual atomic candidate tool
"""

from __future__ import annotations

from enum import Enum

from .template_tools._common import (  # noqa: F401
    RUN_LOG,
    _CAPTION_ATTEMPTS,
    _embed_texts,
    _get_image_captions,
    _get_text,
    _l2norm,
    _raw_image_for_state,
    _readable_class,
    _state_get,
    _state_set,
)
from .template_tools.counterfactual import (  # noqa: F401
    _candidates_file_cache,
    _get_candidates,
    _lookup_candidates,
    counterfactual_atomic_candidate_tool,
)
from .template_tools.general import (  # noqa: F401
    _WINCLIP_ANOMALY_STATES,
    _WINCLIP_NORMAL_STATES,
    _WINCLIP_TEMPLATES,
    _class_prototypes,
    _keyword_heuristic_decision,
    _sigmoid,
    _winclip_prompts,
)
from .visual_tools._common import (  # noqa: F401
    _BASE64_RE,
    _DATA_URL_RE,
    _MAGIC_TO_MIME,
    _MAX_PIXELS,
    _b64decode_lenient,
    _clamp,
    _cv2_to_pil,
    _decode_base64_image,
    _encode_image_to_base64,
    _extract_image_from_messages,
    _load_tool_image,
    _message_image_urls,
    _normalize_to_data_url,
    _pil_to_cv2,
    _sniff_mime,
    _tool_result,
    to_pure_base64,
)
from .visual_tools.brightness import (  # noqa: F401
    image_brightness_enhancement,
)
from .visual_tools.deblurring import (  # noqa: F401
    image_deblurring,
)
from .visual_tools.denoising import (  # noqa: F401
    image_denoising,
)
from .visual_tools.super_resolution import (  # noqa: F401
    _SR_CACHE,
    _get_realesrgan_upsampler,
    image_super_resolution,
)
from .visual_tools.zooming import (  # noqa: F401
    image_zooming,
)
from .visual_tools import TOOLS  # noqa: F401
from .visual_tools._common import IMAGE_PROCESSING_AVAILABLE, cv2  # noqa: F401


# ============ PromptOrder ============

class PromptOrder(str, Enum):
    """Placeholder. Imported by agent_v1_16 / agent_v1_27 but never referenced there."""

    DEFAULT = "default"
