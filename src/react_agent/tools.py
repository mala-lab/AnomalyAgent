"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, Tuple, Dict, cast, Annotated
from pathlib import Path
import os
import uuid
import base64
import io
import json
import re
import hashlib
import warnings
from datetime import UTC, datetime

from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.store.base import BaseStore
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime, get_runtime

# from react_agent.context import Context
from .context import Context
from .state import State
from .prompts import format_keyword_heuristic_report, format_keyword_heuristic_report_v2, PROMPT4CANDIDATE_GENERATION
from .utils import generate_and_save_atomic_candidates, safe_api_invoke


DEFAULT_REALESRGAN_MODEL_PATH = os.environ.get("REALESRGAN_MODEL_PATH", "./models/RealESRGAN_x4plus.pth")
DEFAULT_REALESRNET_MODEL_PATH = os.environ.get("REALESRNET_MODEL_PATH", "./models/RealESRNet_x4plus.pth")

# Image processing imports (must be before stable_sigmoid which uses np)
try:
    import cv2
    import numpy as np
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    cv2 = None
    np = None
    Image = None

# Stable sigmoid implementation (requires np, so after image processing imports)
try:
    from scipy.special import expit as stable_sigmoid, logsumexp
except ImportError:
    # Fallback: stable sigmoid implementation with piecewise approach for extreme values
    def stable_sigmoid(x):
        """Stable sigmoid implementation that prevents overflow for extreme values.
        
        Uses piecewise implementation:
        - For x >= 0: 1 / (1 + exp(-x)) - safe because exp(-x) E (0, 1], won't overflow
        - For x < 0: exp(x) / (1 + exp(x)) - safe because exp(x) E (0, 1), won't overflow
        This prevents overflow when x is very large or very small.
        """
        if np is None:
            raise ImportError("numpy is required for stable_sigmoid fallback")
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        pos = x >= 0
        neg = x < 0
        
        # x >= 0: safe because exp(-x) E (0, 1]
        if np.any(pos):
            out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        
        # x < 0: safe because exp(x) E (0, 1)
        if np.any(neg):
            exp_x = np.exp(x[neg])
            out[neg] = exp_x / (1.0 + exp_x)
        
        # return scalar if scalar input
        return out.item() if out.ndim == 0 else out
    
    # Fallback logsumexp implementation
    def logsumexp(a, axis=None, keepdims=False):
        """Stable log-sum-exp implementation."""
        if np is None:
            raise ImportError("numpy is required for logsumexp fallback")
        a = np.asarray(a)
        a_max = np.max(a, axis=axis, keepdims=True)
        if not keepdims:
            a_max = np.squeeze(a_max, axis=axis)
        result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
        return result.item() if result.ndim == 0 else result

# Real-ESRGAN for super-resolution (optional, lazy import)
try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANer = None
    SRVGGNetCompact = None

@tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

@tool
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"

# ============ Image Processing Utilities ============

def _extract_image_from_messages(messages: List[Any]) -> Optional[str]:
    """Extract base64 image string from messages.
    
    Searches through messages to find image data in various formats:
    - Base64 data URL in image_url format
    - PIL Image objects (converts to base64)
    
    Args:
        messages: List of message objects (from state.messages).
        
    Returns:
        Base64 encoded image string with data URL prefix, or None if not found.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return None
    
    # Search through messages in reverse (most recent first)
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content is None:
            continue
        
        # Handle list content (multimodal)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Check for image_url format
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            return url
                    # Check for PIL Image object
                    elif item.get("type") == "image" and "image" in item:
                        pil_img = item["image"]
                        if Image and isinstance(pil_img, Image.Image):
                            return _encode_image_to_base64(pil_img)
                # Check if item is PIL Image directly
                elif Image and isinstance(item, Image.Image):
                    return _encode_image_to_base64(item)
        
        # Handle string content that might contain base64
        elif isinstance(content, str) and "data:image" in content:
            # Try to extract base64 from string
            import re
            match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
            if match:
                return match.group(0)
    
    return None


def _decode_base64_image(image_base64: str) -> Optional[Image.Image]:
    """Decode base64 image string to PIL Image.
    
    Args:
        image_base64: Base64 encoded image string, optionally with data URL prefix.
        
    Returns:
        PIL Image object or None if decoding fails.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return None
    
    # Set MAX_IMAGE_PIXELS to allow very large images (up to 1 gigapixel)
    # This must be set before Image.open() to take effect
    old_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1024 * 1024 * 1024  # 1 gigapixel (32768 x 32768)
    
    try:
        # Remove data URL prefix only when it's a valid data URL
        if image_base64.startswith("data:image") and "base64," in image_base64:
            image_base64 = image_base64.split("base64,", 1)[1]
        cleaned = image_base64.strip()
        cleaned += "=" * (-len(cleaned) % 4)
        # Decode base64 with validation (Python 3.9+)
        image_data = base64.b64decode(cleaned, validate=True)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = old_max_pixels


def _encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object.
        format: Image format (PNG, JPEG, etc.).
        
    Returns:
        Base64 encoded image string with data URL prefix.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return ""
    
    try:
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_base64}"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return ""


def _normalize_to_data_url(image_data: str, default_format: str = "png") -> Optional[str]:
    """
    Normalize image data to a valid data URL format.
    
    This function handles various input formats:
    - Already a data URL (e.g., "data:image/png;base64,...") -> return as-is
    - Raw base64 string -> wrap it as "data:image/{format};base64,{base64}"
    - Empty or invalid -> return None
    
    Args:
        image_data: Image data string (can be data URL or raw base64)
        default_format: Default image format if not specified (default: "png")
        
    Returns:
        Normalized data URL string, or None if input is invalid
    """
    if not image_data or not isinstance(image_data, str):
        return None
    
    def _detect_format(decoded: bytes) -> Optional[str]:
        if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if decoded.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if decoded.startswith(b"GIF"):
            return "gif"
        if decoded.startswith(b"RIFF") and b"WEBP" in decoded[:16]:
            return "webp"
        return None

    def _safe_b64decode(raw: str) -> Optional[bytes]:
        cleaned = raw.strip()
        cleaned += "=" * (-len(cleaned) % 4)
        try:
            return base64.b64decode(cleaned, validate=True)
        except Exception:
            return None

    # Already a data URL: validate and ensure it's actually an image
    if image_data.startswith("data:image"):
        try:
            _, encoded = image_data.split(",", 1)
        except ValueError:
            return None
        decoded = _safe_b64decode(encoded)
        if not decoded:
            return None
        fmt = _detect_format(decoded)
        if not fmt:
            return None
        return image_data
    
    # Try to detect if it's a raw base64 string
    # Base64 strings are typically long and contain only base64 characters
    import re
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
    
    # Remove whitespace
    cleaned = image_data.strip()
    
    # Check if it looks like base64 (at least 20 chars, only base64 chars)
    # Lower threshold to catch smaller images or partial base64 strings
    if len(cleaned) >= 20 and base64_pattern.match(cleaned):
        decoded = _safe_b64decode(cleaned)
        if not decoded:
            return None
        format_type = _detect_format(decoded)
        if not format_type:
            return None
        return f"data:image/{format_type};base64,{cleaned}"
    
    # Doesn't look like valid image data
    return None


def to_pure_base64(s: str | None) -> str | None:
    """
    Convert image data (data URL or raw base64) to pure base64 string.
    
    This is the inverse of _normalize_to_data_url: it strips the data URL header
    if present, returning only the base64 payload.
    
    Use this when passing images to tools that expect raw base64 (not data URLs).
    
    Args:
        s: Image data string (data URL like "data:image/...;base64,..." or raw base64)
    
    Returns:
        Pure base64 string without data URL prefix, or None if input is invalid
    
    Example:
        >>> to_pure_base64("data:image/png;base64,iVBORw0KGgo...")
        "iVBORw0KGgo..."
        >>> to_pure_base64("iVBORw0KGgo...")  # already pure base64
        "iVBORw0KGgo..."
    """
    if not isinstance(s, str) or not s:
        return None
    s = s.strip()
    
    # If it's a data URL, extract the base64 part after the comma
    if s.startswith("data:image"):
        if "base64," in s:
            return s.split("base64,", 1)[1]
        return None  # Malformed data URL
    
    # Already looks like pure base64 (best-effort check)
    # Minimum length check to avoid treating error messages as base64
    if len(s) > 100:
        return s
    
    return None


def _pil_to_cv2(pil_image: Image.Image) -> Optional[Any]:
    """Convert PIL Image to OpenCV format (numpy array).
    
    Args:
        pil_image: PIL Image object.
        
    Returns:
        OpenCV image (numpy array in BGR format) or None.
    """
    if not IMAGE_PROCESSING_AVAILABLE or cv2 is None or np is None:
        return None
    
    try:
        # Convert PIL RGB to OpenCV BGR
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        print(f"Error converting PIL to OpenCV: {e}")
        return None


def _cv2_to_pil(cv_image: Any) -> Optional[Image.Image]:
    """Convert OpenCV image to PIL Image.
    
    Args:
        cv_image: OpenCV image (numpy array in BGR format).
        
    Returns:
        PIL Image object or None.
    """
    if not IMAGE_PROCESSING_AVAILABLE or cv2 is None or np is None:
        return None
    
    try:
        # Convert OpenCV BGR to PIL RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image
    except Exception as e:
        print(f"Error converting OpenCV to PIL: {e}")
        return None


# ============ Image Processing Tools ============

@tool
async def image_denoising(
    image_base64: Optional[str] = None,
    h: float = 10.0,
    h_color: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> str:
    """Apply image denoising using OpenCV's fast non-local means filter.
    
    This tool reduces color and spatial noise in images, enhancing clarity
    in low-light or noisy environments.
    
    IMPORTANT: The image_base64 parameter is optional. If not provided, the tool
    will automatically attempt to extract the image from the conversation context.
    You can simply call this tool without any parameters to process the image in the conversation.
    
    Args:
        image_base64: (Optional) Base64 encoded image string with data URL prefix.
                     If not provided, the tool will attempt to extract the image from the
                     conversation messages. Example: "data:image/png;base64,iVBORw0KGgo..."
        h: Filter strength for luminance component. Higher values remove more noise
           but may blur details. Default: 10.0
        h_color: Filter strength for color components. Default: 10.0
        template_window_size: Size of the template patch for comparison. Default: 7
        search_window_size: Size of the search window. Default: 21
        
    Returns:
        Success message indicating denoising was applied. The processed image is automatically
        available in the conversation context for further analysis.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: Image processing libraries (OpenCV, PIL) not available. Please install: pip install opencv-python pillow numpy"
    
    # If image_base64 not provided, try to extract from conversation messages (fallback mechanism)
    # This ensures the function behavior matches its docstring promise of automatic extraction
    if not image_base64:
        try:
            # Try to get runtime context and extract image from state messages
            # This is a fallback - normally tools_node_with_image_injection handles this before calling
            runtime = get_runtime(Context)
            if runtime and hasattr(runtime, 'state') and runtime.state is not None:
                messages = getattr(runtime.state, 'messages', [])
                if messages:
                    extracted_image = _extract_image_from_messages(messages)
                    if extracted_image:
                        image_base64 = extracted_image
        except (RuntimeError, AttributeError, ImportError, Exception):
            # Runtime not available or extraction failed - continue to check if image_base64 is still None
            # This is expected in some execution contexts where runtime is not accessible
            pass
    
    # If still no image after all extraction attempts, return helpful error message
    if not image_base64:
        return "Error: No image provided. The image_base64 parameter was not provided, and the tool could not automatically extract an image from the conversation messages. Please ensure an image is present in the conversation context, or explicitly provide the image_base64 parameter."
    
    # Decode image
    pil_image = _decode_base64_image(image_base64)
    if pil_image is None:
        return "Error: Failed to decode base64 image."
    
    try:
        # Convert to OpenCV format
        cv_image = _pil_to_cv2(pil_image)
        if cv_image is None:
            return "Error: Failed to convert image to OpenCV format."
        
        # Apply fast non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            cv_image,
            None,
            h=h,
            hColor=h_color,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size,
        )
        
        # Convert back to PIL and encode
        denoised_pil = _cv2_to_pil(denoised)
        if denoised_pil is None:
            return "Error: Failed to convert denoised image back to PIL format."
        
        # Encode processed image to base64
        processed_base64 = _encode_image_to_base64(denoised_pil)
        
        # Return JSON string with text message and processed image
        # This will be parsed by ToolNode to create multimodal ToolMessage
        import json
        return json.dumps({
            "text": f"Image denoising completed successfully. Parameters: h={h}, h_color={h_color}. The denoised image is included below.",
            "processed_image_base64": processed_base64
        })
        
    except Exception as e:
        return f"Error during denoising: {str(e)}"


@tool
async def image_deblurring(
    image_base64: str,
    sigma: float = 1.0,
    strength: float = 1.5,
    threshold: int = 0,
) -> str:
    """Apply image deblurring using unsharp masking with Gaussian blur subtraction.
    
    This tool sharpens edge details for motion blur or out-of-focus issues,
    improving perceptual clarity without retraining.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URL prefix).
        sigma: Standard deviation for Gaussian blur kernel. Default: 1.0
        strength: Strength of the sharpening effect. Higher values = more sharpening. Default: 1.5
        threshold: Minimum brightness change to apply sharpening. Default: 0
        
    Returns:
        Success message indicating deblurring was applied. The processed image is automatically
        available in the conversation context for further analysis.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: Image processing libraries (OpenCV, PIL) not available. Please install: pip install opencv-python pillow numpy"
    
    # Decode image
    pil_image = _decode_base64_image(image_base64)
    if pil_image is None:
        return "Error: Failed to decode base64 image."
    
    try:
        # Convert to OpenCV format
        cv_image = _pil_to_cv2(pil_image)
        if cv_image is None:
            return "Error: Failed to convert image to OpenCV format."
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(cv_image, (0, 0), sigma)
        
        # Unsharp masking: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(cv_image, 1.0 + strength, blurred, -strength, 0)
        
        # Apply threshold if specified
        if threshold > 0:
            diff = cv2.absdiff(cv_image, blurred)
            mask = diff > threshold
            result = np.where(mask, sharpened, cv_image)
        else:
            result = sharpened
        
        # Convert back to PIL and encode
        deblurred_pil = _cv2_to_pil(result)
        if deblurred_pil is None:
            return "Error: Failed to convert deblurred image back to PIL format."
        
        # Encode processed image to base64
        processed_base64 = _encode_image_to_base64(deblurred_pil)
        
        # Return JSON string with text message and processed image
        import json
        return json.dumps({
            "text": f"Image deblurring completed successfully. Parameters: sigma={sigma}, strength={strength}, threshold={threshold}. The deblurred image is included below.",
            "processed_image_base64": processed_base64
        })
        
    except Exception as e:
        return f"Error during deblurring: {str(e)}"


@tool
async def image_super_resolution(
    image_base64: str,
    scale: int = 4,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
) -> str:
    """Apply image super-resolution using Real-ESRGAN or RealESRNet model.
    
    This tool improves detail preservation and restores textures in low-resolution
    images using a deep RRDBNet-based super-resolution pipeline.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URL prefix).
        scale: Upscaling factor (2, 4, or 8). Default: 4
        model_name: Model name. If None, uses default based on context configuration.
                    Options: 'realesrgan-x4plus', 'realesrgan-x4plus-anime', 'realesrnet-x4plus'.
        model_path: Optional path to local model file (.pth). If provided, uses this model.
                   If None, uses default path based on context configuration (GAN or non-GAN version).
        
    Returns:
        Success message indicating super-resolution was applied. The processed image is automatically
        available in the conversation context for further analysis.
    """
    def _debug_return(val):
        """Debug helper: print return type and first 80 chars."""
        val_str = str(val)
        print(f"[DEBUG image_super_resolution] return type={type(val).__name__}, content[:80]={val_str[:80]!r}")
        return val
    
    if not IMAGE_PROCESSING_AVAILABLE:
        return _debug_return("Error: Image processing libraries (OpenCV, PIL) not available. Please install: pip install opencv-python pillow numpy")
    
    # Debug: print input image_base64 info
    input_len = len(image_base64) if isinstance(image_base64, str) else None
    input_prefix = image_base64[:100] if isinstance(image_base64, str) else str(image_base64)[:100]
    print(f"[DEBUG image_super_resolution] INPUT: type={type(image_base64).__name__}, len={input_len}, prefix={input_prefix!r}")
    
    # Decode image
    pil_image = _decode_base64_image(image_base64)
    if pil_image is None:
        return _debug_return("Error: Failed to decode base64 image.")
    
    try:
        # Convert to OpenCV format
        cv_image = _pil_to_cv2(pil_image)
        if cv_image is None:
            return _debug_return("Error: Failed to convert image to OpenCV format.")
        
        height, width = cv_image.shape[:2]
        
        # Estimate output image size and adjust scale if needed to avoid API limits
        # Base64 encoding increases size by ~33%, and we need to account for compression
        # Conservative limit: 5MB for base64 encoded image (API typically allows 5MB)
        MAX_BASE64_SIZE = 5 * 1024 * 1024  # 5MB
        
        # Estimate original binary size (rough estimate: pixels * channels * bytes_per_pixel)
        # For RGB images: width * height * 3 bytes per pixel (uncompressed)
        # We use a conservative compression ratio of 0.3 (upper bound, typical JPEG is 0.05~0.2)
        raw_size = width * height * 3
        compression_ratio = 0.3  # Conservative upper bound for JPEG/PNG compression
        estimated_binary_size = raw_size * compression_ratio
        
        # Estimate upscaled size: binary_size * scale^2 (area scales quadratically)
        # Then add 33% for base64 encoding
        estimated_upscaled_binary = estimated_binary_size * (scale ** 2)
        estimated_base64_size = estimated_upscaled_binary * 1.33
        
        # Hard limit on output dimensions to prevent extremely large images
        MAX_OUTPUT_EDGE = 4096  # Maximum edge length after upscaling
        MAX_OUTPUT_PIXELS = 16 * 1024 * 1024  # Maximum total pixels (16 megapixels)
        
        # Allowed scale values for RealESRGAN (discrete steps)
        ALLOWED_SCALES = [8, 4, 2, 1]
        
        # Adjust scale if estimated size exceeds limit
        original_scale = scale
        if estimated_base64_size > MAX_BASE64_SIZE and scale > 1:
            # Try to find a safe scale from allowed values
            for test_scale in [s for s in ALLOWED_SCALES if s < scale]:
                test_binary = estimated_binary_size * (test_scale ** 2)
                test_base64 = test_binary * 1.33
                if test_base64 <= MAX_BASE64_SIZE:
                    scale = test_scale
                    print(f"[WARNING] Estimated upscaled image size ({estimated_base64_size / 1024 / 1024:.2f}MB) exceeds limit ({MAX_BASE64_SIZE / 1024 / 1024}MB). Reducing scale from {original_scale}x to {scale}x.")
                    break
            else:
                # Even scale=1 exceeds limit, bypass upscaling
                scale = 1
                print(f"[WARNING] Even scale=1 exceeds estimated limit. Bypassing super-resolution.")
        
        # Apply hard limit on output dimensions
        new_height_candidate = height * scale
        new_width_candidate = width * scale
        
        # Check if output exceeds dimension limits
        if max(new_height_candidate, new_width_candidate) > MAX_OUTPUT_EDGE or \
           new_height_candidate * new_width_candidate > MAX_OUTPUT_PIXELS:
            # Calculate maximum safe scale
            max_scale_edge = MAX_OUTPUT_EDGE / max(height, width)
            max_scale_pixels = (MAX_OUTPUT_PIXELS / (height * width)) ** 0.5
            safe_scale_float = min(max_scale_edge, max_scale_pixels, scale)
            
            if safe_scale_float < 1.0:
                # Image is already too large, don't upscale at all
                print(f"[WARNING] Input image ({width}x{height}) already exceeds safe limits. Skipping upscaling.")
                return f"Warning: Input image ({width}x{height}) is already large. Super-resolution skipped to avoid memory issues. The original image is preserved."
            
            # Pick the largest allowed scale that is <= safe_scale_float
            new_scale = 1
            for s in ALLOWED_SCALES:
                if s <= safe_scale_float:
                    new_scale = s
                    break
            
            if new_scale < scale:
                print(f"[WARNING] Output would be {new_width_candidate}x{new_height_candidate} ({new_width_candidate * new_height_candidate / 1e6:.1f}MP), exceeds limits. Reducing scale from {scale}x to {new_scale}x.")
                scale = new_scale
        
        # If scale is 1, bypass RealESRGAN entirely (no point in running the model)
        if scale <= 1:
            print(f"[INFO] scale={scale}, bypassing super-resolution. Returning original image.")
            # Encode original image directly and return as base64 (not text message)
            output_base64 = _encode_image_to_base64(pil_image)
            if output_base64:
                try:
                    runtime = get_runtime(Context)
                    if runtime and hasattr(runtime, 'context') and runtime.context is not None:
                        runtime.context.augmented_image_url = output_base64
                except Exception:
                    pass
                # Return with BYPASS: prefix so tool_node knows image was not modified
                return f"BYPASS:{output_base64}"
            return "Error: Failed to encode original image."
        
        new_height, new_width = height * scale, width * scale
        
        # Determine model path and name from context if not explicitly provided
        if model_path is None:
            try:
                runtime = get_runtime(Context)
                if runtime and hasattr(runtime, 'context') and runtime.context is not None:
                    version = getattr(runtime.context, 'super_resolution_version', 'nongan')
                    version_lower = version.lower() if isinstance(version, str) else 'nongan'
                    if version_lower == 'gan':
                        model_path = DEFAULT_REALESRGAN_MODEL_PATH
                        if model_name is None:
                            model_name = "realesrgan-x4plus"
                    else:  # non-GAN version (default, including invalid values)
                        model_path = DEFAULT_REALESRNET_MODEL_PATH
                        if model_name is None:
                            model_name = "realesrnet-x4plus"
                else:
                    # Runtime context not available, use non-GAN version as default
                    model_path = DEFAULT_REALESRNET_MODEL_PATH
                    if model_name is None:
                        model_name = "realesrnet-x4plus"
            except (RuntimeError, AttributeError, ImportError, Exception):
                # If runtime not available, use non-GAN version as default
                model_path = DEFAULT_REALESRNET_MODEL_PATH
                if model_name is None:
                    model_name = "realesrnet-x4plus"
        
        # Track which method is used for debug info
        method_used = None
        
        if not REALESRGAN_AVAILABLE:
            # Fallback to bicubic interpolation if Real-ESRGAN is not available
            # Note: For production use, install Real-ESRGAN: pip install realesrgan
            upscaled = cv2.resize(
                cv_image,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC,
            )
            method_used = "naive (bicubic)"
        else:
            # Real-ESRGAN/RealESRNet branch: Use deep learning-based super-resolution
            try:
                # Create Real-ESRGAN upsampler (works for both GAN and non-GAN models)
                # Model management:
                # - If model_path is provided: uses the specified local model file
                # - Real-ESRGAN library can handle both RealESRGAN and RealESRNet models
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=model_path,  # Path to either GAN or non-GAN model
                    # Additional optional parameters:
                    # tile=0,  # 0 means no tiling (process whole image at once)
                    # tile_pad=10,  # Padding for tiling
                    # pre_pad=0,  # Pre-padding size
                    # half=False,  # Use FP32 (set to True for FP16 if GPU available)
                )
                
                # Apply super-resolution
                # enhance() returns (output_image, output_path)
                upscaled, _ = upsampler.enhance(cv_image, outscale=scale)
                # Determine method name from model path
                if "RealESRNet" in str(model_path):
                    method_used = "RealESRNet (non-GAN)"
                else:
                    method_used = "Real-ESRGAN (GAN)"
                
            except Exception as e:
                # Fallback to bicubic if model loading fails
                print(f"Super-resolution model failed, falling back to bicubic: {e}")
                upscaled = cv2.resize(
                    cv_image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                method_used = f"naive (bicubic, model failed: {str(e)[:50]})"
        
        # Convert back to PIL and encode
        upscaled_pil = _cv2_to_pil(upscaled)
        if upscaled_pil is None:
            return _debug_return("Error: Failed to convert upscaled image back to PIL format.")
        
        # Encode processed image to base64
        processed_base64 = _encode_image_to_base64(upscaled_pil)
        
        # Return JSON string with text message and processed image
        import json
        result = json.dumps({
            "text": f"Image super-resolution completed successfully. Scale: {scale}x, Output size: {new_width}x{new_height} (from {width}x{height}). Method: {method_used}. The upscaled image is included below.",
            "processed_image_base64": processed_base64
        })
        return _debug_return(result)
        
    except Exception as e:
        return _debug_return(f"Error during super-resolution: {str(e)}")


@tool
async def image_zooming(
    image_base64: str,
    scale: float = 2.0,
    method: str = "bicubic",
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
) -> str:
    """Apply image zooming using bicubic interpolation-based magnification.
    
    This tool magnifies regions for enhanced spatial detail, useful for detecting
    small-scale interactions or distant activities.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URL prefix).
        scale: Zoom factor (e.g., 2.0 = 2x zoom). Default: 2.0
        method: Interpolation method. Options: 'bicubic', 'bilinear', 'nearest', 'lanczos'.
                Default: 'bicubic'
        center_x: X coordinate of zoom center (0.0-1.0, normalized). If None, centers on image center.
        center_y: Y coordinate of zoom center (0.0-1.0, normalized). If None, centers on image center.
        crop_width: Width of region to crop before zooming (pixels). If None, uses full image.
        crop_height: Height of region to crop before zooming (pixels). If None, uses full image.
        
    Returns:
        Success message indicating zooming was applied. The processed image is automatically
        available in the conversation context for further analysis.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: Image processing libraries (OpenCV, PIL) not available. Please install: pip install opencv-python pillow numpy"
    
    # Decode image
    pil_image = _decode_base64_image(image_base64)
    if pil_image is None:
        return "Error: Failed to decode base64 image."
    
    try:
        # Convert to OpenCV format
        cv_image = _pil_to_cv2(pil_image)
        if cv_image is None:
            return "Error: Failed to convert image to OpenCV format."
        
        height, width = cv_image.shape[:2]
        
        # Determine interpolation method
        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interpolation = interpolation_map.get(method.lower(), cv2.INTER_CUBIC)
        
        # Crop region if specified
        if crop_width is not None and crop_height is not None:
            # Determine crop center
            if center_x is None:
                center_x = 0.5
            if center_y is None:
                center_y = 0.5
            
            # Calculate crop coordinates
            x = int((center_x * width) - crop_width / 2)
            y = int((center_y * height) - crop_height / 2)
            
            # Clamp to image bounds
            x = max(0, min(x, width - crop_width))
            y = max(0, min(y, height - crop_height))
            
            # Crop
            cropped = cv_image[y:y+crop_height, x:x+crop_width]
            crop_info = f"Cropped region: ({x},{y}) to ({x+crop_width},{y+crop_height})"
        else:
            cropped = cv_image
            crop_info = "Full image"
        
        # Apply zoom (resize)
        new_width = int(cropped.shape[1] * scale)
        new_height = int(cropped.shape[0] * scale)
        zoomed = cv2.resize(cropped, (new_width, new_height), interpolation=interpolation)
        
        # Convert back to PIL and encode
        zoomed_pil = _cv2_to_pil(zoomed)
        if zoomed_pil is None:
            return "Error: Failed to convert zoomed image back to PIL format."
        
        # Encode processed image to base64
        processed_base64 = _encode_image_to_base64(zoomed_pil)
        
        # Return JSON string with text message and processed image
        import json
        return json.dumps({
            "text": f"Image zooming completed successfully. Scale: {scale}x, Method: {method}, {crop_info}, Output size: {new_width}x{new_height}. The zoomed image is included below.",
            "processed_image_base64": processed_base64
        })
        
    except Exception as e:
        return f"Error during zooming: {str(e)}"


@tool
async def image_brightness_enhancement(
    image_base64: str,
    clip_limit: float = 2.0,
    tile_grid_rows: int = 8,
    tile_grid_cols: int = 8,
) -> str:
    """Apply brightness enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    PANDA uses OpenCV's CLAHE on the L-channel in LAB color space. This ensures localized
    brightness normalization for dimly lit or overexposed frames.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URL prefix).
        clip_limit: Threshold for contrast limiting. Higher values increase contrast.
                   Default: 2.0
        tile_grid_rows: Number of rows in the grid for histogram equalization. Default: 8.
        tile_grid_cols: Number of columns in the grid for histogram equalization. Default: 8.
                       Smaller tiles provide more localized enhancement.
        
    Returns:
        Success message indicating brightness enhancement was applied. The processed image
        is automatically available in the conversation context for further analysis.
    """
    if not IMAGE_PROCESSING_AVAILABLE:
        return "Error: Image processing libraries (OpenCV, PIL) not available. Please install: pip install opencv-python pillow numpy"
    
    # Decode image
    pil_image = _decode_base64_image(image_base64)
    if pil_image is None:
        return "Error: Failed to decode base64 image."
    
    try:
        # Convert to OpenCV format
        cv_image = _pil_to_cv2(pil_image)
        if cv_image is None:
            return "Error: Failed to convert image to OpenCV format."
        
        # Convert BGR to LAB color space
        lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels: L (lightness), A, B
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Apply CLAHE to L channel only
        tile_grid_size = (tile_grid_rows, tile_grid_cols)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_enhanced = clahe.apply(l_channel)
        
        # Merge enhanced L channel with original A and B channels
        lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL and encode
        enhanced_pil = _cv2_to_pil(enhanced_image)
        if enhanced_pil is None:
            return "Error: Failed to convert enhanced image back to PIL format."
        
        # Encode processed image to base64
        processed_base64 = _encode_image_to_base64(enhanced_pil)
        
        # Return JSON string with text message and processed image
        import json
        return json.dumps({
            "text": f"Image brightness enhancement completed successfully. CLAHE applied to L-channel in LAB color space. Clip limit: {clip_limit}, Tile grid size: ({tile_grid_rows}, {tile_grid_cols}). The enhanced image is included below.",
            "processed_image_base64": processed_base64
        })
        
    except Exception as e:
        return f"Error during brightness enhancement: {str(e)}"


# TOOLS = [search, upsert_memory, image_denoising, image_deblurring, image_super_resolution, image_zooming]
TOOLS = [image_denoising, image_deblurring, image_super_resolution, image_zooming, image_brightness_enhancement]
# TOOLS = [image_deblurring, image_super_resolution, image_zooming]
# TOOLS: List[Callable[..., Any]] = [search]


# ============ Keyword Heuristic Utilities ============

# Helper functions for keyword heuristic decision
JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})",
    re.IGNORECASE,
)

def _safe_json_loads(candidate: str) -> Dict[str, Any]:
    """Best-effort JSON parsing with simple trimming fallback."""
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except Exception:
        # Last resort: strip backticks and whitespace
        try:
            cleaned = candidate.strip("` \n\t")
            return json.loads(cleaned)
        except Exception:
            return {}

def _parse_json_from_text(text: str | None) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from the model text.
    Falls back to {} if nothing valid is found.
    """
    if not text:
        return {}
    match = JSON_BLOCK_RE.search(text)
    if match:
        candidate = match.group(1) or match.group(2)
    else:
        candidate = text.strip()
    if not candidate:
        return {}
    return _safe_json_loads(candidate)

def _sys_msg(system_prompt: str) -> Dict[str, str]:
    return {"role": "system", "content": system_prompt}

def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()

def _get_text(content: Any) -> str:
    """Normalize AIMessage.content into string.

    Supports both legacy string content and Responses API content blocks like:
    [{"type": "text", "text": "..."}]
    """
    if isinstance(content, str):
        return content

    def _collect_text_fields(value: Any) -> List[str]:
        fragments: List[str] = []
        if isinstance(value, dict):
            text_value = value.get("text")
            if isinstance(text_value, str) and text_value:
                fragments.append(text_value)
            for child in value.values():
                fragments.extend(_collect_text_fields(child))
        elif isinstance(value, list):
            for child in value:
                fragments.extend(_collect_text_fields(child))
        return fragments

    fragments = _collect_text_fields(content)
    if fragments:
        return "\n".join(fragment for fragment in fragments if fragment)

    return ""


class PromptOrder:
    """Generate template statements for normal and anomalous objects."""
    
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]
        
        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]
        
        self.template_list = [
            "a cropped photo of a {}.",
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of a {}.",
            "a dark photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of a {}.",
            "a blurry photo of a {}.",
            "a blurry photo of the {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]
    
    def prompt(self, class_name: str) -> Tuple[List[str], List[str]]:
        """Generate normal and anomalous template statements.
        
        Args:
            class_name: The name of the object class (e.g., "bottle", "carpet").
            
        Returns:
            Tuple of (normal_statements, anomaly_statements).
        """
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [
            class_template.format(ele) 
            for ele in class_state 
            for class_template in self.template_list
        ]
        
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [
            class_template.format(ele) 
            for ele in class_state 
            for class_template in self.template_list
        ]
        
        return normal_ensemble_template, anomaly_ensemble_template


async def _keyword_heuristic_decision(
    state: State, 
    runtime: Runtime[Context],
    decision: bool = False
) -> Tuple[str, float, float, float, str]:
    """
    Use embedding-based prototype matching to determine if sample is anomalous or normal.
    
    Process:
    1. Generate template statements for normal and anomalous objects using class_name
    2. Embed all template statements and compute prototypes (average embeddings) for each class
    3. Generate three descriptions of current sample using Qwen VL model
    4. Embed each description
    5. For each description, calculate score using prototype-based comparison:
       - s_n = cos(c, p_n) where p_n is normal prototype
       - s_a = cos(c, p_a) where p_a is anomalous prototype
       - logit = (s_a - s_n) / tau
       - score = sigmoid(logit)
    6. Return three scores
    
    Args:
        decision: If False, return only reason in the last element. If True, return only predicted_label in the first element.
                 Default is False.
    
    Returns:
        (label, score1, score2, score3, reason_suffix) where label is "anomalous" or "normal",
        and score1, score2, score3 are the three scores from the three descriptions.
        When decision=False: reason_suffix contains the full report, other values are still computed.
        When decision=True: label contains only the predicted_label, other values are still computed.
    """
    ctx = runtime.context
    
    # Get class_name from state (handle both dict and State object)
    if isinstance(state, dict):
        class_name = state.get("class_name")
    else:
        class_name = getattr(state, "class_name", None)
    
    if not class_name:
        return "anomalous", 0.5, 0.5, 0.5, "No class_name in state, defaulting to anomalous."
    
    # Get embedding model
    embedding_model = ctx.embedding_model
    if embedding_model is None:
        return "anomalous", 0.5, 0.5, 0.5, "Embedding model not available, defaulting to anomalous."
    
    # Temperature parameter for logit calculation (from context, configurable)
    # Validate and clamp tau to prevent numerical explosion
    tau = max(ctx.tau, 1e-6)  # Clamp to minimum to prevent division by zero
    if ctx.tau < 1e-6:
        warnings.warn(f"tau={ctx.tau} is too small, clamped to 1e-6 for numerical stability")
    
    # Generate prompt lists and hash for version
    prompt_order = PromptOrder()
    normal_statements, anomaly_statements = prompt_order.prompt(class_name)
    # Hash prompt lists to generate version (automatically updates when prompts change)
    prompt_content = json.dumps([normal_statements, anomaly_statements], sort_keys=True)
    prompt_version = hashlib.md5(prompt_content.encode()).hexdigest()[:8]  # Use first 8 chars
    
    embedding_model_id = ctx.embedding_model_path or "default"
    # Get embedding model fingerprint for better cache key uniqueness
    # Try to get model identifier from embedding_model object
    try:
        model_fingerprint = getattr(embedding_model, 'model_name', None) or \
                           getattr(embedding_model, '__class__.__name__', None) or \
                           str(type(embedding_model).__name__)
    except:
        model_fingerprint = "unknown"
    # Create a simple fingerprint hash
    model_fingerprint_hash = hashlib.md5(str(model_fingerprint).encode()).hexdigest()[:8]
    
    # Check cache for prototypes
    # Strategy: Try to get embedding_dim from cache first (if matching entry exists)
    # Only call sample embedding if cache miss (to avoid unnecessary embedding call)
    if not hasattr(ctx, '_prototype_cache'):
        ctx._prototype_cache = {}
    
    # Try to find embedding_dim from cache (fast path: if we have a matching entry)
    embedding_dim = None
    partial_key_prefix = f"{class_name}_{embedding_model_id}_{prompt_version}_{model_fingerprint_hash}_"
    # Quick scan: if cache is small, find matching entry to get embedding_dim
    # Only do this if cache is reasonably small (< 100 entries) to avoid performance hit
    if len(ctx._prototype_cache) < 100:
        for key in ctx._prototype_cache.keys():
            if key.startswith(partial_key_prefix):
                # Extract embedding_dim from key (format: ..._dim)
                try:
                    embedding_dim = int(key.split('_')[-1])
                    break
                except (ValueError, IndexError):
                    continue
    
    # If embedding_dim not found in cache, get it from sample embedding (cache miss path)
    if embedding_dim is None:
        sample_statement = normal_statements[0] if normal_statements else "sample"
        sample_embedding = await embedding_model.aembed_query(sample_statement)
        embedding_dim = len(sample_embedding)
    
    # Build exact cache key with all components including model fingerprint
    cache_key = f"{class_name}_{embedding_model_id}_{prompt_version}_{model_fingerprint_hash}_{embedding_dim}"
    
    # Use exact match for cache lookup (fast O(1) dict lookup)
    if cache_key in ctx._prototype_cache:
        # Cache hit: use cached prototypes
        p_n, p_a = ctx._prototype_cache[cache_key]
    else:
        # Compute prototypes (cache miss)
        # Embed all template statements
        normal_embeddings = await embedding_model.aembed_documents(normal_statements)
        anomaly_embeddings = await embedding_model.aembed_documents(anomaly_statements)
        
        # Compute prototypes (average embeddings) for each class
        # Use np.asarray with dtype=np.float32 to prevent dtype=object issues
        normal_embeddings_array = np.asarray(normal_embeddings, dtype=np.float32)
        anomaly_embeddings_array = np.asarray(anomaly_embeddings, dtype=np.float32)
        
        # Validate shapes are consistent
        if normal_embeddings_array.ndim != 2 or anomaly_embeddings_array.ndim != 2:
            raise ValueError(f"Embeddings must be 2D arrays. Got shapes: {normal_embeddings_array.shape}, {anomaly_embeddings_array.shape}")
        if normal_embeddings_array.shape[1] != anomaly_embeddings_array.shape[1]:
            raise ValueError(f"Embedding dimensions must match. Got: {normal_embeddings_array.shape[1]} vs {anomaly_embeddings_array.shape[1]}")
        if normal_embeddings_array.shape[1] != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {normal_embeddings_array.shape[1]}")
        
        # Calculate mean (prototype) for each class
        p_n = np.mean(normal_embeddings_array, axis=0)  # Normal prototype
        p_a = np.mean(anomaly_embeddings_array, axis=0)  # Anomalous prototype
        
        # Normalize prototypes
        p_n_norm = np.linalg.norm(p_n)
        p_a_norm = np.linalg.norm(p_a)
        if p_n_norm > 0:
            p_n = p_n / p_n_norm
        if p_a_norm > 0:
            p_a = p_a / p_a_norm
        
        # Cache the prototypes with exact key
        ctx._prototype_cache[cache_key] = (p_n, p_a)
    
    try:
        
        # Generate three descriptions of current sample using Qwen VL
        # Only use system + image + prompt4description to avoid bias from historical dialogue
        model = ctx.image_description_model  # Use image description model for description
        system_message = ctx.system_prompt.format(system_time=_now_iso())
        
        # Extract image: Priority 1) state.raw_image_url, 2) fallback to messages
        extracted_image = None
        if isinstance(state, dict):
            extracted_image = state.get("raw_image_url")
            messages = state.get("messages", [])
        else:
            extracted_image = getattr(state, "raw_image_url", None)
            messages = state.messages
        
        # Fallback: extract from messages if raw_image_url is not set
        if not extracted_image:
            extracted_image = _extract_image_from_messages(messages)
        
        if not extracted_image:
            return "anomalous", 0.5, 0.5, 0.5, "No image found in state.raw_image_url or messages, defaulting to anomalous."
        
        # Build minimal message list: only system + image + prompt4description
        image_content = [{"type": "image_url", "image_url": {"url": extracted_image}}]
        msg_list: List[BaseMessage | Dict[str, str]] = [
            _sys_msg(system_message),
            {"role": "user", "content": image_content + [{"type": "text", "text": ctx.prompt4description}]},
        ]
        
        # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
        if ctx.image_description_model_type == "api":
            resp = await safe_api_invoke(model, msg_list, "image_description", ctx)
        else:
            resp = cast(AIMessage, await model.ainvoke(msg_list))
        
        # Check if API call failed (error message from safe_api_invoke)
        resp_content = _get_text(resp.content)
        api_error_detected = False
        api_error_msg = ""
        
        if resp_content.startswith("[ERROR:") or "[ERROR:" in resp_content:
            # API call failed - extract error message
            api_error_detected = True
            api_error_msg = resp_content
            # Try to extract JSON if present, but prioritize error message
            parsed = _parse_json_from_text(resp_content)
        else:
            parsed = _parse_json_from_text(resp_content)
        
        # Extract captions with fallback mechanism for robustness
        captions = []
        fallback_used = False
        
        if "captions" in parsed and isinstance(parsed["captions"], list):
            captions = parsed["captions"]
        elif "captions" in parsed and isinstance(parsed["captions"], str):
            # Fallback: if captions is a string, convert to list
            captions = [parsed["captions"]]
            fallback_used = True
        elif "description" in parsed and isinstance(parsed["description"], str):
            # Fallback: if description key exists, use it
            captions = [parsed["description"]]
            fallback_used = True
        else:
            # No captions found in any format
            captions = []
        
        # Allow 1-2 captions to be missing (fill with "(missing)" instead of failing)
        if not captions or not isinstance(captions, list):
            captions = []
        
        # Take first three captions, pad with "(missing)" if less than 3
        while len(captions) < 3:
            captions.append("(missing)")
        captions = captions[:3]
        
        # Store captions in state for reuse by other functions
        if not isinstance(state, dict):
            state.image_captions = captions
        
        # Embed each caption and calculate scores, margins, and similarities
        scores = []
        margins = []
        s_n_list = []
        s_a_list = []
        captions_display = []  # Display captions (always 3, with placeholders for missing/invalid)
        caption_status = []  # Track status: "valid", "invalid", "missing"
        valid_captions = []  # Only truly valid captions (for batch embedding)
        valid_caption_indices = []  # Map from valid_captions index to original index
        
        # First pass: collect valid captions for batch embedding
        # Initialize all lists with placeholders (will be updated for valid captions in second pass)
        for i, caption in enumerate(captions):
            if not isinstance(caption, str) or not caption.strip() or caption == "(missing)":
                # Mark as missing/invalid
                caption_status.append("missing" if caption == "(missing)" else "invalid")
                # Use np.nan instead of 0.5 to avoid bias in aggregation
                scores.append(np.nan)
                margins.append(np.nan)
                s_n_list.append(np.nan)
                s_a_list.append(np.nan)
                captions_display.append("(invalid caption)" if caption != "(missing)" else "(missing caption)")
            else:
                caption_status.append("valid")
                captions_display.append(caption)
                valid_captions.append(caption)
                valid_caption_indices.append(i)
                # Initialize with placeholder (will be updated in second pass)
                scores.append(np.nan)
                margins.append(np.nan)
                s_n_list.append(np.nan)
                s_a_list.append(np.nan)
        
        # Batch embed all valid captions at once (performance optimization)
        if valid_captions:
            caption_embeddings = await embedding_model.aembed_documents(valid_captions)
        else:
            caption_embeddings = []
        
        # Second pass: process embeddings and calculate scores
        for valid_idx, orig_idx in enumerate(valid_caption_indices):
            # Embed the caption (from batch)
            caption_embedding = caption_embeddings[valid_idx]
            # Use np.asarray with dtype=np.float32 and validate dimension
            c = np.asarray(caption_embedding, dtype=np.float32)
            if c.ndim != 1:
                raise ValueError(f"Caption embedding must be 1D. Got shape: {c.shape}")
            if c.shape[0] != embedding_dim:
                raise ValueError(f"Caption embedding dimension {c.shape[0]} does not match prototype dimension {embedding_dim}")
            
            # Normalize caption embedding
            c_norm = np.linalg.norm(c)
            if c_norm > 0:
                c = c / c_norm
            else:
                # Zero vector, mark as invalid
                caption_status[orig_idx] = "invalid"
                scores[orig_idx] = np.nan
                margins[orig_idx] = np.nan
                s_n_list[orig_idx] = np.nan
                s_a_list[orig_idx] = np.nan
                continue
            
            # Calculate cosine similarities
            s_n = float(np.dot(c, p_n))  # Cosine similarity with normal prototype
            s_a = float(np.dot(c, p_a))  # Cosine similarity with anomalous prototype
            s_n_list[orig_idx] = s_n
            s_a_list[orig_idx] = s_a
            
            # Calculate margin = (similarity to anomalous) - (similarity to normal)
            margin = s_a - s_n
            margins[orig_idx] = margin
            
            # Calculate logit and score using stable sigmoid
            logit = (s_a - s_n) / tau
            score = stable_sigmoid(logit)
            scores[orig_idx] = float(score)
        
        # Ensure we have exactly 3 items (should already be 3, but safety check)
        scores = scores[:3]
        margins = margins[:3]
        s_n_list = s_n_list[:3]
        s_a_list = s_a_list[:3]
        captions_display = captions_display[:3]
        caption_status = caption_status[:3]
        
        # Calculate diagnostics - ONLY use valid captions to avoid distortion
        valid_indices = [i for i, status in enumerate(caption_status) if status == "valid"]
        valid_margins = [margins[i] for i in valid_indices]
        valid_scores = [scores[i] for i in valid_indices]
        num_valid = len(valid_indices)
        
        # 1. Agreement pattern based on margins (only valid captions)
        if num_valid < 2:
            agreement_pattern = "insufficient evidence (less than 2 valid captions)"
        else:
            lean_anomalous_count = sum(1 for m in valid_margins if m > 0)
            if lean_anomalous_count == num_valid:
                agreement_pattern = "all perspectives lean anomalous"
            elif lean_anomalous_count == num_valid - 1:
                agreement_pattern = f"{lean_anomalous_count} of {num_valid} perspectives lean anomalous"
            elif lean_anomalous_count == 1:
                agreement_pattern = "mixed signals across perspectives"
            else:
                agreement_pattern = "all perspectives lean normal"
        
        # 2. Top perspectives (only valid captions, only select max_margin indices)
        if num_valid == 0:
            top_perspectives = "None (no valid captions)"
        else:
            max_margin = max(valid_margins)
            # Use small epsilon for floating point comparison
            float_epsilon = 1e-6
            # Only select perspectives with max margin (tolerating floating point errors)
            # But if max_margin <= 0, indicate no anomaly-leaning perspective
            if max_margin <= 0:
                top_perspectives = "None (no anomaly-leaning perspective)"
            else:
                # Map back to original indices
                top_valid_indices = [valid_indices[i] for i, m in enumerate(valid_margins) if abs(m - max_margin) < float_epsilon]
                if top_valid_indices:
                    top_perspectives = ", ".join([f"Perspective {i+1}" for i in top_valid_indices])
                else:
                    top_perspectives = "None (no strong anomaly signal)"
        
        # 3. Score spread and margin spread (only valid captions)
        if num_valid < 2:
            score_spread = 0.0
            margin_spread = 0.0
        else:
            score_spread = float(max(valid_scores) - min(valid_scores))
            margin_spread = float(max(valid_margins) - min(valid_margins))
        
        # 4. Uncertainty note based on margin consistency (only valid captions)
        if num_valid < 2:
            uncertainty_note = "High uncertainty: insufficient valid captions"
        else:
            margin_std = float(np.std(valid_margins))
            if margin_std < 0.05:
                uncertainty_note = "Low uncertainty: captions are consistent"
            elif margin_std < 0.1:
                uncertainty_note = "Moderate uncertainty: one perspective disagrees"
            else:
                uncertainty_note = "High uncertainty: large disagreement across captions"
        
        # Determine label based on average score (only valid captions, use nanmean to ignore NaN)
        if num_valid > 0:
            avg_score = float(np.nanmean(valid_scores))
            predicted_label = "anomalous" if avg_score > 0.5 else "normal"
        else:
            # No valid captions, default to uncertain
            avg_score = 0.5
            predicted_label = "anomalous"  # Keep default behavior for now

        avg_margin = float(np.mean(valid_margins)) if valid_margins else None
        
        # Get tool name - hardcode to avoid getting wrong tool from executed_tool_calls
        # (executed_tool_calls[-1] might not be this tool if it's not the last in pipeline)
        tool_name = "keyword_heuristic"  # Hardcoded tool name for this function
        
        # Escape captions to prevent injection/formatting issues
        # Use json.dumps for robust escaping of all special characters
        def escape_caption(caption: str) -> str:
            """Escape caption text for safe insertion into report using json.dumps."""
            if not isinstance(caption, str):
                return json.dumps(str(caption))
            # Use json.dumps to handle all special characters (quotes, backslashes, newlines, etc.)
            return json.dumps(caption)
        
        escaped_captions = [escape_caption(cap) for cap in captions_display]
        
        # Format numbers with proper handling of NaN values (prepare strings outside f-string)
        def fmt_num(x, fmt="{:.3f}"):
            """Format number, returning 'N/A' if NaN or None."""
            return "N/A" if (x is None or (isinstance(x, float) and np.isnan(x))) else fmt.format(x)
        
        # Prepare formatted strings for scores and margins
        score1_s = fmt_num(scores[0])
        score2_s = fmt_num(scores[1])
        score3_s = fmt_num(scores[2])
        margin1_s = "N/A" if np.isnan(margins[0]) else f"{margins[0]:+.3f}"
        margin2_s = "N/A" if np.isnan(margins[1]) else f"{margins[1]:+.3f}"
        margin3_s = "N/A" if np.isnan(margins[2]) else f"{margins[2]:+.3f}"
        
        # Build status/errors summary
        invalid_count = sum(1 for s in caption_status if s == "invalid")
        missing_count = sum(1 for s in caption_status if s == "missing")
        status_summary = []
        status_summary.append(f"num_valid={num_valid}")
        if invalid_count > 0:
            status_summary.append(f"{invalid_count} invalid caption(s)")
        if missing_count > 0:
            status_summary.append(f"{missing_count} missing caption(s)")
        if fallback_used:
            status_summary.append("fallback_used")
        status_text = "; ".join(status_summary)
        
        # Generate detailed report using template from prompts.py
        reason = format_keyword_heuristic_report_v2(
            tool_name=tool_name,
            class_name=class_name,
            escaped_captions=escaped_captions,
            score1_s=score1_s,
            score2_s=score2_s,
            score3_s=score3_s,
            margin1_s=margin1_s,
            margin2_s=margin2_s,
            margin3_s=margin3_s,
            caption_status=caption_status,
            agreement_pattern=agreement_pattern,
            top_perspectives=top_perspectives,
            score_spread=score_spread,
            margin_spread=margin_spread,
            uncertainty_note=uncertainty_note,
            status_text=status_text,
        )
        
        # Prepend structured bias summary to make heuristic evidence explicit
        heuristic_bias = {
            "heuristic_vote": predicted_label,
            "confidence": {
                "avg_margin": avg_margin,
                "margin_range": {"min": -2.0, "max": 2.0},
            },
            "supporting_factors": {
                "agreement_pattern": agreement_pattern,
                "top_perspectives": top_perspectives,
                "num_valid_captions": num_valid,
                "margin_spread": margin_spread,
                "uncertainty_note": uncertainty_note,
                "caption_status": status_text,
            },
        }
        bias_json = json.dumps(heuristic_bias, ensure_ascii=True)
        # reason = f"Heuristic Bias (JSON):\n{bias_json}\n\n{reason}"

        # If API error was detected during image description, prepend error message to reason
        if api_error_detected:
            if reason:
                reason = f"[API_ERROR: image_description {api_error_msg}] {reason}"
            else:
                reason = f"[API_ERROR: image_description {api_error_msg}] API call failed during image description, using fallback captions."
        
        # According to decision parameter: False -> only output reason, True -> only output predicted_label
        if decision:
            # decision=True: only output predicted_label, reason is empty
            return predicted_label, scores[0], scores[1], scores[2], ""
        else:
            # decision=False: only output reason, label is still computed but reason is the main output
            return predicted_label, scores[0], scores[1], scores[2], reason
        
    except Exception as e:
        error_msg = f"Error in embedding-based decision: {str(e)}, defaulting to anomalous."
        if decision:
            return "anomalous", 0.5, 0.5, 0.5, ""
        else:
            return "anomalous", 0.5, 0.5, 0.5, error_msg


async def counterfactual_atomic_candidate_tool(
    state: State,
    runtime: Runtime[Context],
    k: int = 3,
) -> str:
    """
    Counterfactual atomic-candidate tool: Generate evidence report using atomic candidate matching.
    
    Process:
    1. Load candidate sets (anomaly_candidates and normal_candidates) from pre-generated file
       (file path should be set in context.atomic_candidates_file, generated in evaluation)
    2. Embed candidate sets and cache embeddings (one-time)
    3. Use existing captions from state.image_captions (generated by _keyword_heuristic_decision)
       or generate new captions using VLM if not available (fallback)
    4. Similarity matching and evidence extraction (online):
       - Calculate cosine similarity between caption embeddings and candidate embeddings
       - Retrieve top-k matches for each perspective
       - Calculate evidence strength (margin = max_sim_anom - max_sim_norm)
       - Format evidence report
    
    Args:
        state: Current state containing class_name, messages, and image_captions (if available).
        runtime: Runtime context containing models and atomic_candidates_file path.
        k: Number of top matches to retrieve (default: 3).
    
    Returns:
        Formatted evidence report string for planner/reflector/reasoner.
    """
    ctx = runtime.context
    
    # Get class_name from state
    if isinstance(state, dict):
        class_name = state.get("class_name")
        messages = state.get("messages", [])
    else:
        class_name = getattr(state, "class_name", None)
        messages = state.messages
    
    if not class_name:
        return "Error: No class_name in state."
    
    # Get embedding model
    embedding_model = ctx.embedding_model
    if embedding_model is None:
        return "Error: Embedding model not available."
    
    # Initialize embedding cache if not present
    if not hasattr(ctx, '_atomic_embedding_cache'):
        ctx._atomic_embedding_cache = {}  # Cache for candidate embeddings
    
    # Step 1: Load candidate sets from file (pre-generated in evaluation)
    # The file path should be set in context.atomic_candidates_file
    if not ctx.atomic_candidates_file:
        return "Error: atomic_candidates_file not set in context. Candidates should be pre-generated and saved to file."
    
    try:
        # Load candidates from file
        candidates_file_path = Path(ctx.atomic_candidates_file)
        if not candidates_file_path.exists():
            return f"Error: Atomic candidates file not found: {ctx.atomic_candidates_file}"
        
        with open(candidates_file_path, 'r', encoding='utf-8') as f:
            all_candidates = json.load(f)
        
        # Extract candidates for this class (handle case variations)
        class_candidates = all_candidates.get(class_name) or all_candidates.get(class_name.lower()) or all_candidates.get(class_name.strip().lower())
        if not class_candidates:
            return f"Error: Class '{class_name}' not found in atomic candidates file."
        
        valid_anomaly_candidates = class_candidates.get("anomaly_candidates", [])
        valid_normal_candidates = class_candidates.get("normal_candidates", [])
        
        if not isinstance(valid_anomaly_candidates, list):
            valid_anomaly_candidates = []
        if not isinstance(valid_normal_candidates, list):
            valid_normal_candidates = []
        
        # Calculate candidate_version from the candidates (for report)
        candidate_content = json.dumps([valid_anomaly_candidates, valid_normal_candidates], sort_keys=True)
        candidate_version = hashlib.md5(candidate_content.encode()).hexdigest()[:8]
        
    except Exception as e:
        return f"Error loading atomic candidates from file '{ctx.atomic_candidates_file}': {str(e)}"
    
    # Generate prompt hash for version tracking (for report)
    prompt_text = PROMPT4CANDIDATE_GENERATION
    prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
    
    # Standardize class_name for cache key (same as generate_and_save_atomic_candidates)
    standardized_class_name = class_name.strip().lower()
    
    # Step 2: Embed candidate sets and cache (one-time)
    embedding_model_id_raw = ctx.embedding_model_path or "default"
    # Hash embedding_model_id for cache key to improve readability (avoid long paths/special chars in keys)
    embedding_model_id_hash = hashlib.md5(str(embedding_model_id_raw).encode()).hexdigest()[:8]
    embedding_model_id = embedding_model_id_hash  # Use hash for cache key
    
    try:
        embedding_model_fingerprint = getattr(embedding_model, 'model_name', None) or type(embedding_model).__name__
    except:
        embedding_model_fingerprint = "unknown"
    embedding_model_fingerprint_hash = hashlib.md5(str(embedding_model_fingerprint).encode()).hexdigest()[:8]
    
    # Cache key for embeddings (embedding_dim will be stored in cache value for O(1) lookup)
    # Use standardized_class_name to match generate_and_save_atomic_candidates
    embedding_cache_key = f"{standardized_class_name}_{embedding_model_id}_{candidate_version}_{embedding_model_fingerprint_hash}"
    
    # Check cache for embeddings
    if embedding_cache_key in ctx._atomic_embedding_cache:
        # Cache hit: get embeddings and embedding_dim from cache value
        cache_value = ctx._atomic_embedding_cache[embedding_cache_key]
        if len(cache_value) == 5:  # New format: (E_A, E_N, valid_anomaly_candidates, valid_normal_candidates, embedding_dim)
            E_A, E_N, valid_anomaly_candidates, valid_normal_candidates, embedding_dim = cache_value
        else:  # Legacy format: (E_A, E_N, valid_anomaly_candidates, valid_normal_candidates)
            E_A, E_N, valid_anomaly_candidates, valid_normal_candidates = cache_value
            # Extract embedding_dim from embeddings (backward compatibility)
            if E_A:
                E_A_array_temp = np.asarray(E_A, dtype=np.float32)
                embedding_dim = E_A_array_temp.shape[1]
            elif E_N:
                E_N_array_temp = np.asarray(E_N, dtype=np.float32)
                embedding_dim = E_N_array_temp.shape[1]
            else:
                # Fallback: both empty, use conservative default (embedding_dim only used for shape when arrays are empty)
                # This avoids expensive remote call to embedding_model.aembed_query("sample")
                embedding_dim = getattr(ctx, "_embedding_dim", 1024)  # Default: 1024 (common embedding dimension)
    else:
        # Cache miss: need to embed candidates first, then get dim from embeddings
        # valid_anomaly_candidates and valid_normal_candidates are already obtained from generate_and_save_atomic_candidates
        
        # Embed candidates with error handling (embedding service instability)
        try:
            if valid_anomaly_candidates:
                E_A = await embedding_model.aembed_documents(valid_anomaly_candidates)
            else:
                E_A = []
        except Exception as e:
            # Return clear error report instead of crashing the agent task
            return f"Error embedding anomaly candidates: {str(e)}\n" \
                   f"candidate_version: {candidate_version}\n" \
                   f"prompt_hash: {prompt_hash}\n" \
                   f"class_name: {class_name}"
        
        try:
            if valid_normal_candidates:
                E_N = await embedding_model.aembed_documents(valid_normal_candidates)
            else:
                E_N = []
        except Exception as e:
            # Return clear error report instead of crashing the agent task
            return f"Error embedding normal candidates: {str(e)}\n" \
                   f"candidate_version: {candidate_version}\n" \
                   f"prompt_hash: {prompt_hash}\n" \
                   f"class_name: {class_name}"
        
        # Get embedding_dim from embeddings (performance optimization: avoid extra aembed_query)
        if E_A:
            E_A_array_temp = np.asarray(E_A, dtype=np.float32)
            embedding_dim = E_A_array_temp.shape[1]
        elif E_N:
            E_N_array_temp = np.asarray(E_N, dtype=np.float32)
            embedding_dim = E_N_array_temp.shape[1]
        else:
            # Fallback: both empty, use conservative default (embedding_dim only used for shape when arrays are empty)
            # This avoids expensive remote call to embedding_model.aembed_query("sample")
            embedding_dim = getattr(ctx, "_embedding_dim", 1024)  # Default: 1024 (common embedding dimension)
        
        # Cache size limit to prevent memory bloat (FIFO: remove oldest when limit reached)
        # Eviction only when preparing to insert new entry (not on cache hit)
        MAX_CACHE_SIZE = 512
        if len(ctx._atomic_embedding_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(ctx._atomic_embedding_cache))
            del ctx._atomic_embedding_cache[oldest_key]
        
        # Cache embeddings along with valid candidate texts and embedding_dim (store dim in cache value)
        ctx._atomic_embedding_cache[embedding_cache_key] = (E_A, E_N, valid_anomaly_candidates, valid_normal_candidates, embedding_dim)
    
    # Step 3: Use existing captions from state.image_captions (generated by _keyword_heuristic_decision)
    # This avoids redundant VLM calls since captions are already generated in template_tools
    if not isinstance(state, dict):
        captions = getattr(state, "image_captions", None)
    else:
        captions = state.get("image_captions", None)
    
    if captions and isinstance(captions, list) and len(captions) >= 3:
        # Use existing captions from state (preferred)
        captions = captions[:3]
    else:
        # Fallback: Generate new captions using VLM (should not happen in normal flow)
        # Generate new captions using VLM (fallback if not in state)
        vlm_model = ctx.image_description_model  # Use image description model
        
        # Use a clean system message for captioning (only constrains JSON format)
        # Avoid using ctx.system_prompt which may contain other agent rules that interfere with caption output
        # This improves stability during evaluations, similar to candidate generation
        system_message = """You are a visual condition inspection assistant.
        Your task is to analyze images and generate descriptions in JSON format.
        Return ONLY valid JSON objects, with no additional text or explanations."""
        
        # Extract image: Priority 1) state.raw_image_url, 2) fallback to messages
        extracted_image = None
        if isinstance(state, dict):
            extracted_image = state.get("raw_image_url")
        else:
            extracted_image = getattr(state, "raw_image_url", None)
        
        # Fallback: extract from messages if raw_image_url is not set
        if not extracted_image:
            extracted_image = _extract_image_from_messages(messages)
        
        if not extracted_image:
            return "Error: No image found in state.raw_image_url or messages."
        
        # Build minimal message list: only system + image + prompt4description
        image_content = [{"type": "image_url", "image_url": {"url": extracted_image}}]
        msg_list: List[BaseMessage | Dict[str, str]] = [
            _sys_msg(system_message),
            {"role": "user", "content": image_content + [{"type": "text", "text": ctx.prompt4description}]},
        ]
        
        # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
        if ctx.image_description_model_type == "api":
            resp = await safe_api_invoke(vlm_model, msg_list, "image_description_caption", ctx)
        else:
            resp = cast(AIMessage, await vlm_model.ainvoke(msg_list))
        parsed = _parse_json_from_text(_get_text(resp.content))
        
        # Extract captions
        captions = []
        if "captions" in parsed and isinstance(parsed["captions"], list):
            captions = parsed["captions"]
        elif "captions" in parsed and isinstance(parsed["captions"], str):
            captions = [parsed["captions"]]
        elif "description" in parsed and isinstance(parsed["description"], str):
            captions = [parsed["description"]]
        
        # Normalize to 3 captions
        while len(captions) < 3:
            captions.append("(missing)")
        captions = captions[:3]
    
    # Track caption status
    caption_status = []
    valid_captions = []
    valid_caption_indices = []
    
    for i, caption in enumerate(captions):
        if not isinstance(caption, str) or not caption.strip() or caption == "(missing)":
            caption_status.append("missing" if caption == "(missing)" else "invalid")
        else:
            caption_status.append("valid")
            valid_captions.append(caption)
            valid_caption_indices.append(i)
    
    # Step 4: Similarity matching and evidence extraction (online)
    # Embed captions (batch) with error handling
    if valid_captions:
        try:
            caption_embeddings = await embedding_model.aembed_documents(valid_captions)
        except Exception as e:
            # Return clear error report instead of crashing the agent task
            return f"Error embedding captions: {str(e)}\n" \
                   f"candidate_version: {candidate_version}\n" \
                   f"prompt_hash: {prompt_hash}\n" \
                   f"class_name: {class_name}\n" \
                   f"caption_status: {json.dumps(caption_status)}"
    else:
        caption_embeddings = []
    
    # Convert embeddings to numpy arrays for similarity calculation
    if E_A:
        E_A_array = np.asarray(E_A, dtype=np.float32)
        # Normalize embeddings
        E_A_norms = np.linalg.norm(E_A_array, axis=1, keepdims=True)
        E_A_array = E_A_array / np.maximum(E_A_norms, 1e-8)
    else:
        E_A_array = np.array([]).reshape(0, embedding_dim)
    
    if E_N:
        E_N_array = np.asarray(E_N, dtype=np.float32)
        # Normalize embeddings
        E_N_norms = np.linalg.norm(E_N_array, axis=1, keepdims=True)
        E_N_array = E_N_array / np.maximum(E_N_norms, 1e-8)
    else:
        E_N_array = np.array([]).reshape(0, embedding_dim)
    
    # Process each perspective
    perspectives = []
    for valid_idx, orig_idx in enumerate(valid_caption_indices):
        caption = valid_captions[valid_idx]
        caption_embedding = np.asarray(caption_embeddings[valid_idx], dtype=np.float32)
        
        # Normalize caption embedding
        caption_norm = np.linalg.norm(caption_embedding)
        if caption_norm > 0:
            caption_embedding = caption_embedding / caption_norm
        else:
            # Zero vector: mark as invalid but still append perspective (don't silently drop)
            # This prevents perspective count reduction that could confuse planner/reflector
            perspectives.append({
                "index": orig_idx + 1,
                "caption": caption,
                "status": "invalid_embedding_zero_vector",
                "candidate_status": "invalid_caption_embedding",
                "top_k_anom": [],
                "top_k_norm": [],
                "margin": None,
                "evidence_strength_logsumexp": None,
                "max_sim_anom": None,
                "max_sim_norm": None,
            })
            continue
        
        # Calculate cosine similarities
        if E_A_array.size > 0:
            sims_anom = np.dot(E_A_array, caption_embedding)
        else:
            sims_anom = np.array([])
        
        if E_N_array.size > 0:
            sims_norm = np.dot(E_N_array, caption_embedding)
        else:
            sims_norm = np.array([])
        
        # Get top-k matches (indices correspond to valid_anomaly_candidates and valid_normal_candidates)
        if len(sims_anom) > 0:
            top_k_anom_indices = np.argsort(sims_anom)[-k:][::-1]
            top_k_anom = [(float(sims_anom[i]), valid_anomaly_candidates[i] if i < len(valid_anomaly_candidates) else "") 
                          for i in top_k_anom_indices]
        else:
            top_k_anom = []
        
        if len(sims_norm) > 0:
            top_k_norm_indices = np.argsort(sims_norm)[-k:][::-1]
            top_k_norm = [(float(sims_norm[i]), valid_normal_candidates[i] if i < len(valid_normal_candidates) else "") 
                          for i in top_k_norm_indices]
        else:
            top_k_norm = []
        
        # Calculate evidence strength: margin = max_sim_anom - max_sim_norm
        # Use -inf for empty side to avoid false evidence (instead of 0.0 which is misleading)
        if len(sims_anom) > 0:
            max_sim_anom = float(np.max(sims_anom))
        else:
            max_sim_anom = float('-inf')
        
        if len(sims_norm) > 0:
            max_sim_norm = float(np.max(sims_norm))
        else:
            max_sim_norm = float('-inf')
        
        # Calculate margin (handle -inf cases)
        if max_sim_anom == float('-inf') and max_sim_norm == float('-inf'):
            margin = float('-inf')  # Both empty: insufficient candidates
        elif max_sim_anom == float('-inf'):
            margin = float('-inf')  # Anomaly side empty
        elif max_sim_norm == float('-inf'):
            margin = float('inf')  # Normal side empty (strong anomaly signal)
        else:
            margin = max_sim_anom - max_sim_norm
        
        # Alternative evidence strength: logsumexp(topA) - logsumexp(topN)
        # Add temperature parameter for cosine similarity (tau_cand ≈ 0.05-0.2)
        # Validate and clamp tau_cand to prevent numerical explosion (similar to tau in WinCLIP)
        tau_cand = getattr(ctx, "tau_cand", 0.1)  # Default 0.1 if not configured
        tau_cand = max(float(tau_cand), 1e-6)  # Clamp to minimum to prevent division by zero
        if tau_cand < 1e-5:
            warnings.warn(f"tau_cand={tau_cand} too small; clamped for stability")
        if len(top_k_anom) > 0:
            top_anom_sims = [sim for sim, _ in top_k_anom]
            # Apply temperature: logsumexp(top_sims / tau_cand)
            logsumexp_anom = float(logsumexp(np.array(top_anom_sims) / tau_cand))
        else:
            logsumexp_anom = float('-inf')
        
        if len(top_k_norm) > 0:
            top_norm_sims = [sim for sim, _ in top_k_norm]
            # Apply temperature: logsumexp(top_sims / tau_cand)
            logsumexp_norm = float(logsumexp(np.array(top_norm_sims) / tau_cand))
        else:
            logsumexp_norm = float('-inf')
        
        evidence_strength_logsumexp = logsumexp_anom - logsumexp_norm
        
        # Handle nan case: -inf - -inf = nan (when both sides are empty)
        if not np.isfinite(evidence_strength_logsumexp):
            evidence_strength_logsumexp = float("-inf")
        
        # Determine status for insufficient candidates
        candidate_status = "valid"
        if len(sims_anom) == 0 and len(sims_norm) == 0:
            candidate_status = "insufficient_candidates_both"
        elif len(sims_anom) == 0:
            candidate_status = "insufficient_anomaly_candidates"
        elif len(sims_norm) == 0:
            candidate_status = "insufficient_normal_candidates"
        
        # Set invalid values to None for structured data safety (when candidate_status != "valid")
        # This prevents reasoner from misusing inf values when perspectives are passed as structured data
        if candidate_status != "valid":
            margin = None
            evidence_strength_logsumexp = None
            max_sim_anom = None
            max_sim_norm = None
        
        perspectives.append({
            "index": orig_idx + 1,
            "caption": caption,
            "status": caption_status[orig_idx],
            "candidate_status": candidate_status,
            "top_k_anom": top_k_anom,
            "top_k_norm": top_k_norm,
            "margin": margin,
            "evidence_strength_logsumexp": evidence_strength_logsumexp,
            "max_sim_anom": max_sim_anom,
            "max_sim_norm": max_sim_norm,
        })
    
    # Calculate summary statistics for printing
    valid_perspectives = [p for p in perspectives if p['candidate_status'] == "valid"]
    if valid_perspectives:
        margin_values = [p['margin'] for p in valid_perspectives if p['margin'] is not None and np.isfinite(p['margin'])]
        evidence_values = [p['evidence_strength_logsumexp'] for p in valid_perspectives if p['evidence_strength_logsumexp'] is not None and np.isfinite(p['evidence_strength_logsumexp'])]
        sim_anom_values = [p['max_sim_anom'] for p in valid_perspectives if p['max_sim_anom'] is not None and np.isfinite(p['max_sim_anom'])]
        sim_norm_values = [p['max_sim_norm'] for p in valid_perspectives if p['max_sim_norm'] is not None and np.isfinite(p['max_sim_norm'])]
        
        avg_margin = np.mean(margin_values) if margin_values else None
        avg_evidence_strength = np.mean(evidence_values) if evidence_values else None
        avg_max_sim_anom = np.mean(sim_anom_values) if sim_anom_values else None
        avg_max_sim_norm = np.mean(sim_norm_values) if sim_norm_values else None
        num_valid = len(valid_perspectives)
    else:
        avg_margin = None
        avg_evidence_strength = None
        avg_max_sim_anom = None
        avg_max_sim_norm = None
        num_valid = 0
    
    # Store statistics in state for printing in agent_v1_12.py
    # This is per-sample data, so it should be in state, not context
    if not isinstance(state, dict):
        state.counterfactual_stats = {
            'avg_margin': avg_margin,
            'avg_evidence_strength': avg_evidence_strength,
            'avg_max_sim_anom': avg_max_sim_anom,
            'avg_max_sim_norm': avg_max_sim_norm,
            'num_valid': num_valid,
            'num_total': len(perspectives),
        }
    else:
        state['counterfactual_stats'] = {
            'avg_margin': avg_margin,
            'avg_evidence_strength': avg_evidence_strength,
            'avg_max_sim_anom': avg_max_sim_anom,
            'avg_max_sim_norm': avg_max_sim_norm,
            'num_valid': num_valid,
            'num_total': len(perspectives),
        }
    
    # Format evidence report
    tool_name = "counterfactual_atomic_candidate_tool"
    raw_id = str(embedding_model_id_raw)
    report_lines = [
        f"[{tool_name}] Evidence report for class=\"{class_name}\":",
        "",
        "Method:",
        "- Generated atomic candidates (anomaly_candidates and normal_candidates) using LLM",
        "- Embedded candidates and cached embeddings",
        "- Generated multi-perspective captions for current image using VLM",
        "- Calculated cosine similarity between caption embeddings and candidate embeddings",
        "- Retrieved top-k matches and calculated evidence strength",
        f"- candidate_version: {candidate_version}",
        f"- prompt_hash: {prompt_hash}",
        f"- embedding_model_id_raw: {raw_id[:80] if len(raw_id) > 80 else raw_id}",
        "",
        "Evidence by perspective:",
    ]
    
    # Check if there are no valid captions
    if not perspectives:
        report_lines.append("\nWarning: No valid captions; cannot compute candidate evidence.")
        report_lines.append(f"Caption status: {json.dumps(caption_status)}")
        report_lines.append("Consider re-generating captions using VLM.")
    else:
        for persp in perspectives:
            report_lines.append(f"\nPerspective {persp['index']}:")
            report_lines.append(f"  - caption: {json.dumps(persp['caption'])}")
            report_lines.append(f"  - status: {persp['status']}")
            if persp['candidate_status'] != "valid":
                report_lines.append(f"  - candidate_status: {persp['candidate_status']}")
            report_lines.append(f"  - top-{k} anomaly candidates:")
            for i, (sim, candidate) in enumerate(persp['top_k_anom'], 1):
                report_lines.append(f"    {i}. sim={sim:.4f}: {json.dumps(candidate)}")
            report_lines.append(f"  - top-{k} normal candidates:")
            for i, (sim, candidate) in enumerate(persp['top_k_norm'], 1):
                report_lines.append(f"    {i}. sim={sim:.4f}: {json.dumps(candidate)}")
            # Format margin and logsumexp (handle -inf and inf, and invalid candidate_status)
            # When candidate_status != "valid", margin/evidence_strength are not comparable (mark as N/A)
            if persp['candidate_status'] != "valid":
                margin_str = "N/A (incomplete candidate set)"
                logsumexp_str = "N/A (incomplete candidate set)"
            else:
                margin_str = "inf" if persp['margin'] == float('inf') else ("-inf" if persp['margin'] == float('-inf') else f"{persp['margin']:.4f}")
                logsumexp_str = "-inf" if persp['evidence_strength_logsumexp'] == float('-inf') else f"{persp['evidence_strength_logsumexp']:.4f}"
            report_lines.append(f"  - margin (max_sim_anom - max_sim_norm): {margin_str}")
            report_lines.append(f"  - evidence_strength (logsumexp with temperature): {logsumexp_str}")
    
    report_lines.append("\nUsage note:")
    report_lines.append("- This report provides evidence signals based on atomic candidate matching.")
    report_lines.append("- Do NOT treat any single score as a final decision.")
    report_lines.append("- Use margins, evidence strength, and top matches to inform judgment.")
    
    return "\n".join(report_lines)
