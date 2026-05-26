"""Utility & helper functions (with micro-batching for Qwen-VL)."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from datetime import UTC, datetime
from typing import cast
from .api_model_utils import get_api_model_provider, is_google_model, is_openai_model
from .prompts_v1_27 import PROMPT4CANDIDATE_GENERATION

if TYPE_CHECKING:
    from .context_v1_27 import Context

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings

import json
import re
import hashlib
import torch
import asyncio
import threading
import traceback
from tqdm import tqdm
# from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoEForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
from pydantic import PrivateAttr


# ================================================================
# API call utilities with retry and error handling
# ================================================================

# Global semaphore for API call concurrency control (lazy initialized)
_api_call_semaphore: Optional[asyncio.Semaphore] = None
_api_call_semaphore_max: Optional[int] = None
_api_call_semaphore_lock = threading.Lock()


def _get_api_call_semaphore(max_concurrency: Optional[int]) -> Optional[asyncio.Semaphore]:
    """Get or create the global API call semaphore.
    
    Args:
        max_concurrency: Maximum concurrent API calls. If None, returns None (no limit).
    
    Returns:
        Optional[asyncio.Semaphore]: The semaphore instance, or None if no limit.
    """
    global _api_call_semaphore, _api_call_semaphore_max
    
    if max_concurrency is None:
        return None
    
    if _api_call_semaphore is None or _api_call_semaphore_max != max_concurrency:
        with _api_call_semaphore_lock:
            # Double-check locking pattern
            if _api_call_semaphore is None or _api_call_semaphore_max != max_concurrency:
                _api_call_semaphore = asyncio.Semaphore(max_concurrency)
                _api_call_semaphore_max = max_concurrency
    
    return _api_call_semaphore


def _should_retry_error(error: Exception, error_str: str) -> Tuple[bool, str]:
    """Determine if an error should be retried.
    
    Args:
        error: The exception object.
        error_str: String representation of the error (lowercased).
    
    Returns:
        Tuple[bool, str]: (should_retry, error_type)
    """
    error_class = type(error).__name__
    
    # Fast-fail: Authentication/authorization errors (401/403)
    auth_keywords = ["401", "403", "unauthorized", "forbidden", "invalid_api_key", 
                     "authentication", "auth", "api key", "apikey", "invalid key",
                     "invalid key format", "key invalid", "invalid credentials"]
    if any(keyword in error_str for keyword in auth_keywords):
        return False, "AuthenticationError"
    
    # Fast-fail: Bad request errors (400) - usually parameter/format errors
    bad_request_keywords = ["400", "bad request", "invalid parameter", "invalid format",
                           "malformed", "invalid input", "parse error", "validation error"]
    if any(keyword in error_str for keyword in bad_request_keywords):
        # For 400 errors, we might want to retry in some cases (e.g., temporary format issues),
        # but by default, we don't retry
        return False, "BadRequestError"
    
    # Retry: Rate limit errors (429)
    rate_limit_keywords = ["rate limit", "ratelimit", "rate_limit", "429", 
                          "too many requests", "quota", "quota exceeded", 
                          "throttle", "throttled"]
    if any(keyword in error_str for keyword in rate_limit_keywords):
        return True, "RateLimitError"
    
    # Retry: Server errors (5xx)
    server_error_keywords = ["500", "502", "503", "504", "internal server error",
                            "bad gateway", "service unavailable", "gateway timeout",
                            "server error", "service error"]
    if any(keyword in error_str for keyword in server_error_keywords):
        return True, "ServerError"
    
    # Retry: Network/timeout errors (default behavior)
    network_error_classes = ["ConnectionError", "TimeoutError", "ConnectTimeout",
                            "ReadTimeout", "NetworkError", "ConnectionResetError"]
    if error_class in network_error_classes:
        return True, "NetworkError"
    
    # Default: Don't retry unknown errors (safer than retrying everything)
    return False, error_class


def _get_api_model_id_for_node(context: "Context", node_name: str) -> Optional[str]:
    """Best-effort mapping from node name to configured API model id."""
    node = (node_name or "").strip().lower()

    if node == "planner":
        return getattr(context, "planner_api_model", None)
    if node == "reflector":
        return getattr(context, "reflector_api_model", None)
    if node in {"reasoner", "memory"}:
        return getattr(context, "reasoner_api_model", None)
    if node.startswith("image_description"):
        return getattr(context, "image_description_api_model", None)
    if node.startswith("atomic_candidate"):
        return getattr(context, "atomic_candidate_llm_api_model", None)

    # Skip ambiguous nodes (e.g., ROI can route through planner or reflector model).
    return None


def _normalize_messages_for_api(messages: List[BaseMessage | Dict[str, str]]) -> List[BaseMessage | Dict[str, str]]:
    """Normalize messages for API calls by converting PIL Image objects to base64 data URLs.
    
    API models (like OpenAI, Anthropic) cannot handle PIL Image objects directly.
    This function converts PIL Image objects in message content to base64-encoded data URLs.
    If conversion fails, replaces PIL Image with a text placeholder to prevent JSON serialization errors.
    
    Args:
        messages: List of messages that may contain PIL Image objects.
    
    Returns:
        List of messages with PIL Image objects converted to base64 data URLs.
    """
    # Lazy import to avoid circular dependency: context -> utils -> tools -> context
    try:
        from .tools_v1_27 import _encode_image_to_base64, _normalize_to_data_url
    except ImportError:
        _encode_image_to_base64 = None
        _normalize_to_data_url = None
    
    try:
        from PIL import Image as PILImage
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
    
    if not PIL_AVAILABLE:
        return messages
    
    # Check if encoding function is available
    if _encode_image_to_base64 is None:
        print("[WARNING] _encode_image_to_base64 is not available, PIL Images will be replaced with placeholders")
    
    normalized_messages = []
    for msg in messages:
        # Handle dict messages (like {"role": "system", "content": ...})
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict):
                        # Check for PIL Image object in "image" field
                        if item.get("type") == "image" and "image" in item:
                            pil_img = item.get("image")
                            if isinstance(pil_img, PILImage.Image):
                                try:
                                    if _encode_image_to_base64 is None:
                                        raise ImportError("_encode_image_to_base64 is not available")
                                    base64_url = _encode_image_to_base64(pil_img)
                                    if not base64_url:
                                        raise ValueError("_encode_image_to_base64 returned empty string")
                                    normalized_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": base64_url}
                                    })
                                except Exception as e:
                                    # If conversion fails, replace with text placeholder instead of keeping PIL Image
                                    error_msg = f"[IMAGE_CONVERSION_FAILED: {type(e).__name__}: {str(e)[:100]}]"
                                    print(f"[WARNING] Failed to convert PIL Image to base64 in dict message: {error_msg}")
                                    normalized_content.append({
                                        "type": "text",
                                        "text": error_msg
                                    })
                            else:
                                normalized_content.append(item)
                        else:
                            normalized_content.append(item)
                    elif isinstance(item, PILImage.Image):
                        # Direct PIL Image in list
                        try:
                            if _encode_image_to_base64 is None:
                                raise ImportError("_encode_image_to_base64 is not available")
                            base64_url = _encode_image_to_base64(item)
                            if not base64_url:
                                raise ValueError("_encode_image_to_base64 returned empty string")
                            normalized_content.append({
                                "type": "image_url",
                                "image_url": {"url": base64_url}
                            })
                        except Exception as e:
                            # If conversion fails, replace with text placeholder instead of keeping PIL Image
                            error_msg = f"[IMAGE_CONVERSION_FAILED: {type(e).__name__}: {str(e)[:100]}]"
                            print(f"[WARNING] Failed to convert PIL Image to base64 in dict message (direct PIL): {error_msg}")
                            normalized_content.append({
                                "type": "text",
                                "text": error_msg
                            })
                    else:
                        normalized_content.append(item)
                normalized_messages.append({**msg, "content": normalized_content})
            else:
                normalized_messages.append(msg)
        # Handle BaseMessage objects (HumanMessage, AIMessage, etc.)
        else:
            try:
                content = getattr(msg, "content", None)
                if content is None:
                    normalized_messages.append(msg)
                    continue
                
                # Handle list content (multimodal)
                if isinstance(content, list):
                    normalized_content = []
                    for item in content:
                        if isinstance(item, dict):
                            # Check for PIL Image object
                            if item.get("type") == "image" and "image" in item:
                                pil_img = item.get("image")
                                if isinstance(pil_img, PILImage.Image):
                                    try:
                                        if _encode_image_to_base64 is None:
                                            raise ImportError("_encode_image_to_base64 is not available")
                                        base64_url = _encode_image_to_base64(pil_img)
                                        if not base64_url:
                                            raise ValueError("_encode_image_to_base64 returned empty string")
                                        normalized_content.append({
                                            "type": "image_url",
                                            "image_url": {"url": base64_url}
                                        })
                                    except Exception as e:
                                        # If conversion fails, replace with text placeholder instead of keeping PIL Image
                                        error_msg = f"[IMAGE_CONVERSION_FAILED: {type(e).__name__}: {str(e)[:100]}]"
                                        print(f"[WARNING] Failed to convert PIL Image to base64 in BaseMessage: {error_msg}")
                                        normalized_content.append({
                                            "type": "text",
                                            "text": error_msg
                                        })
                                else:
                                    normalized_content.append(item)
                            else:
                                normalized_content.append(item)
                        elif isinstance(item, PILImage.Image):
                            # Direct PIL Image in list
                            try:
                                if _encode_image_to_base64 is None:
                                    raise ImportError("_encode_image_to_base64 is not available")
                                base64_url = _encode_image_to_base64(item)
                                if not base64_url:
                                    raise ValueError("_encode_image_to_base64 returned empty string")
                                normalized_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": base64_url}
                                })
                            except Exception as e:
                                # If conversion fails, replace with text placeholder instead of keeping PIL Image
                                error_msg = f"[IMAGE_CONVERSION_FAILED: {type(e).__name__}: {str(e)[:100]}]"
                                print(f"[WARNING] Failed to convert PIL Image to base64 in BaseMessage (direct PIL): {error_msg}")
                                normalized_content.append({
                                    "type": "text",
                                    "text": error_msg
                                })
                        else:
                            normalized_content.append(item)
                    
                    # Create a new message object with normalized content
                    # Use pydantic's model_copy or copy(update=...) for stable message reconstruction
                    # This preserves all fields and avoids issues with read-only/frozen fields
                    try:
                        # Try model_copy (pydantic v2+)
                        if hasattr(msg, "model_copy"):
                            new_msg = msg.model_copy(update={"content": normalized_content})
                        # Fallback to copy(update=...) for older pydantic versions
                        elif hasattr(msg, "copy"):
                            new_msg = msg.copy(update={"content": normalized_content})
                        else:
                            # Last resort: use type() but log a warning
                            print(f"[WARNING] Message type {type(msg)} does not support model_copy or copy, using type() fallback")
                            new_msg = type(msg)(content=normalized_content)
                            # Copy other attributes if they exist
                            for attr in ["id", "name", "additional_kwargs"]:
                                if hasattr(msg, attr):
                                    try:
                                        setattr(new_msg, attr, getattr(msg, attr))
                                    except Exception:
                                        pass  # Some attributes might be read-only
                    except Exception as e:
                        # If message reconstruction fails, log and use original message
                        print(f"[WARNING] Failed to reconstruct message with normalized content: {type(e).__name__}: {str(e)[:100]}")
                        normalized_messages.append(msg)
                        continue
                    
                    normalized_messages.append(new_msg)
                else:
                    # String or other content - no conversion needed
                    normalized_messages.append(msg)
            except Exception:
                # If normalization fails, use original message
                normalized_messages.append(msg)
    
    return normalized_messages


async def safe_api_invoke(
    model: BaseChatModel,
    messages: List[BaseMessage | Dict[str, str]],
    node_name: str,
    context: "Context",
    default_error_content: Optional[str] = None,
) -> AIMessage:
    """Safely invoke an API model with retry and error handling.
    
    This function ensures that API call failures don't crash the entire batch or agent flow.
    On failure, returns a default error message instead of raising an exception.
    
    Args:
        model: The model to invoke (BaseChatModel instance).
        messages: List of messages to send to the model.
        node_name: Name of the node making the call (for logging, e.g., 'planner', 'reasoner').
        context: Context object containing retry/timeout configuration.
        default_error_content: Optional custom error message. If None, uses a generic error message.
    
    Returns:
        AIMessage: The model's response, or an error message if all retries failed.
    """
    timeout = context.api_call_timeout
    max_retries = context.api_call_max_retries
    retry_delay = context.api_call_retry_delay
    max_retry_delay = context.api_call_max_retry_delay
    debug_errors = context.debug_api_errors
    
    # Get semaphore for concurrency control
    semaphore = _get_api_call_semaphore(context.api_max_concurrency)
    
    last_exception = None
    error_type = None
    
    async def _do_invoke():
        """Inner function to perform the actual API call."""
        # Normalize messages: convert PIL Image objects to base64 data URLs
        # API models cannot handle PIL Image objects directly
        normalized_messages = _normalize_messages_for_api(messages)
        
        # Build model invocation parameters from context
        invoke_kwargs = {}
        api_model_id = _get_api_model_id_for_node(context, node_name)
        api_provider = get_api_model_provider(api_model_id)
        
        # Apply generation parameters if specified in context
        if context.api_temperature is not None:
            invoke_kwargs["temperature"] = context.api_temperature
        if context.api_max_tokens is not None:
            if is_google_model(api_model_id):
                invoke_kwargs["max_output_tokens"] = context.api_max_tokens
            else:
                invoke_kwargs["max_tokens"] = context.api_max_tokens
        if context.api_top_p is not None:
            invoke_kwargs["top_p"] = context.api_top_p

        # OpenAI Responses API reasoning effort (provider/node guarded).
        reasoning_effort = getattr(context, "reasoning_effort", None)
        use_responses_api = bool(getattr(context, "use_responses_api", False))

        # Google Gemini does not accept OpenAI-style penalty parameters.
        # In Responses API mode, do not pass Chat Completions penalty params at all.
        if not use_responses_api and not is_google_model(api_model_id):
            if context.api_frequency_penalty is not None:
                invoke_kwargs["frequency_penalty"] = context.api_frequency_penalty
            if context.api_presence_penalty is not None:
                invoke_kwargs["presence_penalty"] = context.api_presence_penalty
        elif (
            debug_errors
            and (
                context.api_frequency_penalty is not None
                or context.api_presence_penalty is not None
            )
        ):
            print(
                f"[DEBUG] [{node_name}] Skipping frequency_penalty/presence_penalty "
                f"for provider={api_provider or 'unknown'}."
            )

        if (
            isinstance(reasoning_effort, str)
            and reasoning_effort.strip()
            and use_responses_api
            and is_openai_model(api_model_id)
        ):
            invoke_kwargs["reasoning"] = {"effort": reasoning_effort.strip().lower()}
        
        # Use bind() to apply parameters if any are specified
        # This is more reliable than passing kwargs to ainvoke directly
        if invoke_kwargs:
            bound_model = model.bind(**invoke_kwargs)
            return await asyncio.wait_for(
                bound_model.ainvoke(normalized_messages),
                timeout=timeout
            )
        else:
            return await asyncio.wait_for(
                model.ainvoke(normalized_messages),
                timeout=timeout
            )
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Use semaphore for concurrency control if enabled
            if semaphore is not None:
                async with semaphore:
                    response = await _do_invoke()
            else:
                response = await _do_invoke()
            
            # If we get here, the call succeeded
            if attempt > 0 and debug_errors:
                print(f"[DEBUG] [{node_name}] API call succeeded on retry attempt {attempt + 1}")
            return cast(AIMessage, response)
            
        except asyncio.TimeoutError as e:
            last_exception = e
            error_type = "TimeoutError"
            error_msg = f"API call timeout after {timeout}s"
            # Timeout errors should be retried
            if attempt < max_retries:
                delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                if debug_errors:
                    print(f"[DEBUG] [{node_name}] {error_msg} (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                if debug_errors:
                    print(f"[DEBUG] [{node_name}] {error_msg} (attempt {attempt + 1}/{max_retries + 1}). All retries exhausted.")
                
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Determine if we should retry this error
            should_retry, detected_error_type = _should_retry_error(e, error_str)
            error_type = detected_error_type
            
            error_msg = f"API call error ({detected_error_type}): {str(e)[:200]}"  # Truncate to 200 chars
            
            if should_retry:
                # Retry this error
                if attempt < max_retries:
                    if detected_error_type == "RateLimitError":
                        # For rate limits, use longer delay
                        delay = min(retry_delay * (2 ** (attempt + 1)), max_retry_delay * 2)
                        if debug_errors:
                            print(f"[DEBUG] [{node_name}] {error_msg} (attempt {attempt + 1}/{max_retries + 1}). Waiting {delay:.1f}s before retry...")
                    else:
                        delay = min(retry_delay * (2 ** attempt), max_retry_delay)
                        if debug_errors:
                            print(f"[DEBUG] [{node_name}] {error_msg} (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    if debug_errors:
                        print(f"[DEBUG] [{node_name}] {error_msg} (attempt {attempt + 1}/{max_retries + 1}). All retries exhausted.")
            else:
                # Fast-fail: Don't retry this error
                if debug_errors:
                    print(f"[DEBUG] [{node_name}] {error_msg} (fast-fail, no retry)")
                break  # Exit the retry loop immediately
    
    # All retries failed or fast-failed - return error message instead of raising exception
    error_content = default_error_content or (
        f"[ERROR: {node_name} API call failed. "
        f"Error type: {error_type}. "
        f"Last error: {str(last_exception)[:200] if last_exception else 'Unknown error'}]"
    )
    
    # Always print final failure message (this is important for users to know something went wrong)
    error_class = type(last_exception).__name__ if last_exception else "Unknown"
    error_msg_truncated = str(last_exception)[:200] if last_exception else "Unknown error"
    print(f"[ERROR] [{node_name}] API call failed. Error type: {error_type}. Error: {error_class}: {error_msg_truncated}")
    
    # Print detailed error info only in debug mode
    if debug_errors and last_exception:
        # Debug mode: Print full traceback (may leak sensitive info)
        # Use print_exception instead of print_exc because we're outside the except block
        # print_exc relies on sys.exc_info() which may be empty outside except blocks
        traceback.print_exception(type(last_exception), last_exception, last_exception.__traceback__)
    
    # Return an AIMessage with error content (this allows the flow to continue)
    return AIMessage(content=error_content)


# ================================================================
# Message utilities
# ================================================================

def get_message_text(msg: BaseMessage) -> str:
    """Extract plain text from a LangChain message (best-effort)."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()
    return str(content).strip()


def extract_images_for_reasoner(
    messages: List[BaseMessage],
    encode_image_to_base64: Optional[Callable] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract raw image (first image from user message) and 
    latest augmented image (from most recent ToolMessage with processed image).
    
    Improved version: More robust image extraction with fallback normalization.
    
    Args:
        messages: List of messages from state.messages
        encode_image_to_base64: Optional function to encode PIL Image to base64.
            If None, will try to import from tools module.
    
    Returns:
        (raw_image_base64, augmented_image_base64) - both can be None, but normalized to data URL if found
    """
    # Lazy import to avoid circular dependency: context -> utils -> tools -> context
    try:
        from .tools_v1_27 import _normalize_to_data_url, _encode_image_to_base64
    except ImportError:
        _normalize_to_data_url = None
        _encode_image_to_base64 = None
    
    raw_image = None
    augmented_image = None
    
    # Use imported function if not provided
    if encode_image_to_base64 is None:
        encode_image_to_base64 = _encode_image_to_base64
    
    # Extract raw image: Scan for the FIRST user message containing an image
    # (Not fixed to messages[0], as it might be a SystemMessage)
    for msg in messages:
        # Only check HumanMessage (user input)
        if not isinstance(msg, HumanMessage):
            continue
            
        content = getattr(msg, "content", None)
        if not content:
            continue
            
        # Check list content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Handle image_url format
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            # Normalize to data URL if needed
                            normalized = _normalize_to_data_url(url)
                            if normalized:
                                raw_image = normalized
                                break
                    # Handle PIL Image object
                    elif item.get("type") == "image" and "image" in item:
                        try:
                            from PIL import Image as PILImage
                            pil_img = item.get("image")
                            if pil_img and isinstance(pil_img, PILImage.Image):
                                if encode_image_to_base64:
                                    raw_image = encode_image_to_base64(pil_img)
                                    break
                        except ImportError:
                            pass
        # Check string content (might contain base64)
        elif isinstance(content, str):
            # Try to normalize if it looks like base64
            normalized = _normalize_to_data_url(content)
            if normalized:
                raw_image = normalized
                break
        
        if raw_image:
            break
    
    # Extract latest augmented image from ToolMessages (processed image)
    # Search in reverse to find the most recent one
    # Support both list content and string content (with normalization)
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
            
        content = getattr(msg, "content", None)
        if not content:
            continue
        
        # Handle list content (standard format)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url:
                        # Normalize to data URL if needed
                        normalized = _normalize_to_data_url(url)
                        if normalized:
                            augmented_image = normalized
                            break
            if augmented_image:
                break
        # Handle string content (fallback: might be JSON string or base64)
        elif isinstance(content, str):
            # Try to parse as JSON first
            try:
                import json
                tool_response = json.loads(content)
                if isinstance(tool_response, dict) and "processed_image_base64" in tool_response:
                    url = tool_response["processed_image_base64"]
                    normalized = _normalize_to_data_url(url)
                    if normalized:
                        augmented_image = normalized
                        break
            except (json.JSONDecodeError, KeyError):
                # Not JSON, try direct normalization
                normalized = _normalize_to_data_url(content)
                if normalized:
                    augmented_image = normalized
                    break
    
    return raw_image, augmented_image


def format_potential_anomalies(anomalies_list: List[str]) -> str:
    """Format potential anomalies list into a readable string."""
    if not anomalies_list:
        return ""
    return "\n\n".join([
        f"[{i+1}] {item}" for i, item in enumerate(anomalies_list) if item
    ])


def format_heuristic_prompts(heuristic_list: List[str]) -> str:
    """Format heuristic prompts list into a readable string."""
    if not heuristic_list:
        return ""
    return "\n\n".join([
        f"[{i+1}] {item}" for i, item in enumerate(heuristic_list) if item
    ])


def resize_and_compress_image(
    data_url: str,
    max_edge: int = 1024,
    jpeg_quality: int = 85,
    decode_func: Optional[Callable] = None,
) -> str:
    """Resize image to max_edge and compress as JPEG to reduce size.
    
    Args:
        data_url: Base64 data URL of the image (must start with 'data:image').
        max_edge: Maximum edge length (default 1024).
        jpeg_quality: JPEG compression quality (default 85).
        decode_func: Optional function to decode base64 image to PIL.Image.
                     If None, uses internal decoding logic.
    
    Returns:
        Compressed data URL string, or original on error.
    """
    import base64
    import io
    from PIL import Image as PILImage, ImageFile, ImageOps
    
    # Validate input: must be a data URL (not a file path or regular URL)
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        print(f"[WARNING] resize_and_compress_image: input is not a valid data URL, skipping")
        return data_url
    
    max_base64_chars = 30 * 1024 * 1024
    max_jpeg_bytes = 10 * 1024 * 1024
    min_jpeg_quality = 50
    min_edge = 256
    max_decode_pixels = max_edge * max_edge * 16
    
    # NOTE: These are global settings that affect ALL PIL image loading in this process.
    # Restore them in finally to avoid leaking side effects across the process.
    old_truncated_images = ImageFile.LOAD_TRUNCATED_IMAGES
    old_max_pixels = PILImage.MAX_IMAGE_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PILImage.MAX_IMAGE_PIXELS = max_decode_pixels
    
    try:
        # Decode image
        pil_img = None
        if decode_func is not None:
            if len(data_url) > max_base64_chars:
                print(f"[WARNING] resize_and_compress_image: input data URL too large ({len(data_url):,} chars), skipping decode")
                return data_url
            pil_img = decode_func(data_url)
        else:
            # Internal decoding logic - only for data URLs
            try:
                # Extract base64 part after the comma
                if ";base64," in data_url:
                    b64_data = data_url.split(";base64,")[1]
                elif "," in data_url:
                    b64_data = data_url.split(",")[1]
                else:
                    print(f"[WARNING] resize_and_compress_image: cannot parse data URL format")
                    return data_url
                
                b64_data = b64_data.strip()
                if len(b64_data) > max_base64_chars:
                    print(f"[WARNING] resize_and_compress_image: base64 payload too large ({len(b64_data):,} chars), skipping decode")
                    return data_url
                b64_data += "=" * (-len(b64_data) % 4)
                
                # Try strict validation first, fallback to lenient decode
                # (some sources produce non-standard base64 with newlines/url-safe chars)
                try:
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except Exception:
                    image_bytes = base64.b64decode(b64_data, validate=False)
                
                pil_img = PILImage.open(io.BytesIO(image_bytes))
                w, h = pil_img.size
                if w * h > max_decode_pixels:
                    print(f"[WARNING] resize_and_compress_image: image too large ({w}x{h} > {max_decode_pixels:,} pixels), skipping")
                    return data_url
                # Force load to catch truncated/corrupt images early
                pil_img.load()
                # Apply EXIF orientation BEFORE convert (preserves EXIF info)
                try:
                    pil_img = ImageOps.exif_transpose(pil_img)
                except Exception:
                    pass  # Ignore if no EXIF or transpose fails
                pil_img = pil_img.convert("RGB")
            except Exception as e:
                print(f"[WARNING] resize_and_compress_image: decode failed: {e}")
                pil_img = None
        
        if pil_img is None:
            return data_url  # Return original if decode fails
        
        w, h = pil_img.size
        if w * h > max_decode_pixels:
            print(f"[WARNING] resize_and_compress_image: decoded image too large ({w}x{h} > {max_decode_pixels:,} pixels), skipping")
            return data_url
        
        # Resize if larger than max_edge
        if max(w, h) > max_edge:
            if w > h:
                new_w, new_h = max_edge, int(h * max_edge / w)
            else:
                new_w, new_h = int(w * max_edge / h), max_edge
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            print(f"[INFO] Resized image from {w}x{h} to {new_w}x{new_h}")
        
        # Compress as JPEG with optimize and progressive for better compression
        buffer = io.BytesIO()
        def _encode_jpeg(img: PILImage.Image, quality: int) -> bytes:
            buffer.seek(0)
            buffer.truncate(0)
            img.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            return buffer.getvalue()
        
        quality = jpeg_quality
        current_img = pil_img
        jpeg_blob = _encode_jpeg(current_img, quality)
        if len(jpeg_blob) > max_jpeg_bytes:
            for _ in range(6):
                if len(jpeg_blob) <= max_jpeg_bytes:
                    break
                if quality > min_jpeg_quality:
                    quality = max(min_jpeg_quality, int(quality * 0.85))
                    jpeg_blob = _encode_jpeg(current_img, quality)
                    continue
                w, h = current_img.size
                new_edge = int(max(w, h) * 0.85)
                if new_edge < min_edge:
                    break
                if w > h:
                    new_w, new_h = new_edge, int(h * new_edge / w)
                else:
                    new_w, new_h = int(w * new_edge / h), new_edge
                current_img = current_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                jpeg_blob = _encode_jpeg(current_img, quality)
        
        jpeg_bytes = len(jpeg_blob)
        compressed_b64 = base64.b64encode(jpeg_blob).decode("utf-8")
        result = f"data:image/jpeg;base64,{compressed_b64}"
        print(f"[INFO] Compressed image: jpeg_bytes={jpeg_bytes:,}, b64_chars={len(compressed_b64):,}")
        return result
    except Exception as e:
        print(f"[WARNING] Failed to resize/compress image: {e}")
        return data_url  # Return original on error
    finally:
        PILImage.MAX_IMAGE_PIXELS = old_max_pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = old_truncated_images


def estimate_bytes_from_data_url(data_url: str) -> int:
    """Estimate decoded byte size from a data URL without full base64 decode."""
    if not isinstance(data_url, str):
        return 0
    if "," in data_url:
        payload = data_url.split(",", 1)[1]
    else:
        payload = data_url
    return (len(payload) * 3) // 4


def _pil_to_jpeg_data_url_and_bytes(pil_img, quality: int = 85) -> Tuple[str, int]:
    """Encode PIL Image to JPEG data URL with specified quality.
    
    Args:
        pil_img: PIL Image object.
        quality: JPEG compression quality (1-100).
    
    Returns:
        Tuple of (data_url, jpeg_bytes) where jpeg_bytes is the real JPEG file size.
    """
    import base64
    import io
    from PIL import Image as PILImage
    
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    jpeg_blob = buffer.getvalue()
    jpeg_bytes = len(jpeg_blob)
    b64_str = base64.b64encode(jpeg_blob).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"
    return (data_url, jpeg_bytes)


def _guaranteed_under_budget(
    pil_img,
    max_bytes: int,
    start_edge: int = 64,
    quality: int = 50,
    max_iters: int = 200,
    no_progress_patience: int = 10,
) -> Tuple[str, int]:
    """Force shrink until under budget.

    Design goals:
    - scale-first: prioritize shrinking size; quality stays mostly unchanged
    - boundary-hugging: each round edge *= 0.9, aiming to just fit <= max_bytes
    - quality only when necessary: only reduce q when edge is already at min_edge
    - absolute termination: max_iters / stall detection / explicit raise when impossible
    - never return over-budget: always check b <= max_bytes before return

    Requirements:
    - Depends on: _pil_to_jpeg_data_url_and_bytes(pil_img, q) -> (data_url, jpeg_bytes)
    """
    from PIL import Image as PILImage

    if pil_img is None:
        raise ValueError("_guaranteed_under_budget: pil_img is None")

    if max_bytes <= 0:
        raise ValueError(f"_guaranteed_under_budget: max_bytes must be > 0, got {max_bytes}")

    # --- Tunable constants ---
    MIN_EDGE = 8          # Semantic minimum edge: don't shrink below this (absolute termination)
    MIN_Q = 5             # Minimum quality (rarely used)
    EDGE_DECAY = 0.90     # Each round shrink to 90%
    AGGRESSIVE_DECAY = 0.80  # More aggressive shrink when stalled
    Q_STEP = 2            # Only reduce q by this when edge is at MIN_EDGE
    STALL_DROP_RATIO = 0.01  # 1%: less than this improvement counts as stall
    SAME_SIZE_PATIENCE = 5   # Force push after this many rounds of same size due to rounding

    w0, h0 = pil_img.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"_guaranteed_under_budget: invalid image size {w0}x{h0}")

    # edge is the floating-point control for target max edge (decayed by 0.9), actual resize uses int(edge)
    edge = float(max(start_edge, MIN_EDGE))
    q = int(quality)

    # Track last valid encode info to avoid UnboundLocalError
    last_b: Optional[int] = None
    last_state: Optional[Tuple[int, int, int]] = None  # (new_w, new_h, q)

    # Stall / rounding detection
    prev_b: Optional[int] = None
    stalled = 0
    same_state = 0

    # Never return over-budget; if budget is unreasonably small, raise instead of returning garbage
    for it in range(1, max_iters + 1):
        # Compute new size preserving aspect ratio (max edge ~ edge)
        if w0 >= h0:
            new_w = max(MIN_EDGE, int(edge))
            new_h = max(MIN_EDGE, int(h0 * new_w / w0))
        else:
            new_h = max(MIN_EDGE, int(edge))
            new_w = max(MIN_EDGE, int(w0 * new_h / h0))

        # Due to rounding, (new_w, new_h, q) might not change -> advance edge/q directly to avoid no-op
        state = (new_w, new_h, q)
        if state == last_state:
            same_state += 1
            if same_state >= SAME_SIZE_PATIENCE:
                # Force push: prefer shrinking edge more; if at MIN_EDGE, reduce q
                if edge > MIN_EDGE:
                    edge = max(MIN_EDGE, edge * AGGRESSIVE_DECAY)
                else:
                    q = max(MIN_Q, q - max(Q_STEP, 5))
                same_state = 0
            else:
                # Normal push (gentle)
                if edge > MIN_EDGE:
                    edge = max(MIN_EDGE, edge * EDGE_DECAY)
                else:
                    q = max(MIN_Q, q - Q_STEP)
            continue

        last_state = state
        same_state = 0

        # Resize + encode
        img_s = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        u, b = _pil_to_jpeg_data_url_and_bytes(img_s, q)
        last_b = b

        print(f"[_guaranteed_under_budget] iter={it:03d} edge={edge:.2f} size={new_w}x{new_h} q={q} bytes={b:,}")

        # Success
        if b <= max_bytes:
            return u, b

        # Stall detection (use integers to avoid float boundary issues)
        if prev_b is not None:
            # If b didn't drop by at least 1% (i.e., b >= 0.99 * prev_b), count as stall
            if b * 100 >= prev_b * 99:
                stalled += 1
            else:
                stalled = 0
        prev_b = b

        # Stalled too long: do aggressive push
        if stalled >= no_progress_patience:
            print(f"[_guaranteed_under_budget] stalled={stalled} -> aggressive push")
            stalled = 0
            if edge > MIN_EDGE:
                edge = max(MIN_EDGE, edge * AGGRESSIVE_DECAY)
            else:
                q = max(MIN_Q, q - max(Q_STEP, 5))
            continue

        # Normal advancement: scale-first; only reduce quality when edge is at MIN_EDGE
        if edge > MIN_EDGE:
            edge = max(MIN_EDGE, edge * EDGE_DECAY)
        else:
            # At minimum semantic edge but still over: have to reduce q
            if q > MIN_Q:
                q = max(MIN_Q, q - Q_STEP)
            else:
                # Both edge and q at minimum but still over: budget is truly unreachable
                raise RuntimeError(
                    f"_guaranteed_under_budget: cannot fit under budget.\n"
                    f"  max_bytes={max_bytes:,}\n"
                    f"  last_bytes={b:,}\n"
                    f"  size={new_w}x{new_h}\n"
                    f"  edge={edge:.2f} (MIN_EDGE={MIN_EDGE})\n"
                    f"  q={q} (MIN_Q={MIN_Q})\n"
                )

    # max_iters exceeded: explicit error (absolute termination)
    raise RuntimeError(
        f"_guaranteed_under_budget: exceeded max_iters={max_iters}.\n"
        f"  max_bytes={max_bytes:,}\n"
        f"  last_bytes={(last_b if last_b is not None else 'unknown')}\n"
        f"  edge={edge:.2f}\n"
        f"  q={q}\n"
        f"  MIN_EDGE={MIN_EDGE}, MIN_Q={MIN_Q}\n"
    )



def fit_image_to_budget(
    data_url: str,
    max_bytes: int,
    jpeg_quality: int = 85,
    scale_decay: float = 0.9,
    decode_func: Optional[Callable] = None,
) -> str:
    """Fit image to byte budget using coarse search + binary search for optimal scale.
    
    This function finds the largest possible image (highest resolution) that fits
    within the byte budget, maximizing visual quality while respecting size limits.
    
    Algorithm:
    1. Shortcut: only if input is JPEG AND estimate <= max_bytes, return as-is
    2. Decode data_url to PIL once, then encode to JPEG to get real bytes
    3. Coarse search: prioritize scale*=0.9; only try q-2 when very close (<=1.02)
    4. Binary search (16 iterations): find optimal scale between last_over and first_under
    5. Use real jpeg_bytes (not estimate) for all comparisons
    6. Guaranteed fallback: loop shrink until under budget (never return over-budget)
    
    Args:
        data_url: Base64 data URL of the image.
        max_bytes: Maximum byte size for the output (JPEG file size).
        jpeg_quality: Initial JPEG quality (default 85).
        scale_decay: Scale multiplier for coarse search (default 0.9).
        decode_func: Optional function to decode base64 to PIL.Image.
    
    Returns:
        Data URL string that fits within max_bytes (guaranteed).
    """
    from PIL import Image as PILImage, ImageFile, ImageOps
    import io
    import base64
    
    # Constants
    CLOSE_RATIO = 1.02  # Only try q reduction when very close to boundary
    MIN_QUALITY = 50
    MIN_EDGE = 16  # Semantic termination: don't go below 16px
    
    # Validate input - strict contract: must be data URL
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        raise ValueError("fit_image_to_budget expects a data:image... data URL")
    
    # Step 1: Shortcut only for JPEG that is already under budget
    is_jpeg = data_url.startswith("data:image/jpeg")
    cur_bytes_est = estimate_bytes_from_data_url(data_url)
    print(f"[fit_image_to_budget] Original est_bytes: {cur_bytes_est:,}, max_bytes: {max_bytes:,}, is_jpeg: {is_jpeg}")
    if is_jpeg and cur_bytes_est <= max_bytes:
        return data_url
    
    # Step 2: Decode to PIL once
    # Set limits to handle very large images (same as resize_and_compress_image)
    max_decode_pixels = 1024 * 1024 * 256  # 256 megapixels (16384 x 16384)
    old_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    old_max_pixels = PILImage.MAX_IMAGE_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PILImage.MAX_IMAGE_PIXELS = max_decode_pixels
    
    # Track best under-budget result as fallback
    best_under_url: Optional[str] = None
    best_under_bytes: Optional[int] = None
    pil_img = None
    
    try:
        if decode_func is not None:
            pil_img = decode_func(data_url)
        else:
            # Internal decode
            if ";base64," in data_url:
                b64_data = data_url.split(";base64,")[1]
            elif "," in data_url:
                b64_data = data_url.split(",")[1]
            else:
                print(f"[fit_image_to_budget] Cannot parse data URL format")
                raise ValueError("Cannot parse data URL format")
            b64_data = b64_data.strip()
            b64_data += "=" * (-len(b64_data) % 4)
            try:
                image_bytes = base64.b64decode(b64_data, validate=True)
            except Exception:
                image_bytes = base64.b64decode(b64_data, validate=False)
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            pil_img.load()
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            pil_img = pil_img.convert("RGB")
        
        if pil_img is None:
            print(f"[fit_image_to_budget] Decode failed")
            raise ValueError("Decode failed")
        
        w0, h0 = pil_img.size
        print(f"[fit_image_to_budget] Original size: {w0}x{h0}")
        
        # Step 2a: Fast path for extremely large images (avoid slow iterations on huge images)
        # If image exceeds 16 megapixels, pre-shrink to a reasonable size first
        FAST_PATH_THRESHOLD = 16 * 1024 * 1024  # 16 megapixels
        MAX_FAST_PATH_EDGE = 4096  # Target max edge for fast path
        
        total_pixels = w0 * h0
        if total_pixels > FAST_PATH_THRESHOLD:
            # Calculate scale to bring the larger edge down to MAX_FAST_PATH_EDGE
            fast_scale = MAX_FAST_PATH_EDGE / max(w0, h0)
            if fast_scale < 1.0:
                new_w = max(1, int(w0 * fast_scale))
                new_h = max(1, int(h0 * fast_scale))
                print(f"[fit_image_to_budget] Fast path: {w0}x{h0} ({total_pixels/1e6:.1f}MP) -> {new_w}x{new_h} (scale={fast_scale:.4f})")
                pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                w0, h0 = new_w, new_h
        
        # Step 2b: Encode once at full scale to get real bytes and check if already under
        q = max(MIN_QUALITY, min(int(jpeg_quality), 95))  # Clamp to valid range
        u0, b0 = _pil_to_jpeg_data_url_and_bytes(pil_img, q)
        print(f"[fit_image_to_budget] Initial encode: q={q}, bytes={b0:,}")
        if b0 <= max_bytes:
            return u0
        
        # Step 3: Coarse search with scale-first strategy
        scale = 1.0
        last_over = (scale, b0)
        first_under = None
        
        iteration = 0
        max_coarse_iter = 60
        
        while iteration < max_coarse_iter:
            iteration += 1
            
            # Reduce scale
            scale *= scale_decay
            new_w = max(1, int(w0 * scale))
            new_h = max(1, int(h0 * scale))
            
            # Semantic termination: if size <= MIN_EDGE, go to guaranteed fallback
            if new_w <= MIN_EDGE or new_h <= MIN_EDGE:
                print(f"[fit_image_to_budget] Size {new_w}x{new_h} <= {MIN_EDGE}, entering guaranteed fallback")
                break
            
            # Resize image
            img_s = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            
            # Encode JPEG and get real bytes
            u, b = _pil_to_jpeg_data_url_and_bytes(img_s, q)
            
            print(f"[fit_image_to_budget] Coarse iter={iteration}: scale={scale:.4f}, size={new_w}x{new_h}, q={q}, bytes={b:,}")
            
            if b > max_bytes:
                last_over = (scale, b)
                
                # Strategy: only try quality reduction when VERY close to boundary
                over_ratio = b / max_bytes
                if over_ratio <= CLOSE_RATIO and q > MIN_QUALITY:
                    q2 = max(MIN_QUALITY, q - 2)
                    u2, b2 = _pil_to_jpeg_data_url_and_bytes(img_s, q2)
                    print(f"[fit_image_to_budget] Quality fine-tune: q={q2}, bytes={b2:,}")
                    
                    if b2 <= max_bytes:
                        first_under = (scale, u2, b2)
                        best_under_url, best_under_bytes = u2, b2
                        q = q2
                        break
                    else:
                        last_over = (scale, b2)
                        q = q2
            else:
                first_under = (scale, u, b)
                best_under_url, best_under_bytes = u, b
                break
        
        # If coarse search failed to find under, use guaranteed fallback
        if first_under is None:
            print(f"[fit_image_to_budget] Coarse search exhausted, using guaranteed fallback")
            u_fallback, b_fallback = _guaranteed_under_budget(pil_img, max_bytes, start_edge=64, quality=MIN_QUALITY)
            return u_fallback
        
        # Step 4: Binary search between last_over.scale and first_under.scale
        low = first_under[0]
        high = last_over[0]
        best = first_under
        
        for i in range(16):
            mid = (low + high) / 2.0
            new_w = max(1, int(w0 * mid))
            new_h = max(1, int(h0 * mid))
            
            if mid < 1.0:
                img_s = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            else:
                img_s = pil_img
            
            u, b = _pil_to_jpeg_data_url_and_bytes(img_s, q)
            
            print(f"[fit_image_to_budget] Binary iter={i+1}: scale={mid:.4f}, size={new_w}x{new_h}, bytes={b:,}")
            
            if b <= max_bytes:
                best = (mid, u, b)
                best_under_url, best_under_bytes = u, b
                low = mid
            else:
                high = mid
        
        final_scale, final_url, final_bytes = best
        ratio = final_bytes / max_bytes if max_bytes > 0 else 0
        print(f"[fit_image_to_budget] Final: scale={final_scale:.4f}, q={q}, bytes={final_bytes:,}, ratio={ratio:.4f}")
        return final_url
        
    except Exception as e:
        print(f"[fit_image_to_budget] Error: {e}")
        # Never return over-budget - use best_under if available
        if best_under_url is not None:
            print(f"[fit_image_to_budget] Returning best_under: bytes={best_under_bytes:,}")
            return best_under_url
        # Use guaranteed fallback if we have pil_img
        if pil_img is not None:
            print(f"[fit_image_to_budget] Using guaranteed fallback after error")
            u_fallback, _ = _guaranteed_under_budget(pil_img, max_bytes, quality=MIN_QUALITY)
            return u_fallback
        # This should never happen, but if it does, raise to make it visible
        raise RuntimeError(f"fit_image_to_budget failed completely: {e}")
    finally:
        PILImage.MAX_IMAGE_PIXELS = old_max_pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = old_truncated


def _safe_image_for_mllm(
    img_url: str,
    max_bytes: int,
    max_edge: int,
    jpeg_quality: int,
    decode_func: Optional[Callable] = None,
) -> Optional[str]:
    """Return a safe data URL within size limits, or None if not usable."""
    if not isinstance(img_url, str) or not img_url.startswith("data:image"):
        return None
    if estimate_bytes_from_data_url(img_url) <= max_bytes:
        return img_url
    compressed = resize_and_compress_image(
        img_url,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
        decode_func=decode_func,
    )
    if not isinstance(compressed, str) or not compressed.startswith("data:image"):
        return None
    if estimate_bytes_from_data_url(compressed) > max_bytes:
        return None
    return compressed


def _redact_tool_args(args: Dict[str, Any], max_len: int = 200, max_list_items: int = 5) -> Dict[str, Any]:
    """Redact image/base64/large blob fields in tool args recursively.
    
    Covers: image_base64, processed_image_base64, mask, seg, image, data_url, url,
    and any key containing 'base64' substring.
    Preserves scalar parameters like sigma, scale, strength, roi_json for debugging.
    """
    # Explicit keys to always redact
    REDACT_KEYS = {"image_base64", "processed_image_base64", "mask", "seg", "image", "data_url", "url"}
    
    def _redact_value(key: str, value: Any) -> Any:
        key_lower = key.lower() if key else ""
        # Key-level redaction: explicit sensitive keys or any key containing 'base64'
        if key_lower in REDACT_KEYS or "base64" in key_lower:
            return "[REDACTED]"
        # Value-level handling
        if isinstance(value, str):
            # Redact long strings, data URLs, or strings containing base64 patterns
            if len(value) > max_len or value.startswith("data:") or "base64," in value:
                return f"[REDACTED len={len(value)}]"
            return value
        if isinstance(value, dict):
            # Recursively redact dict
            return {k: _redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            # Keep first N items, recursively redact each
            truncated = value[:max_list_items]
            redacted_list = [_redact_value("", item) for item in truncated]
            if len(value) > max_list_items:
                redacted_list.append(f"... ({len(value) - max_list_items} more items)")
            return redacted_list
        # Scalars (int, float, bool, None) pass through
        return value
    
    return {k: _redact_value(k, v) for k, v in args.items()}


def format_executed_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Format executed tool calls into a readable string, redacting image data."""
    if not tool_calls:
        return ""
    
    tool_calls_info = []
    for tc in tool_calls:
        tool_name = tc.get("tool_name", "unknown")
        args = tc.get("args", {})
        safe_args = _redact_tool_args(args)
        tool_calls_info.append(f"- {tool_name}({safe_args})")
    
    return "Executed tool calls:\n" + "\n".join(tool_calls_info)


def format_planner_reflector_sequence(
    potential_anomalies_list: List[str],
    heuristic_prompt_list: List[str],
    executed_tool_calls: List[Dict[str, Any]],
    reasoner_judgments: List[Dict[str, Any]],
    num_tool_calls: int,
) -> str:
    """
    Format the sequence of planner/reflector outputs, tool calls, and reasoner judgments.
    
    Args:
        potential_anomalies_list: List of potential anomalies from planner/reflector
        heuristic_prompt_list: List of heuristic prompts from planner/reflector
        executed_tool_calls: List of executed tool calls with 'node' field
        reasoner_judgments: List of reasoner judgments with 'call_index', 'result', 'reason'
        num_tool_calls: Total number of tool calls made so far
    
    Returns:
        Formatted string showing the analysis sequence
    """
    parts = []
    
    # Group tool calls by node (planner/reflector)
    tool_calls_by_node = {"planner": [], "reflector": []}
    for tc in executed_tool_calls:
        node = tc.get("node", "unknown")
        if node in tool_calls_by_node:
            tool_calls_by_node[node].append(tc)
        else:
            # If node is unknown, try to infer from order
            # First tool calls are usually from planner
            if len(tool_calls_by_node["planner"]) == 0:
                tool_calls_by_node["planner"].append(tc)
            else:
                tool_calls_by_node["reflector"].append(tc)
    
    # Build rounds from potential_anomalies and heuristic_prompt
    # Each item in the list represents one round (planner or reflector)
    rounds = []
    max_rounds = max(len(potential_anomalies_list), len(heuristic_prompt_list), 1)
    
    for i in range(max_rounds):
        round_info = {
            "round_index": i + 1,
            "node": "unknown",
            "potential_anomalies": "",
            "heuristic_prompt": "",
            "tool_calls": [],
            "reasoner_judgment": None,
        }
        
        # Extract from potential_anomalies
        if i < len(potential_anomalies_list):
            anomaly_text = potential_anomalies_list[i]
            if isinstance(anomaly_text, str):
                # Check for prefix
                if anomaly_text.startswith("[planner]"):
                    round_info["node"] = "planner"
                    round_info["potential_anomalies"] = anomaly_text.replace("[planner]\n", "").replace("[planner]", "").strip()
                elif anomaly_text.startswith("[reflector]"):
                    round_info["node"] = "reflector"
                    round_info["potential_anomalies"] = anomaly_text.replace("[reflector]\n", "").replace("[reflector]", "").strip()
                else:
                    # No prefix: first is planner, rest are reflector
                    round_info["node"] = "planner" if i == 0 else "reflector"
                    round_info["potential_anomalies"] = anomaly_text.strip()
        
        # Extract from heuristic_prompt
        if i < len(heuristic_prompt_list):
            prompt_text = heuristic_prompt_list[i]
            if isinstance(prompt_text, str):
                # Check for prefix
                if prompt_text.startswith("[planner]"):
                    if round_info["node"] == "unknown":
                        round_info["node"] = "planner"
                    round_info["heuristic_prompt"] = prompt_text.replace("[planner]\n", "").replace("[planner]", "").strip()
                elif prompt_text.startswith("[reflector]"):
                    if round_info["node"] == "unknown":
                        round_info["node"] = "reflector"
                    round_info["heuristic_prompt"] = prompt_text.replace("[reflector]\n", "").replace("[reflector]", "").strip()
                else:
                    if round_info["node"] == "unknown":
                        round_info["node"] = "planner" if i == 0 else "reflector"
                    round_info["heuristic_prompt"] = prompt_text.strip()
        
        # Assign tool calls: planner gets first set, reflector gets subsequent sets
        if round_info["node"] == "planner" and i == 0:
            round_info["tool_calls"] = tool_calls_by_node.get("planner", [])
        elif round_info["node"] == "reflector":
            # For reflector rounds, we need to distribute tool calls
            # Simple approach: if this is the first reflector round (i==1), take first reflector tools
            # This is a simplification - in reality, we'd need to track which tools belong to which round
            reflector_tools = tool_calls_by_node.get("reflector", [])
            if reflector_tools:
                # For now, show all reflector tools in the first reflector round
                # A more sophisticated version would track rounds more carefully
                if i == 1:  # First reflector round
                    round_info["tool_calls"] = reflector_tools
        
        rounds.append(round_info)
    
    # Match reasoner judgments to rounds
    # Reasoner is called after each tool execution round
    for judgment in reasoner_judgments:
        call_index = judgment.get("call_index", 0)
        # call_index is 1-based
        if 1 <= call_index <= len(rounds):
            rounds[call_index - 1]["reasoner_judgment"] = judgment
    
    # Format the output
    parts.append(f"Analysis Sequence (Total Tool Calls: {num_tool_calls}):")
    parts.append("=" * 80)
    
    for round_info in rounds:
        # Skip completely empty rounds
        if (round_info["node"] == "unknown" and 
            not round_info["potential_anomalies"] and 
            not round_info["heuristic_prompt"] and
            not round_info["tool_calls"]):
            continue
        
        node_name = round_info["node"].upper() if round_info["node"] != "unknown" else "UNKNOWN"
        round_num = round_info["round_index"]
        parts.append(f"\n--- Round {round_num}: {node_name} Analysis ---")
        
        if round_info["potential_anomalies"]:
            parts.append(f"\nPotential Anomalies:\n{round_info['potential_anomalies']}")
        
        if round_info["heuristic_prompt"]:
            parts.append(f"\nHeuristic Prompt:\n{round_info['heuristic_prompt']}")
        
        tool_calls = round_info["tool_calls"]
        if tool_calls:
            parts.append(f"\nTool Calls ({len(tool_calls)} tool(s)):")
            for tc in tool_calls:
                tool_name = tc.get("tool_name", "unknown")
                args = tc.get("args", {})
                safe_args = _redact_tool_args(args)
                parts.append(f"  - {tool_name}({safe_args})")
        else:
            parts.append("\nTool Calls: (none)")
        
        judgment = round_info["reasoner_judgment"]
        if judgment:
            result = judgment.get("result", "unknown")
            reason = judgment.get("reason", "")
            forced = judgment.get("forced", False)
            forced_str = " [FORCED]" if forced else ""
            parts.append(f"\nReasoner Judgment{forced_str}:")
            parts.append(f"  Result: {result}")
            if reason:
                parts.append(f"  Reason: {reason}")
        else:
            parts.append("\nReasoner Judgment: (pending)")
    
    return "\n".join(parts)


# ================================================================
# Token counting utilities (for tracking model usage)
# ================================================================

def estimate_message_tokens(msg: BaseMessage) -> int:
    """
    Estimate token count for a single message.
    
    This is a rough estimate. For accurate counts, use the model's tokenizer.
    Actual token counts are available from QwenBatcher via additional_kwargs.
    
    Args:
        msg: A BaseMessage object
        
    Returns:
        Estimated token count
    """
    content = getattr(msg, "content", None)
    tokens = 50  # Base overhead per message
    
    if isinstance(content, str):
        # Rough estimate: 1 token ≈ 4 characters for English
        tokens += len(content) // 4
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str):
                    tokens += len(text) // 4
                # Image URLs are very long (base64 encoded)
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    # Base64 images are roughly 100K-500K tokens
                    if url.startswith("data:image"):
                        # Rough estimate: ~150K tokens per image
                        tokens += 150000
            elif isinstance(item, str):
                tokens += len(item) // 4
    elif content is not None:
        tokens += len(str(content)) // 4
    
    # Account for tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        # Each tool call adds overhead
        tokens += len(msg.tool_calls) * 100
    
    return tokens


def estimate_messages_tokens(messages: List[BaseMessage]) -> int:
    """
    Estimate total token count for a list of messages.
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        Estimated total token count
    """
    return sum(estimate_message_tokens(msg) for msg in messages)


def count_input_output_tokens(
    input_messages: List[BaseMessage],
    output_message: Optional[BaseMessage] = None,
) -> Dict[str, int]:
    """
    Count tokens for input messages and optional output message.
    
    This is an estimation method. For accurate counts, use actual token counts
    from QwenBatcher (available in output_message.additional_kwargs).
    
    Args:
        input_messages: List of input messages
        output_message: Optional output message
        
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens' keys
    """
    input_tokens = estimate_messages_tokens(input_messages)
    output_tokens = estimate_message_tokens(output_message) if output_message else 0
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


# ================================================================
# Qwen Micro-Batcher
# ================================================================

class QwenBatcher:
    """
    Micro-batching for Qwen-VL.
    Automatically batches concurrent ainvoke() calls into one forward pass.

    Parameters:
      max_batch_size: maximum batch size before immediate flush
      max_wait_ms: max waiting time before flushing incomplete batches
      verbose: print batch size when running
    """

    def __init__(
        self,
        model,
        processor,
        *,
        max_batch_size: int = 8,
        max_wait_ms: int = 5,
        verbose: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.verbose = verbose

        self.queue: List[tuple] = []  # (qmsgs, gen_kwargs, future)
        self.lock = asyncio.Lock()  # protects _queue
        self._infer_lock = asyncio.Lock()  # protects model.generate() to prevent concurrent calls
        self.flush_task: asyncio.Task | None = None

    async def submit(self, qmsgs: List[dict], gen_kwargs: Dict[str, Any]) -> tuple[str, int, int]:
        """
        Submit one sample into the batcher and return future result with token counts.
        
        Returns:
            (text, input_tokens, output_tokens) tuple
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()  # Future will contain (text, input_tokens, output_tokens)

        async with self.lock:
            self.queue.append((qmsgs, gen_kwargs, fut))

            # If batch is full → flush immediately
            if len(self.queue) >= self.max_batch_size:
                batch = self.queue
                self.queue = []
                if self.flush_task:
                    self.flush_task.cancel()
                    self.flush_task = None

                asyncio.create_task(self.run_batch(batch))

            else:
                # Not full → start / keep the timeout task
                if self.flush_task is None:
                    self.flush_task = asyncio.create_task(self.flush_later())

        return await fut

    async def flush_later(self):
        """Flush remaining items when max_wait expires."""
        try:
            await asyncio.sleep(self.max_wait)
            async with self.lock:
                if not self.queue:
                    self.flush_task = None
                    return
                batch = self.queue
                self.queue = []
                self.flush_task = None

            asyncio.create_task(self.run_batch(batch))

        except asyncio.CancelledError:
            # Cancelled because batch was filled earlier
            return

    async def run_batch(self, batch):
        """Run actual batched forward.
        
        Serialized by _infer_lock to ensure only one batch calls model.generate() at a time.
        This prevents concurrent generate() calls on the same model instance, which can cause
        GPU/HF issues.
        """
        # Unpack batch (pure CPU operation, can be done outside lock)
        qmsgs_list, gen_kwargs_list, futs = zip(*batch)
        gen_kwargs = gen_kwargs_list[0] if gen_kwargs_list else {}

        # Debug print (pure CPU operation, moved outside lock for better throughput)
        if self.verbose:
            print(f"[QwenBatcher] Running batch size = {len(batch)}")
        
            # Debug: Count images in batch
            # qmsgs_list is a list of message lists, where each element is a list of dicts for one query
            total_images = 0
            for qmsgs in qmsgs_list:  # qmsgs is a list of message dicts for one query
                if isinstance(qmsgs, list):
                    for msg in qmsgs:  # msg is a dict with "role" and "content"
                        if isinstance(msg, dict):
                            content = msg.get("content", [])
                            if isinstance(content, list):
                                images_in_msg = sum(1 for item in content 
                                                  if isinstance(item, dict) and item.get("type") in ("image", "image_url"))
                                total_images += images_in_msg
                            elif isinstance(content, dict) and content.get("type") in ("image", "image_url"):
                                total_images += 1
            print(f"[DEBUG QwenBatcher] Total images in batch: {total_images} (batch_size={len(batch)}, avg={total_images/len(batch) if len(batch) > 0 else 0:.2f} images per query)")

        # Ensure only one batch calls model.generate() at a time
        # Lock protects tokenize and generate operations (GPU/CPU intensive)
        async with self._infer_lock:
            try:
                # ==== Tokenize ====
                inputs = self.processor.apply_chat_template(
                    list(qmsgs_list),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,  # Enable padding for batched inputs
                )
                
                # Debug: Check tensor sizes after tokenization (quick CPU check, keep in lock for consistency)
                if self.verbose and "input_ids" in inputs:
                    input_ids = inputs["input_ids"]
                    total_elements = input_ids.numel()
                    print(f"[DEBUG QwenBatcher] After tokenization: input_ids shape={input_ids.shape}, total_elements={total_elements:,} (INT_MAX={2**31-1:,})")
                    if total_elements > 2**31 - 1:
                        print(f"[WARNING QwenBatcher] Tensor size exceeds INT_MAX! This will cause nonzero() error.")

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # ==== Batched generation ====
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
                        do_sample=gen_kwargs.get("do_sample", False),
                        temperature=gen_kwargs.get("temperature", None),
                        top_p=gen_kwargs.get("top_p", None),
                        use_cache=True,
                    )

                trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
                texts = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # Calculate actual token counts for each request (accounting for padding)
                # Note: Returns token COUNTS (integers), not actual token IDs
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
                
                # If attention_mask is available, use it to count actual tokens (excluding padding)
                # attention_mask.sum(dim=1) gives the actual number of tokens per sample (excluding padding)
                if attention_mask is not None:
                    input_token_counts = attention_mask.sum(dim=1).cpu().tolist()  # List[int]: actual token counts
                else:
                    # Fallback: use sequence length (will be same for all in batch due to padding)
                    input_token_counts = [input_ids.shape[1]] * len(batch)  # List[int]: sequence lengths
                
                # Output tokens = length of generated sequence (trimmed, excluding input tokens)
                # len(trimmed_seq) gives the number of newly generated tokens
                output_token_counts = [len(trimmed_seq) for trimmed_seq in trimmed]  # List[int]: generated token counts

                # ==== Return results with token counts ====
                # Returns: (text: str, input_tokens: int, output_tokens: int)
                # - input_tokens: actual number of input tokens (excluding padding)
                # - output_tokens: actual number of generated tokens
                for fut, txt, in_tokens, out_tokens in zip(futs, texts, input_token_counts, output_token_counts):
                    if not fut.cancelled():
                        fut.set_result((txt.strip(), in_tokens, out_tokens))
            except Exception as e:
                # Handle errors gracefully - set exception on all futures
                error_msg = f"Error in QwenBatcher.run_batch: {str(e)}"
                if self.verbose:
                    print(f"[QwenBatcher] {error_msg}")
                    import traceback
                    traceback.print_exc()
                
                # Set exception on all futures so callers don't hang
                for fut in futs:
                    if not fut.cancelled():
                        fut.set_exception(e)


# ================================================================
# Qwen-VL Wrapper with micro-batching
# ================================================================

_QWEN_CHAT_SINGLETONS: Dict[str, BaseChatModel] = {}
_QWEN_CHAT_SINGLETONS_LOCK: threading.Lock = threading.Lock()
_QWEN_BATCHER: Optional[QwenBatcher] = None


class QwenVLChat(BaseChatModel):
    """LangChain-compatible wrapper for Qwen-VL with micro-batching."""

    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _device: Any = PrivateAttr()
    _batcher: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: Any, processor: Any, *, max_batch_size: int = 16, max_wait_ms: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._processor = processor
        self._device = next(self._model.parameters()).device

        # Init global batcher with configurable parameters
        # Optimized defaults: larger batch size and longer wait time for better throughput
        global _QWEN_BATCHER
        if _QWEN_BATCHER is None:
            _QWEN_BATCHER = QwenBatcher(
                model=self._model,
                processor=self._processor,
                max_batch_size=max_batch_size,
                max_wait_ms=max_wait_ms,
                verbose=True,
            )
        self._batcher = _QWEN_BATCHER

    @property
    def _llm_type(self) -> str:
        return "qwen_vl"

    # ------------------------------------------------------------
    # Message normalization (LC → Qwen format)
    # ------------------------------------------------------------

    def _lc_to_qwen_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages into Qwen chat template format."""
        import warnings
        qmsgs: List[Dict[str, Any]] = []

        for m in messages:
            # Track if original message should contain image
            original_has_image = False
            # Handle dict input (e.g., {"role": "user", "content": "..."})
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                # Map role from BaseMessage
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, SystemMessage):
                    role = "system"
                else:
                    role = "user"

                content = m.content
            mm_items: List[dict] = []

            # Text
            if isinstance(content, str):
                if content.strip():
                    mm_items.append({"type": "text", "text": content})

            # Single multimodal dict
            elif isinstance(content, dict):
                if content.get("type") in ("image", "image_url"):
                    original_has_image = True
                mm_items.extend(self._normalize_one_mm_item(content))

            # List of multimodal elements
            elif isinstance(content, list):
                for it in content:
                    if isinstance(it, str):
                        if it.strip():
                            mm_items.append({"type": "text", "text": it})
                    elif isinstance(it, dict):
                        # Check if this dict should contain image
                        if it.get("type") in ("image", "image_url"):
                            original_has_image = True
                        mm_items.extend(self._normalize_one_mm_item(it))
                    else:
                        # Handle PIL.Image objects
                        try:
                            from PIL import Image
                            if isinstance(it, Image.Image):
                                original_has_image = True
                                mm_items.append({"type": "image", "image": it})
                                continue
                        except ImportError:
                            pass
                        # Fallback: convert to text
                        txt = str(it).strip()
                        if txt:
                            mm_items.append({"type": "text", "text": txt})

            else:
                txt = str(content).strip()
                if txt:
                    mm_items.append({"type": "text", "text": txt})

            qmsgs.append({"role": role, "content": mm_items or [{"type": "text", "text": ""}]})
            
            # Verify image was preserved if original had image
            if original_has_image:
                has_image_in_output = any(
                    item.get("type") == "image" and "image" in item 
                    for item in mm_items
                )
                if not has_image_in_output:
                    warnings.warn(
                        f"WARNING: Original message contained image data, but image was lost during conversion. "
                        f"Message role: {role}, content types: {[item.get('type') for item in mm_items]}"
                    )

        return qmsgs

    def _normalize_one_mm_item(self, it: dict) -> List[dict]:
        """Normalize one multimodal item."""
        out: List[dict] = []
        t = it.get("type")

        if t == "image":
            out.append(it)

        elif t == "image_url":
            # Convert image_url (base64 data URL) to PIL.Image format
            try:
                from PIL import Image, UnidentifiedImageError
                import base64
                from io import BytesIO
                
                url = it.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    # Extract base64 data from data URL
                    # Format: "data:image/png;base64,<base64_data>"
                    header, encoded = url.split(",", 1)
                    encoded = encoded.strip()
                    encoded += "=" * (-len(encoded) % 4)
                    img_data = base64.b64decode(encoded, validate=True)
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    out.append({"type": "image", "image": img})
                elif url.startswith("http://") or url.startswith("https://"):
                    # HTTP URL - could be handled if needed, but for now skip
                    pass
            except Exception as e:
                url = it.get("image_url", {}).get("url", "")
                encoded_preview = ""
                encoded_len = 0
                if isinstance(url, str) and "," in url:
                    _, encoded_part = url.split(",", 1)
                    encoded_part = encoded_part.strip()
                    encoded_len = len(encoded_part)
                    encoded_preview = encoded_part[:30]
                err_type = type(e).__name__
                print(
                    f"[MM_IMAGE_DECODE] failed type={err_type} url_len={len(url) if isinstance(url, str) else 0} "
                    f"encoded_len={encoded_len} preview={encoded_preview!r}"
                )

        elif t == "text":
            text = it.get("text", "")
            if isinstance(text, str) and text.strip():
                out.append({"type": "text", "text": text})

        return out

    # ------------------------------------------------------------
    # Micro-batched ainvoke()
    # ------------------------------------------------------------

    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Use micro-batcher for async invoke."""
        qmsgs = self._lc_to_qwen_messages(messages)
        result = await self._batcher.submit(qmsgs, kwargs)
        # result is now (text, input_tokens, output_tokens) tuple
        if isinstance(result, tuple) and len(result) == 3:
            text, input_tokens, output_tokens = result
            # Store token counts in the message for later retrieval
            msg = AIMessage(content=text)
            # Use additional_kwargs to store token counts (non-standard but accessible)
            if not hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs = {}
            msg.additional_kwargs["input_tokens"] = input_tokens
            msg.additional_kwargs["output_tokens"] = output_tokens
            return msg
        else:
            # Fallback for backward compatibility
            text = result if isinstance(result, str) else result[0]
            return AIMessage(content=text)

    # ------------------------------------------------------------
    # Legacy single-sample generate (unused by graph)
    # ------------------------------------------------------------

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Fallback single-sample generation."""
        qmsgs = self._lc_to_qwen_messages(messages)

        inputs = self._processor.apply_chat_template(
            [qmsgs],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        device = self._device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", None),
                top_p=kwargs.get("top_p", None),
                use_cache=True,
            )

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# ================================================================
# Model loading
# ================================================================

def load_vl_model(
    model_path: str,
    *,
    trust_remote_code: bool = True,
    use_singleton: bool = True,
    attn_implementation: Optional[str] = "flash_attention_2",
    max_batch_size: int = 16,
    max_wait_ms: int = 50,
) -> BaseChatModel:
    """Load local Qwen-VL and wrap it in QwenVLChat (thread-safe with singleton pattern)."""
    resolved = str(Path(model_path).resolve())
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    
    # Thread-safe singleton check and load
    if use_singleton:
        # Fast path: check if already loaded (read-only, safe without lock)
        if resolved in _QWEN_CHAT_SINGLETONS:
            return _QWEN_CHAT_SINGLETONS[resolved]
        
        # Slow path: need to load, use lock to prevent concurrent loads
        with _QWEN_CHAT_SINGLETONS_LOCK:
            # Double-check after acquiring lock (another thread may have loaded it)
            if resolved in _QWEN_CHAT_SINGLETONS:
                return _QWEN_CHAT_SINGLETONS[resolved]
            
            # Actually load the model
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model path not found: {p}")

            model = Qwen3VLForConditionalGeneration.from_pretrained(
            # model = Qwen3VLMoEForConditionalGeneration.from_pretrained(
                resolved,
                # dtype="auto",
                dtype=torch_dtype,
                device_map="auto",
                local_files_only=True,
                attn_implementation=attn_implementation,
            )
            processor = AutoProcessor.from_pretrained(
                resolved,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            
            # Set padding_side to 'left' for decoder-only models (required for correct generation)
            if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
                processor.tokenizer.padding_side = 'left'
            # Some processors might have the tokenizer under a different attribute name
            if hasattr(processor, 'text_tokenizer') and processor.text_tokenizer is not None:
                processor.text_tokenizer.padding_side = 'left'

            chat = QwenVLChat(model, processor, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
            _QWEN_CHAT_SINGLETONS[resolved] = chat
            return chat
    else:
        # Non-singleton path: load without caching
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved,
            dtype="auto",
            torch_dtype=torch_dtype,
            device_map="auto",
            # torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation=attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(
            resolved,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        
        # Set padding_side to 'left' for decoder-only models (required for correct generation)
        if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
            processor.tokenizer.padding_side = 'left'
        # Some processors might have the tokenizer under a different attribute name
        if hasattr(processor, 'text_tokenizer') and processor.text_tokenizer is not None:
            processor.text_tokenizer.padding_side = 'left'

        chat = QwenVLChat(model, processor, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        return chat


# ================================================================
# Tool-calling JSON parser
# ================================================================

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})", re.I)

def _parse_json_from_text(text: str) -> dict:
    if not text:
        return {}
    m = JSON_BLOCK_RE.search(text)
    s = (m.group(1) or m.group(2)) if m else text.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.strip("` \n\t"))
        except Exception:
            return {}


class ToolCallingShim:
    """Simulate .bind_tools() for chat models without built-in tool support."""

    def __init__(self, base_model):
        self.base = base_model
        self._tool_schemas = {}

    def bind_tools(self, tools):
        self._tool_schemas = {t.name: t for t in tools}
        return self

    async def ainvoke(self, msgs, **kwargs):
        resp = await self.base.ainvoke(msgs, **kwargs)
        text = resp.content if isinstance(resp.content, str) else ""
        data = _parse_json_from_text(text)

        action = data.get("action") or data.get("tool")
        args = data.get("args") if isinstance(data.get("args"), dict) else {}

        if isinstance(action, str) and action:
            if self._tool_schemas and action not in self._tool_schemas:
                return resp

            # IMPORTANT: Preserve the original content so that other fields like
            # "potential_anomalies" and "heuristic_prompt" are not lost
            return AIMessage(
                content=text,  # Keep original JSON content with all fields
                tool_calls=[{
                    "name": action,
                    "args": args,
                    "id": f"call_{hash((action, json.dumps(args, sort_keys=True)))}"
                }]
            )

        return resp


# ================================================================
# Qwen VL Embeddings Wrapper
# ================================================================

class TextModelEmbeddings(Embeddings):
    """Embeddings wrapper for text-only models (e.g., Qwen3, Qwen2).
    
    Uses the model's encoder to generate text embeddings.
    """
    
    def __init__(self, model_path: str, attn_implementation: Optional[str] = None):
        """Initialize with a text-only model path.
        
        Args:
            model_path: Path to the text-only model directory.
            attn_implementation: Attention implementation (e.g., "flash_attention_2", "sdpa", or None).
        """
        super().__init__()
        resolved = str(Path(model_path).resolve())
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        
        # Load model and tokenizer
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "local_files_only": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        self.model = AutoModel.from_pretrained(resolved, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True, trust_remote_code=True)
        
        self.device = next(self.model.parameters()).device
        self._fw_lock = threading.Lock()  # protects model forward calls to prevent concurrent GPU access from thread pool
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as a list of floats.
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the model's forward pass
        # Use lock to prevent concurrent forward calls from thread pool (run_in_executor)
        with self._fw_lock:
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Use last layer's hidden states
                        hidden_states = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
                        
                        # Weighted average pooling using attention mask
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = hidden_states.mean(dim=1)
                    else:
                        raise AttributeError("Model output does not have hidden_states")
                        
                except (AttributeError, RuntimeError) as e:
                    # Fallback: use input embeddings layer
                    if hasattr(self.model, 'get_input_embeddings'):
                        input_embeddings = self.model.get_input_embeddings()
                        token_embeddings = input_embeddings(inputs['input_ids'])
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = token_embeddings.mean(dim=1)
                    else:
                        raise ValueError(f"Cannot extract embeddings from text model: {e}")
        
        # Convert to list and L2 normalize
        embedding = embeddings[0].cpu().numpy().tolist()
        
        # L2 normalization
        import numpy as np
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using batch processing.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        # Tokenize all texts in batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the model's forward pass (batch processing)
        # Use lock to prevent concurrent forward calls from thread pool (run_in_executor)
        with self._fw_lock:
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Use last layer's hidden states
                        hidden_states = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
                        
                        # Weighted average pooling using attention mask
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = hidden_states.mean(dim=1)
                    else:
                        raise AttributeError("Model output does not have hidden_states")
                        
                except (AttributeError, RuntimeError) as e:
                    # Fallback: use input embeddings layer
                    if hasattr(self.model, 'get_input_embeddings'):
                        input_embeddings = self.model.get_input_embeddings()
                        token_embeddings = input_embeddings(inputs['input_ids'])
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = token_embeddings.mean(dim=1)
                    else:
                        raise ValueError(f"Cannot extract embeddings from text model: {e}")
        
        # Convert to numpy and L2 normalize
        import numpy as np
        embeddings_np = embeddings.cpu().numpy()
        
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        embeddings_np = embeddings_np / norms
        
        # Convert to list of lists
        return embeddings_np.tolist()
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


class QwenVLEmbeddings(Embeddings):
    """Embeddings wrapper for Qwen VL model.
    
    Uses the Qwen VL model's encoder to generate text embeddings.
    """
    
    def __init__(self, qwen_chat_model: BaseChatModel):
        """Initialize with a QwenVLChat model instance.
        
        Args:
            qwen_chat_model: The QwenVLChat model instance (from load_vl_model).
        """
        super().__init__()
        self.qwen_chat_model = qwen_chat_model
        
        # Access the underlying model and processor
        # QwenVLChat has _model and _processor attributes
        if hasattr(qwen_chat_model, '_model'):
            self.model = qwen_chat_model._model
        elif hasattr(qwen_chat_model, 'base') and hasattr(qwen_chat_model.base, '_model'):
            # Handle ToolCallingShim wrapper
            self.model = qwen_chat_model.base._model
        else:
            raise ValueError("Cannot access underlying Qwen VL model from chat model instance")
        
        if hasattr(qwen_chat_model, '_processor'):
            self.processor = qwen_chat_model._processor
        elif hasattr(qwen_chat_model, 'base') and hasattr(qwen_chat_model.base, '_processor'):
            self.processor = qwen_chat_model.base._processor
        else:
            raise ValueError("Cannot access processor from chat model instance")
        
        self.device = next(self.model.parameters()).device
        self._fw_lock = threading.Lock()  # protects model forward calls to prevent concurrent GPU access from thread pool
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using batch processing.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        # Get tokenizer from processor (for text-only tokenization)
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            tokenizer = self.processor.tokenizer
        elif hasattr(self.processor, 'text_tokenizer') and self.processor.text_tokenizer is not None:
            tokenizer = self.processor.text_tokenizer
        else:
            raise ValueError("Cannot find tokenizer in processor. Processor must have 'tokenizer' or 'text_tokenizer' attribute.")
        
        # Tokenize all texts in batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the model's forward pass (batch processing)
        # Use lock to prevent concurrent forward calls from thread pool (run_in_executor)
        with self._fw_lock:
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Use last layer's hidden states (most semantically rich)
                        hidden_states = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
                        
                        # Weighted average pooling using attention mask
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = hidden_states.mean(dim=1)
                    else:
                        raise AttributeError("Model output does not have hidden_states")
                        
                except (AttributeError, RuntimeError) as e:
                    # Fallback: use input embeddings layer
                    if hasattr(self.model, 'get_input_embeddings'):
                        input_embeddings = self.model.get_input_embeddings()
                        token_embeddings = input_embeddings(inputs['input_ids'])
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = token_embeddings.mean(dim=1)
                    else:
                        raise ValueError(f"Cannot extract embeddings from Qwen VL model: {e}")
        
        # Convert to numpy and L2 normalize
        import numpy as np
        embeddings_np = embeddings.cpu().numpy()
        
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        embeddings_np = embeddings_np / norms
        
        # Convert to list of lists
        return embeddings_np.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as a list of floats.
        """
        # Get tokenizer from processor (for text-only tokenization)
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            tokenizer = self.processor.tokenizer
        elif hasattr(self.processor, 'text_tokenizer') and self.processor.text_tokenizer is not None:
            tokenizer = self.processor.text_tokenizer
        else:
            raise ValueError("Cannot find tokenizer in processor. Processor must have 'tokenizer' or 'text_tokenizer' attribute.")
        
        # Tokenize the text using tokenizer (not processor directly)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the model's forward pass (preferred method)
        # For Qwen VL, we use the model's hidden states which contain richer semantic information
        # Use lock to prevent concurrent forward calls from thread pool (run_in_executor)
        with self._fw_lock:
            with torch.no_grad():
                # Primary method: use model forward pass to get hidden states
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Use last layer's hidden states (most semantically rich)
                        hidden_states = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
                        
                        # Weighted average pooling using attention mask
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            # Expand attention mask to match hidden_states dimensions
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            # Sum embeddings, masking out padding tokens
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                            # Sum of mask values (excluding padding)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            # Average
                            embeddings = sum_embeddings / sum_mask
                        else:
                            # Simple average if no attention mask
                            embeddings = hidden_states.mean(dim=1)
                    else:
                        # Fallback: use input embeddings if hidden_states not available
                        raise AttributeError("Model output does not have hidden_states")
                        
                except (AttributeError, RuntimeError) as e:
                    # Fallback: use input embeddings layer
                    if hasattr(self.model, 'get_input_embeddings'):
                        input_embeddings = self.model.get_input_embeddings()
                        # Get token embeddings
                        token_embeddings = input_embeddings(inputs['input_ids'])
                        # Average pooling over sequence length
                        attention_mask = inputs.get('attention_mask', None)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                        else:
                            embeddings = token_embeddings.mean(dim=1)
                    else:
                        raise ValueError(f"Cannot extract embeddings from Qwen VL model: {e}")
        
        # Convert to list and L2 normalize
        embedding = embeddings[0].cpu().numpy().tolist()
        
        # L2 normalization (common practice for embeddings)
        import numpy as np
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return embedding
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents."""
        # Run in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


# ================================================================
# Reasoning timeline extraction
# ================================================================

def extract_reasoning_timeline_from_state(state) -> List[Dict[str, Any]]:
    """Extract a chronological timeline of tool calls and judgments.
    
    This function reconstructs the reasoning process by analyzing messages
    in order to show: tool call -> judgment -> tool call -> judgment -> ...
    
    Args:
        state: State object or dict containing messages, executed_tool_calls, and reasoner_judgments.
        
    Returns:
        List of timeline entries, each containing:
        - step_index: sequential step number
        - step_type: "tool_call" or "judgment"
        - tool_call: dict with tool_name, args, node (if step_type is "tool_call")
        - judgment: dict with result, reason, call_index (if step_type is "judgment")
        - timestamp_order: order in which this step occurred
    """
    timeline = []
    
    # Get messages
    if isinstance(state, dict):
        messages = state.get("messages", [])
        executed_tool_calls = state.get("executed_tool_calls", [])
        judgments = state.get("reasoner_judgments", [])
    else:
        messages = getattr(state, "messages", [])
        executed_tool_calls = getattr(state, "executed_tool_calls", [])
        judgments = getattr(state, "reasoner_judgments", [])
    
    if not isinstance(executed_tool_calls, list):
        executed_tool_calls = []
    if not isinstance(judgments, list):
        judgments = []
    
    # Create a mapping of tool_call_id to tool call info
    tool_call_map = {}
    for tc in executed_tool_calls:
        tool_call_id = tc.get("tool_call_id", "")
        if tool_call_id:
            tool_call_map[tool_call_id] = tc
    
    # Create a mapping of call_index to judgment
    judgment_map = {}
    for j in judgments:
        call_index = j.get("call_index")
        if call_index is not None:
            judgment_map[call_index] = j
    
    # Use executed_tool_calls directly - they are already in execution order
    # Group tool calls by node to understand the flow
    planner_tool_calls = []
    reflector_tool_calls = []
    
    for tc in executed_tool_calls:
        node = tc.get("node", "unknown")
        if node == "planner":
            planner_tool_calls.append(tc)
        elif node == "reflector":
            reflector_tool_calls.append(tc)
        else:
            # If node is unknown, try to infer from position
            # First tool calls are usually from planner
            if len(planner_tool_calls) == 0:
                planner_tool_calls.append(tc)
            else:
                reflector_tool_calls.append(tc)
    
    # Sort judgments by call_index
    sorted_judgments = sorted(judgments, key=lambda x: x.get("call_index", 0))
    
    step_index = 0
    
    # Build timeline following the execution flow:
    # 1. Planner tool calls -> judgment(call_index=1)
    # 2. Reflector tool calls (if any) -> judgment(call_index=2)
    
    # Add planner tool calls
    for tc in planner_tool_calls:
        step_index += 1
        args_for_timeline = tc.get("args", {}).copy()
        if "image_base64" in args_for_timeline:
            args_for_timeline["image_base64"] = "[REDACTED: too long]"
        
        timeline.append({
            "step_index": step_index,
            "step_type": "tool_call",
            "tool_call": {
                "tool_name": tc.get("tool_name", "unknown"),
                "tool_call_id": tc.get("tool_call_id", "unknown"),
                "args": args_for_timeline,
                "node": tc.get("node", "unknown")
            }
        })
    
    # Add first judgment (call_index=1) - based on planner tool calls
    if len(sorted_judgments) > 0 and sorted_judgments[0].get("call_index") == 1:
        step_index += 1
        timeline.append({
            "step_index": step_index,
            "step_type": "judgment",
            "judgment": {
                "call_index": 1,
                "result": sorted_judgments[0].get("result", "unknown"),
                "reason": sorted_judgments[0].get("reason", ""),
                "forced": sorted_judgments[0].get("forced", False),
                "score": sorted_judgments[0].get("score")
            },
            "tools_used": [tc.get("tool_name") for tc in planner_tool_calls]
        })
    
    # Add reflector tool calls (if any)
    if len(reflector_tool_calls) > 0:
        for tc in reflector_tool_calls:
            step_index += 1
            args_for_timeline = tc.get("args", {}).copy()
            if "image_base64" in args_for_timeline:
                args_for_timeline["image_base64"] = "[REDACTED: too long]"
            
            timeline.append({
                "step_index": step_index,
                "step_type": "tool_call",
                "tool_call": {
                    "tool_name": tc.get("tool_name", "unknown"),
                    "tool_call_id": tc.get("tool_call_id", "unknown"),
                    "args": args_for_timeline,
                    "node": tc.get("node", "unknown")
                }
            })
    elif len(sorted_judgments) > 1:
        # If there's a second judgment but no reflector tool calls, add placeholder
        step_index += 1
        timeline.append({
            "step_index": step_index,
            "step_type": "tool_call",
            "tool_call": {
                "tool_name": "(no tools called)",
                "tool_call_id": "none",
                "args": {},
                "node": "reflector"
            }
        })
    
    # Add second judgment (call_index=2) - based on reflector tool calls (if any)
    if len(sorted_judgments) > 1 and sorted_judgments[1].get("call_index") == 2:
        step_index += 1
        tools_used = [tc.get("tool_name") for tc in reflector_tool_calls] if reflector_tool_calls else ["(no tools called)"]
        timeline.append({
            "step_index": step_index,
            "step_type": "judgment",
            "judgment": {
                "call_index": 2,
                "result": sorted_judgments[1].get("result", "unknown"),
                "reason": sorted_judgments[1].get("reason", ""),
                "forced": sorted_judgments[1].get("forced", False),
                "score": sorted_judgments[1].get("score")
            },
            "tools_used": tools_used
        })
    
    # Add any remaining judgments (shouldn't happen in normal flow)
    for j in sorted_judgments[2:]:
        step_index += 1
        timeline.append({
            "step_index": step_index,
            "step_type": "judgment",
            "judgment": {
                "call_index": j.get("call_index"),
                "result": j.get("result", "unknown"),
                "reason": j.get("reason", ""),
                "forced": j.get("forced", False),
                "score": j.get("score")
            }
        })
    
    # Sort timeline by step_index to ensure chronological order
    timeline.sort(key=lambda x: x.get("step_index", 0))
    
    return timeline


async def generate_and_save_atomic_candidates(
    object_names: List[str],
    context: "Context",
    output_file: str = "utils/atomic_candidates.json",
    file_name: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate atomic candidates for a list of objects and save them to a file.
    
    This function uses the same process as counterfactual_atomic_candidate_tool (Step 1)
    to generate anomaly_candidates and normal_candidates for each object, then saves
    all results to a JSON file.
    
    Args:
        object_names: List of object/class names (e.g., ["bottle", "carpet", "hazelnut"]).
        context: Context instance containing models and prompts.
        output_file: Path to output file (default: "utils/atomic_candidates.json").
        file_name: Optional file name to check for existing candidates. If provided and file exists,
            the function will load and return the existing candidates without generation.
    
    Returns:
        Dictionary mapping object names to their candidate sets:
        {
            "object_name": {
                "anomaly_candidates": [...],
                "normal_candidates": [...]
            },
            ...
        }
    """
    
    # Ensure output directory exists
    # Convert to absolute path to avoid issues with relative paths and current working directory
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file_name is provided and file exists in the output directory
    if file_name is not None:
        # Check file in the same directory as output_file
        # file_name should be just the filename, not a path
        file_path = output_path.parent / file_name
        if file_path.exists() and file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_candidates = json.load(f)
                print(f"Using existing candidate set from: {file_name}")
                return all_candidates
            except Exception as e:
                print(f"Warning: Failed to load existing candidate set from {file_name}: {e}")
                print("Proceeding with generation...")
    
    # Get LLM model for candidate generation (text-only task)
    model = context.atomic_candidate_llm_model
    
    # Initialize caches if not present (same as counterfactual_atomic_candidate_tool)
    if not hasattr(context, '_atomic_candidate_cache'):
        context._atomic_candidate_cache = {}  # Cache for candidate texts
    
    # Dictionary to store all candidates
    all_candidates: Dict[str, Dict[str, List[str]]] = {}
    
    # Helper function
    def _get_text(content: Any) -> str:
        """Normalize AIMessage.content into string.

        Supports both legacy string content and Responses API content blocks like:
        [{"type": "text", "text": "..."}]
        """
        if isinstance(content, str):
            return content

        def _collect_text_fields(value: Any) -> list[str]:
            fragments: list[str] = []
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
    
    # Generate prompt hash for version tracking (same as counterfactual_atomic_candidate_tool)
    prompt_text = PROMPT4CANDIDATE_GENERATION
    prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
    
    # Get model fingerprint (same as counterfactual_atomic_candidate_tool)
    try:
        model_fingerprint = getattr(model, 'model_name', None) or type(model).__name__
    except:
        model_fingerprint = "unknown"
    model_fingerprint_hash = hashlib.md5(str(model_fingerprint).encode()).hexdigest()[:8]
    
    # Cache size limit to prevent memory bloat (FIFO: remove oldest when limit reached)
    MAX_CACHE_SIZE = 512
    
    # Generate candidates for each object
    pbar = tqdm(object_names, desc="Generating atomic candidates", unit="object")
    for class_name in pbar:
        if not class_name or not isinstance(class_name, str):
            tqdm.write(f"Warning: Skipping invalid object name: {class_name}")
            continue
        
        # Standardize class_name (strip and lowercase) to reduce cache fragmentation
        # Prevents duplicate cache entries when class_name varies in case (e.g., "Bottle" vs "bottle")
        standardized_class_name = class_name.strip().lower()
        
        # Update progress bar description with current object name
        pbar.set_description(f"Generating atomic candidates: {class_name}")
        
        try:
            # Cache key for candidates (same as counterfactual_atomic_candidate_tool)
            candidate_cache_key = f"{standardized_class_name}_{prompt_hash}_{model_fingerprint_hash}"
            
            # Initialize candidate_gen_temperature_applied (needed for report even on cache hit)
            candidate_gen_temperature_applied = None  # None=cache hit; True=bound temp applied; False=not applied
            
            # Check cache for candidates (same as counterfactual_atomic_candidate_tool)
            if candidate_cache_key in context._atomic_candidate_cache:
                anomaly_candidates, normal_candidates, candidate_version = context._atomic_candidate_cache[candidate_cache_key]
                tqdm.write(f"  Using cached candidates (version: {candidate_version})")
            else:
                # Generate candidates using LLM (temperature≈0 for reproducibility)
                # For better reproducibility, omit system_time from system prompt (only for candidate generation)
                # System time can introduce contextual noise even with temperature=0
                prompt = f"{PROMPT4CANDIDATE_GENERATION}\n\nClass name: {class_name}"
                
                # Use minimal message list without system message (or with system message without time)
                # This improves reproducibility by avoiding dynamic system_time in context
                msg_list: List[BaseMessage | Dict[str, str]] = [
                    {"role": "user", "content": prompt},
                ]
                
                # Try to set temperature≈0 for reproducibility
                # Some models support temperature via bind() or invocation_params
                candidate_gen_temperature_applied = False
                try:
                    # Try to bind temperature parameter (method varies by model type)
                    if hasattr(model, 'bind'):
                        # Check if model supports temperature parameter
                        bound_model = model.bind(temperature=0.0)
                        resp = cast(AIMessage, await bound_model.ainvoke(msg_list))
                        candidate_gen_temperature_applied = True
                    else:
                        # Fallback: use model normally
                        resp = cast(AIMessage, await model.ainvoke(msg_list))
                except (TypeError, ValueError, AttributeError):
                    # If binding fails, use model normally
                    # Note: candidate_version will track the actual candidates generated
                    resp = cast(AIMessage, await model.ainvoke(msg_list))
                
                parsed = _parse_json_from_text(_get_text(resp.content))
                
                # Extract candidates
                anomaly_candidates = parsed.get("anomaly_candidates", [])
                normal_candidates = parsed.get("normal_candidates", [])
                
                # Validate and normalize
                if not isinstance(anomaly_candidates, list):
                    anomaly_candidates = []
                if not isinstance(normal_candidates, list):
                    normal_candidates = []
                
                # Filter out empty strings first (cleaner strategy: filter before versioning)
                valid_anomaly_candidates = [c for c in anomaly_candidates if c and isinstance(c, str) and c.strip()]
                valid_normal_candidates = [c for c in normal_candidates if c and isinstance(c, str) and c.strip()]
                
                # Use filtered candidates for versioning (avoids version drift from empty strings)
                candidate_content = json.dumps([valid_anomaly_candidates, valid_normal_candidates], sort_keys=True)
                candidate_version = hashlib.md5(candidate_content.encode()).hexdigest()[:8]
                
                # Cache size limit to prevent memory bloat (FIFO: remove oldest when limit reached)
                # Eviction only when preparing to insert new entry (not on cache hit)
                if len(context._atomic_candidate_cache) >= MAX_CACHE_SIZE:
                    # Remove oldest entry (first key in dict, which is insertion order in Python 3.7+)
                    # This is FIFO (First In First Out), not LRU (Least Recently Used)
                    oldest_key = next(iter(context._atomic_candidate_cache))
                    del context._atomic_candidate_cache[oldest_key]
                
                # Store filtered candidates (no padding needed)
                # Note: If LLM generates unequal quantities, we work with what we have
                # (better than padding with empty strings which would affect versioning)
                context._atomic_candidate_cache[candidate_cache_key] = (valid_anomaly_candidates, valid_normal_candidates, candidate_version)
                
                # Update local variables to use filtered candidates (critical fix: ensures consistency)
                anomaly_candidates = valid_anomaly_candidates
                normal_candidates = valid_normal_candidates
                
                temp_status = "applied" if candidate_gen_temperature_applied else "not applied"
                tqdm.write(f"  Generated new candidates (version: {candidate_version}, temperature: {temp_status})")
            
            # Store candidates
            all_candidates[class_name] = {
                "anomaly_candidates": anomaly_candidates,
                "normal_candidates": normal_candidates,
            }
            
            tqdm.write(f"  Result: {len(anomaly_candidates)} anomaly candidates and {len(normal_candidates)} normal candidates")
            
        except Exception as e:
            tqdm.write(f"Error generating candidates for {class_name}: {e}")
            # Store empty candidates on error
            all_candidates[class_name] = {
                "anomaly_candidates": [],
                "normal_candidates": [],
            }
    
    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_candidates, f, indent=2, ensure_ascii=False)
        print(f"\nSaved atomic candidates to: {output_path}")
        print(f"Total objects processed: {len(all_candidates)}")
    except Exception as e:
        print(f"Error saving to file: {e}")
        raise
    
    return all_candidates
