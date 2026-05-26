"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, fields
from inspect import isawaitable
from typing import Annotated, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from . import prompts as prompts
from .api_model_utils import parse_api_model_id
from .utils import load_vl_model, ToolCallingShim, QwenVLEmbeddings, TextModelEmbeddings


DEFAULT_QWEN_VL_MODEL_PATH = os.environ.get("QWEN_VL_MODEL_PATH", "./models/qwen3_vl_4b")
DEFAULT_QWEN_TEXT_MODEL_PATH = os.environ.get("QWEN_TEXT_MODEL_PATH", "./models/qwen3_4b")


def _load_api_model(api_model: str, use_responses_api: bool = False) -> BaseChatModel:
    """Load API model using langchain's init_chat_model.
    
    Args:
        api_model: Model identifier in format 'provider/model-name'
        use_responses_api: Whether to force OpenAI models to use Responses API mode.
    
    Returns:
        BaseChatModel: The loaded API model instance
    """
    from langchain.chat_models import init_chat_model
    provider, model_name = parse_api_model_id(api_model)
    init_kwargs = {}
    if provider == "openai":
        init_kwargs["use_responses_api"] = use_responses_api
    if provider == "google_genai":
        try:
            import langchain_google_genai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Gemini Developer API models require langchain-google-genai. "
                "Install it with: pip install -U 'langchain-google-genai>=3.1.0'"
            ) from exc
    if provider is None:
        return init_chat_model(model_name, **init_kwargs)
    return init_chat_model(model_name, model_provider=provider, **init_kwargs)


async def _close_api_model_client(model: Optional[BaseChatModel]) -> None:
    """Best-effort cleanup for API model clients."""
    if model is None:
        return

    seen: set[int] = set()

    async def _maybe_close(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        aclose = getattr(obj, "aclose", None)
        if callable(aclose):
            result = aclose()
            if isawaitable(result):
                await result

        close = getattr(obj, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result

    async def _walk(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 3:
            return

        await _maybe_close(obj)

        for attr in (
            "aio",
            "async_client",
            "_async_client",
            "client",
            "_client",
            "_genai_client",
            "_api_client",
            "base",
            "bound",
        ):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                continue
            if child is not None and child is not obj:
                await _walk(child, depth + 1)

    try:
        await _walk(model)
    except Exception as exc:
        print(
            "[Context] Warning: failed to close API model client cleanly: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )



@dataclass(kw_only=True) # key-word only -- need to pass in keyword name when instantiation
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    prompt4planner: str = field(
        default=prompts.PROMPT4PLANNER,
        metadata={
            "description": "The prompt template for the planner node (legacy, use prompt4planner_api or prompt4planner_local instead). "
            "Used to generate initial anomaly detection analysis."
        },
    )

    prompt4planner_api: str = field(
        default=prompts.PROMPT4PLANNER_API,
        metadata={
            "description": "The prompt template for the planner node when using API models. "
            "Allows skipping tool calls if obvious anomalies are already identified."
        },
    )

    prompt4planner_local: str = field(
        default=prompts.PROMPT4PLANNER_LOCAL,
        metadata={
            "description": "The prompt template for the planner node when using local VLM models. "
            "Requires JSON output with action field."
        },
    )

    prompt4reasoner: str = field(
        default=prompts.PROMPT4REASONER,
        metadata={
            "description": "The prompt template for the reasoner node. "
            "Used to make anomaly detection judgments."
        },
    )

    prompt4reasoner_final: str = field(
        default=prompts.PROMPT4REASONER_FINAL,
        metadata={
            "description": "The prompt template for the reasoner node when tool_call_threshold is exceeded. "
            "Used to make final definitive judgments (anomalous or normal only, no uncertain allowed)."
        },
    )

    prompt4reflector: str = field(
        default=prompts.PROMPT4REFLECTOR,
        metadata={
            "description": "The prompt template for the reflector node (legacy, use prompt4reflector_api or prompt4reflector_local instead). "
            "Used when the reasoner result is 'normal' or 'uncertain' and re-analysis/verification is required."
        },
    )

    prompt4reflector_api: str = field(
        default=prompts.PROMPT4REFLECTOR_API,
        metadata={
            "description": "The prompt template for the reflector node when using API models. "
            "Allows direct tool calling without JSON output."
        },
    )

    prompt4reflector_local: str = field(
        default=prompts.PROMPT4REFLECTOR_LOCAL,
        metadata={
            "description": "The prompt template for the reflector node when using local VLM models. "
            "Requires JSON output with action field."
        },
    )

    summary_prompt: str = field(
        default=prompts.SUMMARY_PROMPT,
        metadata={
            "description": "The prompt template for the memory node. "
            "Used to summarize the detection process and results."
        },
    )

    prompt4description: str = field(
        default=prompts.PROMPT4DESCRIPTION,
        metadata={
            "description": "The prompt template for description. "
            "Used to describe the visual condition of the object in the image."
        },
    )

    force_decision_edges: int = field(
        default=15,
        metadata={
            "description": "The number of edges to force the decision to be made."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={
            "description": "The default language model for general LLM interactions."
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Qwen VL model configuration
    qwen_model_path: str = field(
        default=DEFAULT_QWEN_VL_MODEL_PATH,
        metadata={
            "description": "Path to the local Qwen VL model directory. "
            "This model will be used for both VLM (vision-language) and LLM (language-only) tasks."
        },
    )

    device: Optional[str] = field(
        default=None,
        metadata={
            "description": "Device to run the model on ('cuda', 'cpu', or None for auto-detect)."
        },
    )

    # Qwen VL batching parameters (for debugging/optimization)
    max_batch_size: int = field(
        default=16,
        metadata={
            "description": "Maximum batch size for Qwen VL micro-batching. "
            "Set to 1 for debugging single images. Default: 16 for performance (reduced from 32 to avoid INT_MAX tensor overflow)."
        },
    )

    max_wait_ms: int = field(
        default=50,
        metadata={
            "description": "Maximum wait time (milliseconds) before flushing incomplete batches. "
            "Set to 0 for immediate processing (useful for debugging). Default: 50ms."
        },
    )

    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "description": "Attention implementation for Qwen VL model. "
            "Options: 'flash_attention_2', 'sdpa', or None (default: 'flash_attention_2')."
        },
    )

    embedding_model_path: Optional[str] = field(
        default=None,
        metadata={
            "description": "Path to model for embeddings. "
            "If None, defaults to QWEN_TEXT_MODEL_PATH or ./models/qwen3_4b (text-only LLM). "
            "If provided, uses the specified path. "
            "If path matches qwen_model_path, uses VLM for embeddings; otherwise uses LLM."
        },
    )

    tau: float = field(
        default=0.3,
        metadata={
            "description": "Temperature parameter for sigmoid calculation in prototype-based scoring. "
            "Lower values (e.g., 0.1) make scores more saturated (confident but potentially unstable). "
            "Higher values (e.g., 0.3-0.5) provide more stable scores. "
            "Default: 0.3 for better stability. The margin between caption and prototype is typically in [-0.2, 0.2]."
        },
    )

    tau_cand: float = field(
        default=0.1,
        metadata={
            "description": "Temperature parameter for logsumexp calculation in atomic candidate matching. "
            "Controls the sensitivity of logsumexp to cosine similarity differences. "
            "Lower values (e.g., 0.05-0.1) emphasize stronger matches; higher values (e.g., 0.15-0.2) provide more balanced weighting. "
            "Default: 0.1 for cosine similarity range [-1, 1]."
        },
    )

    atomic_candidates_file: Optional[str] = field(
        default=None,
        metadata={
            "description": "Path to the atomic candidates JSON file. "
            "If provided, counterfactual_atomic_candidate_tool will load candidates from this file. "
            "If None, candidates will be generated on-the-fly (not recommended for evaluation)."
        },
    )

    super_resolution_version: str = field(
        default="nongan",
        metadata={
            "description": "Super-resolution model version. Options: 'gan' (RealESRGAN) or 'nongan' (RealESRNet). "
            "Default: 'nongan'. GAN version may provide better perceptual quality but slower; "
            "non-GAN version is faster and uses less memory."
        },
    )

    # VLM model type configuration (local or api)
    planner_model_type: str = field(
        default="local",
        metadata={
            "description": "Model type for planner node. Options: 'local' (use local Qwen VL) or 'api' (use API model). "
            "Default: 'local'."
        },
    )

    reflector_model_type: str = field(
        default="local",
        metadata={
            "description": "Model type for reflector node. Options: 'local' (use local Qwen VL) or 'api' (use API model). "
            "Default: 'local'."
        },
    )

    reasoner_model_type: str = field(
        default="local",
        metadata={
            "description": "Model type for reasoner node. Options: 'local' (use local Qwen VL) or 'api' (use API model). "
            "Default: 'local'."
        },
    )

    # API model configuration (only used when model_type is 'api')
    planner_api_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "API model identifier for planner (e.g., 'anthropic/claude-sonnet-4-5-20250929' or 'google_genai/gemini-3.1-pro-preview'). "
            "Only used when planner_model_type is 'api'. Default: None."
        },
    )

    reflector_api_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "API model identifier for reflector (e.g., 'anthropic/claude-sonnet-4-5-20250929' or 'google_genai/gemini-3.1-pro-preview'). "
            "Only used when reflector_model_type is 'api'. Default: None."
        },
    )

    reasoner_api_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "API model identifier for reasoner (e.g., 'anthropic/claude-sonnet-4-5-20250929' or 'google_genai/gemini-3.1-pro-preview'). "
            "Only used when reasoner_model_type is 'api'. Default: None."
        },
    )

    # Image description model configuration (for image description tasks in tools)
    image_description_model_type: str = field(
        default="local",
        metadata={
            "description": "Model type for image description tasks (e.g., _keyword_heuristic_decision, counterfactual_atomic_candidate_tool caption generation). "
            "Options: 'local' (use local Qwen VL) or 'api' (use API model). Default: 'local'."
        },
    )

    image_description_api_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "API model identifier for image description tasks (e.g., 'anthropic/claude-sonnet-4-5-20250929' or 'google_genai/gemini-2.5-flash'). "
            "Only used when image_description_model_type is 'api'. Default: None."
        },
    )

    # Atomic candidate generation LLM model configuration (for text-only generation)
    atomic_candidate_llm_model_type: str = field(
        default="local",
        metadata={
            "description": "Model type for atomic candidate generation (text-only LLM). "
            "Options: 'local' (use local model path, if configured) or 'api' (use API model). "
            "Default: 'local'. Note: For local, uses the same Qwen VL model (text-only mode)."
        },
    )

    atomic_candidate_llm_api_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "API model identifier for atomic candidate generation (e.g., 'anthropic/claude-sonnet-4-5-20250929' or 'google_genai/gemini-2.5-flash'). "
            "Only used when atomic_candidate_llm_model_type is 'api'. Default: None."
        },
    )

    # API call configuration (for retry, timeout, etc.)
    api_call_timeout: float = field(
        default=300.0,
        metadata={
            "description": "Timeout for API calls in seconds. Default: 300.0 (5 minutes)."
        },
    )

    api_call_max_retries: int = field(
        default=3,
        metadata={
            "description": "Maximum number of retries for API calls on failure. Default: 3."
        },
    )

    api_call_retry_delay: float = field(
        default=1.0,
        metadata={
            "description": "Initial delay between retries in seconds. Uses exponential backoff. Default: 1.0."
        },
    )

    api_call_max_retry_delay: float = field(
        default=60.0,
        metadata={
            "description": "Maximum delay between retries in seconds. Default: 60.0."
        },
    )

    debug_api_errors: bool = field(
        default=False,
        metadata={
            "description": "If True, print full traceback for API errors (may leak sensitive info). "
            "If False (default), only print truncated error info. Default: False."
        },
    )

    api_max_concurrency: Optional[int] = field(
        default=None,
        metadata={
            "description": "Maximum concurrent API calls globally. If None, no limit. "
            "Useful for rate limiting across all API calls. Default: None."
        },
    )

    # API model generation parameters (applied to all API calls)
    api_temperature: Optional[float] = field(
        default=None,
        metadata={
            "description": "Temperature for API model generation. Lower values (e.g., 0.0) for deterministic output, "
            "higher values (e.g., 0.7) for more creative output. If None, uses model default."
        },
    )

    api_max_tokens: Optional[int] = field(
        default=None,
        metadata={
            "description": "Maximum tokens to generate in API calls. If None, uses model default."
        },
    )

    api_top_p: Optional[float] = field(
        default=None,
        metadata={
            "description": "Top-p (nucleus) sampling parameter for API calls. If None, uses model default."
        },
    )

    api_frequency_penalty: Optional[float] = field(
        default=None,
        metadata={
            "description": "Frequency penalty for API calls (reduces repetition). If None, uses model default."
        },
    )

    api_presence_penalty: Optional[float] = field(
        default=None,
        metadata={
            "description": "Presence penalty for API calls (encourages new topics). If None, uses model default."
        },
    )

    use_responses_api: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable OpenAI Responses API mode when loading OpenAI API models. "
            "Only applied when provider is 'openai'. Default: False."
        },
    )

    reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "description": "Reasoning effort for OpenAI reasoning models when using Responses API. "
            "Typical values: none, low, medium, high (model-dependent). "
            "Only applied when use_responses_api is True and provider is 'openai'."
        },
    )

    tool_call_threshold: int = field(
        default=5,
        metadata={
            "description": "Maximum number of tool calls before forcing a decision. When num_tool_calls exceeds this threshold, the reasoner will skip API calls and use heuristic-based decision. Default: 5."
        },
    )

    raw_to_reasoner: bool = field(
        default=True,
        metadata={
            "description": "Whether to include the raw (original) image in the reasoner node input. If True, reasoner sees both raw and augmented images. If False, reasoner only sees augmented images. Default: True."
        },
    )

    raw_to_reflector: bool = field(
        default=True,
        metadata={
            "description": "Whether to include the raw (original) image in the reflector node input. If True, reflector sees both raw and augmented images. If False, reflector only sees augmented images. Note: If False and no augmented image is available (e.g., planner didn't call tools), falls back to showing raw image. Default: True."
        },
    )

    normal_tolerance: int = field(
        default=2,
        metadata={
            "description": "Number of consecutive 'normal' judgments required before accepting 'normal' as final result. The reasoner must return 'normal' this many times (after reflection cycles) before the result is accepted. Default: 2."
        },
    )

    max_reasoner_calls: int = field(
        default=6,
        metadata={
            "description": "Maximum number of reasoner calls before forcing a final decision using prompt4reasoner_final. This prevents infinite loops when the model doesn't call tools. Default: 6."
        },
    )

    counterfactual_top_k: int = field(
        default=3,
        metadata={
            "description": "Top-k for counterfactual atomic candidate matching. Default: 3."
        },
    )

    # Private field for prototype cache (keyed by class_name)
    _prototype_cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # Private field for shared model instance (lazy loaded)
    _qwen_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    # Private fields for API model instances (lazy loaded)
    _planner_api_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    _reflector_api_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    _reasoner_api_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    _image_description_api_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    _atomic_candidate_llm_api_model_instance: Optional[BaseChatModel] = field(
        default=None,
        init=False,
        repr=False,
    )

    # Private field for embedding wrapper (lazy loaded, uses Qwen VL model)
    _embedding_model_instance: Optional[Embeddings] = field(
        default=None,
        init=False,
        repr=False,
    )
    
    # Lock for thread-safe model loading
    _qwen_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    
    # Lock for thread-safe embedding model loading
    _embedding_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    # Locks for thread-safe API model loading
    _planner_api_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    _reflector_api_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    _reasoner_api_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    _image_description_api_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    _atomic_candidate_llm_api_model_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Auto-fill attributes with environment variables if not explicitly provided.
        Automatically run after initialization.
        """
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_value = os.environ.get(f.name.upper(), f.default)
                # Handle Optional[str] fields
                if f.type == Optional[str] and env_value:
                    setattr(self, f.name, env_value)
                elif f.type != Optional[str]:
                    setattr(self, f.name, env_value)

    def _load_qwen_model(self) -> BaseChatModel:
        """Load the Qwen VL model instance (shared by both VLM and LLM tasks, thread-safe).

        Returns:
            BaseChatModel: The loaded Qwen VL model instance.
        """
        # Double-check locking pattern for thread safety
        if self._qwen_model_instance is None:
            with self._qwen_model_lock:
                # Check again after acquiring lock (another thread may have loaded it)
                if self._qwen_model_instance is None:
                    try:
                        print(f"[Context] Loading Qwen VL model: {self.qwen_model_path}...")
                        self._qwen_model_instance = load_vl_model(
                            model_path=self.qwen_model_path,
                            trust_remote_code=True,
                            use_singleton=True,
                            attn_implementation=self.attn_implementation,
                            max_batch_size=self.max_batch_size,
                            max_wait_ms=self.max_wait_ms,
                        )
                        print(f"[Context] Qwen VL model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading Qwen VL model: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
        return ToolCallingShim(self._qwen_model_instance)

    def _load_planner_api_model(self) -> BaseChatModel:
        """Load the API model for planner (thread-safe).

        Returns:
            BaseChatModel: The loaded API model instance.
        """
        if self._planner_api_model_instance is None:
            with self._planner_api_model_lock:
                if self._planner_api_model_instance is None:
                    if self.planner_api_model is None:
                        raise ValueError(
                            "planner_api_model must be specified when planner_model_type is 'api'"
                        )
                    try:
                        print(f"[Context] Loading planner API model: {self.planner_api_model}...")
                        self._planner_api_model_instance = _load_api_model(
                            self.planner_api_model,
                            use_responses_api=self.use_responses_api,
                        )
                        print(f"[Context] Planner API model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading planner API model: {e}")
                        if self.debug_api_errors:
                            import traceback
                            traceback.print_exc()
                        else:
                            error_class = type(e).__name__
                            error_msg = str(e)[:200]
                            print(f"[Context] Error details: {error_class}: {error_msg}")
                        raise
        return self._planner_api_model_instance

    def _load_reflector_api_model(self) -> BaseChatModel:
        """Load the API model for reflector (thread-safe).

        Returns:
            BaseChatModel: The loaded API model instance.
        """
        if self._reflector_api_model_instance is None:
            with self._reflector_api_model_lock:
                if self._reflector_api_model_instance is None:
                    if self.reflector_api_model is None:
                        raise ValueError(
                            "reflector_api_model must be specified when reflector_model_type is 'api'"
                        )
                    try:
                        print(f"[Context] Loading reflector API model: {self.reflector_api_model}...")
                        self._reflector_api_model_instance = _load_api_model(
                            self.reflector_api_model,
                            use_responses_api=self.use_responses_api,
                        )
                        print(f"[Context] Reflector API model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading reflector API model: {e}")
                        if self.debug_api_errors:
                            import traceback
                            traceback.print_exc()
                        else:
                            error_class = type(e).__name__
                            error_msg = str(e)[:200]
                            print(f"[Context] Error details: {error_class}: {error_msg}")
                        raise
        return self._reflector_api_model_instance

    def _load_reasoner_api_model(self) -> BaseChatModel:
        """Load the API model for reasoner (thread-safe).

        Returns:
            BaseChatModel: The loaded API model instance.
        """
        if self._reasoner_api_model_instance is None:
            with self._reasoner_api_model_lock:
                if self._reasoner_api_model_instance is None:
                    if self.reasoner_api_model is None:
                        raise ValueError(
                            "reasoner_api_model must be specified when reasoner_model_type is 'api'"
                        )
                    try:
                        print(f"[Context] Loading reasoner API model: {self.reasoner_api_model}...")
                        self._reasoner_api_model_instance = _load_api_model(
                            self.reasoner_api_model,
                            use_responses_api=self.use_responses_api,
                        )
                        print(f"[Context] Reasoner API model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading reasoner API model: {e}")
                        if self.debug_api_errors:
                            import traceback
                            traceback.print_exc()
                        else:
                            error_class = type(e).__name__
                            error_msg = str(e)[:200]
                            print(f"[Context] Error details: {error_class}: {error_msg}")
                        raise
        return self._reasoner_api_model_instance

    def _load_image_description_api_model(self) -> BaseChatModel:
        """Load the API model for image description tasks (thread-safe).

        Returns:
            BaseChatModel: The loaded API model instance.
        """
        if self._image_description_api_model_instance is None:
            with self._image_description_api_model_lock:
                if self._image_description_api_model_instance is None:
                    if self.image_description_api_model is None:
                        raise ValueError(
                            "image_description_api_model must be specified when image_description_model_type is 'api'"
                        )
                    try:
                        print(f"[Context] Loading image description API model: {self.image_description_api_model}...")
                        self._image_description_api_model_instance = _load_api_model(
                            self.image_description_api_model,
                            use_responses_api=self.use_responses_api,
                        )
                        print(f"[Context] Image description API model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading image description API model: {e}")
                        if self.debug_api_errors:
                            import traceback
                            traceback.print_exc()
                        else:
                            error_class = type(e).__name__
                            error_msg = str(e)[:200]
                            print(f"[Context] Error details: {error_class}: {error_msg}")
                        raise
        return self._image_description_api_model_instance

    def _load_atomic_candidate_llm_api_model(self) -> BaseChatModel:
        """Load the API model for atomic candidate generation (thread-safe).

        Returns:
            BaseChatModel: The loaded API model instance.
        """
        if self._atomic_candidate_llm_api_model_instance is None:
            with self._atomic_candidate_llm_api_model_lock:
                if self._atomic_candidate_llm_api_model_instance is None:
                    if self.atomic_candidate_llm_api_model is None:
                        raise ValueError(
                            "atomic_candidate_llm_api_model must be specified when atomic_candidate_llm_model_type is 'api'"
                        )
                    try:
                        print(f"[Context] Loading atomic candidate LLM API model: {self.atomic_candidate_llm_api_model}...")
                        self._atomic_candidate_llm_api_model_instance = _load_api_model(
                            self.atomic_candidate_llm_api_model,
                            use_responses_api=self.use_responses_api,
                        )
                        print(f"[Context] Atomic candidate LLM API model loaded successfully")
                    except Exception as e:
                        print(f"[Context] Error loading atomic candidate LLM API model: {e}")
                        if self.debug_api_errors:
                            import traceback
                            traceback.print_exc()
                        else:
                            error_class = type(e).__name__
                            error_msg = str(e)[:200]
                            print(f"[Context] Error details: {error_class}: {error_msg}")
                        raise
        return self._atomic_candidate_llm_api_model_instance

    @property
    def atomic_candidate_llm_model(self) -> BaseChatModel:
        """Lazy-load the LLM model for atomic candidate generation (text-only).

        This model is used for:
        - generate_and_save_atomic_candidates: generating text candidates (anomaly_candidates and normal_candidates)

        Note: This is a text-only task, so when using local model, uses Qwen VL model in text-only mode.
        When using API model, uses the specified API model.

        Returns:
            BaseChatModel: The loaded model instance (local Qwen VL or API model).
        """
        if self.atomic_candidate_llm_model_type == "api":
            return self._load_atomic_candidate_llm_api_model()
        else:
            # For local, use the same Qwen VL model (it supports text-only inputs)
            return self._load_qwen_model()

    @property
    def image_description_model(self) -> BaseChatModel:
        """Lazy-load the image description model (for image description tasks in tools).

        This model is used for:
        - _keyword_heuristic_decision: generating image descriptions for heuristic decision
        - counterfactual_atomic_candidate_tool: generating image captions (fallback when captions not in state)

        Returns:
            BaseChatModel: The loaded model instance (local Qwen VL or API model).
        """
        if self.image_description_model_type == "api":
            return self._load_image_description_api_model()
        else:
            return self._load_qwen_model()

    @property
    def planner_model(self) -> BaseChatModel:
        """Lazy-load the planner model (VLM with tool support or API model).

        This model is used for:
        - planner node: initial anomaly detection with vision-language capabilities

        Returns:
            BaseChatModel: The loaded model instance (local Qwen VL or API model).
        """
        if self.planner_model_type == "api":
            return self._load_planner_api_model()
        else:
            return self._load_qwen_model()

    @property
    def reflector_model(self) -> BaseChatModel:
        """Lazy-load the reflector model (VLM with tool support or API model).

        This model is used for:
        - reflector node: re-analysis when reasoner is uncertain

        Returns:
            BaseChatModel: The loaded model instance (local Qwen VL or API model).
        """
        if self.reflector_model_type == "api":
            return self._load_reflector_api_model()
        else:
            return self._load_qwen_model()

    @property
    def planner_reflector_model(self) -> BaseChatModel:
        """Deprecated: Use planner_model or reflector_model instead.
        
        This property is kept for backward compatibility and returns planner_model.
        """
        import warnings
        warnings.warn(
            "planner_reflector_model is deprecated. Use planner_model or reflector_model instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.planner_model

    @property
    def reasoner_model(self) -> BaseChatModel:
        """Lazy-load the reasoner model (VLM or API model).

        This model is used for:
        - reasoner node: final anomaly judgment based on vision-language reasoning (processes images)
        - memory node: summarizing detection results (text-only)

        Note: When using local model, uses Qwen VL model which supports both vision-language and text-only inputs.
        The reasoner node processes images, while memory node uses text-only.
        When using API model, uses the specified API model.

        Returns:
            BaseChatModel: The loaded model instance (local Qwen VL or API model).
        """
        if self.reasoner_model_type == "api":
            return self._load_reasoner_api_model()
        else:
            return self._load_qwen_model()

    def _load_embedding_model(self) -> Optional[Embeddings]:
        """Load the embedding model instance (lazy loaded, thread-safe).
        
        Default: uses QWEN_TEXT_MODEL_PATH or ./models/qwen3_4b (text-only LLM).
        If embedding_model_path is specified:
        - If path contains "vl", uses VLM for embeddings.
        - Otherwise, uses text-only LLM for embeddings.

        Returns:
            Embeddings: A wrapper around model for embedding tasks.
        """
        # Double-check locking pattern for thread safety
        if self._embedding_model_instance is None:
            with self._embedding_model_lock:
                # Check again after acquiring lock (another thread may have loaded it)
                if self._embedding_model_instance is None:
                    try:
                        from pathlib import Path
                        
                        # Determine which model path to use
                        if self.embedding_model_path is None:
                            # Default: use text-only LLM
                            model_path = DEFAULT_QWEN_TEXT_MODEL_PATH
                            use_vlm = False
                        else:
                            # Specified: check if path contains "vl" to determine if it's VLM
                            model_path = self.embedding_model_path
                            use_vlm = "vl" in model_path.lower()
                        
                        # Get basename for logging
                        model_name = Path(model_path).name
                        
                        print(f"[Context] Loading embedding model: {model_name}...")
                        
                        if use_vlm:
                            # Use VLM for embeddings - need to load VLM model at specified path
                            print(f"[Context] Using VLM model for embeddings: {model_name}")
                            # Load VLM model from the specified path
                            qwen_model = load_vl_model(
                                model_path=model_path,
                                trust_remote_code=True,
                                use_singleton=True,
                                attn_implementation=self.attn_implementation,
                                max_batch_size=self.max_batch_size,
                                max_wait_ms=self.max_wait_ms,
                            )
                            self._embedding_model_instance = QwenVLEmbeddings(qwen_model)
                        else:
                            # Use text-only LLM for embeddings
                            print(f"[Context] Using text-only LLM model for embeddings: {model_name}")
                            self._embedding_model_instance = TextModelEmbeddings(
                                model_path=model_path,
                                attn_implementation=self.attn_implementation,
                            )
                        
                        print(f"[Context] Embedding model loaded successfully: {model_name}")
                    except Exception as e:
                        print(f"[Context] Error loading embedding model: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

        return self._embedding_model_instance

    @property
    def embedding_model(self) -> Optional[Embeddings]:
        """Lazy-load the embedding model.

        This model is used for:
        - Text embedding in nodes for similarity search, clustering, etc.
        - Embedding-based heuristic decision in reasoner node.

        If embedding_model_path is None, defaults to text-only LLM.
        If embedding_model_path matches qwen_model_path, uses VLM.
        Otherwise, uses text-only LLM at the specified path.

        Returns:
            Embeddings: The embedding wrapper around model (LLM or VLM).
        """
        return self._load_embedding_model()

    async def aclose(self) -> None:
        """Close loaded API model clients before the event loop exits."""
        for model in (
            self._planner_api_model_instance,
            self._reflector_api_model_instance,
            self._reasoner_api_model_instance,
            self._image_description_api_model_instance,
            self._atomic_candidate_llm_api_model_instance,
        ):
            await _close_api_model_client(model)


if __name__ == "__main__":
    # Example: normal configuration
    ctx = Context(
        qwen_model_path=DEFAULT_QWEN_VL_MODEL_PATH,
        device="cuda"
        )
    print(type(ctx.planner_reflector_model))
    print(ctx.reasoner_model)
    
    # Example: debug configuration (single image, no batching)
    debug_ctx = Context(
        qwen_model_path=DEFAULT_QWEN_VL_MODEL_PATH,
        device="cuda",
        max_batch_size=1,  # Process one image at a time
        max_wait_ms=0,     # No waiting
    )
    print(f"\nDebug Context: batch_size={debug_ctx.max_batch_size}, wait={debug_ctx.max_wait_ms}ms")
