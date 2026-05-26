"""Define a custom Reasoning and Action agent with Planner-Reasoner-Reflector architecture.

Works with a chat model with tool calling support.

YI: the fourth naive agent graph (v1.9)
 -- with limited number of tool use and new heuristic breakout, draft version.
 -- Improved node identification in tools_node for better tool call tracking.
"""

# ============ Imports ============
import asyncio
import base64
import io
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from PIL import Image as PILImage

import warnings
import numpy as np
import hashlib

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore, IndexConfig

from .context_v1_27 import Context
from .state_v1_27 import InputState, State
from .tools_v1_27 import (
    TOOLS, 
    _extract_image_from_messages,
    _encode_image_to_base64,
    _decode_base64_image,
    _normalize_to_data_url,
    to_pure_base64,
    _pil_to_cv2,
    _cv2_to_pil,
    IMAGE_PROCESSING_AVAILABLE,
    PromptOrder,
    _keyword_heuristic_decision,
    counterfactual_atomic_candidate_tool,
)
from .store import short_term_memory_store, long_term_memory_store
from .utils_v1_27 import (
    load_vl_model,
    extract_images_for_reasoner,
    format_potential_anomalies,
    format_heuristic_prompts,
    format_executed_tool_calls,
    format_planner_reflector_sequence,
    count_input_output_tokens,
    _redact_tool_args,
    safe_api_invoke,
    resize_and_compress_image,
    estimate_bytes_from_data_url,
    _safe_image_for_mllm,
    fit_image_to_budget,
)
from .prompts_v1_27 import (
    IMAGE_SOURCE_INSTRUCTION_TEXT,
    TOOL_TARGET_INSTRUCTION_TEXT,
    TOOL_FAILURE_INSTRUCTION_TEXT,
)

# ============ Utilities ============
# Extract the first JSON object from plain text or ```json fenced code.
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

def _get_tool_call_name(tool_call: Any) -> str:
    """Unified accessor for tool_call name, supporting both dict and ToolCall object."""
    if isinstance(tool_call, dict):
        return tool_call.get("name", "")
    else:
        # Assume it's a ToolCall object or similar with .name attribute
        return getattr(tool_call, "name", "")

def _get_tool_call_args(tool_call: Any) -> dict:
    """Unified accessor for tool_call args, supporting both dict and ToolCall object."""
    if isinstance(tool_call, dict):
        return tool_call.get("args", {})
    else:
        # Assume it's a ToolCall object or similar with .args attribute
        args = getattr(tool_call, "args", {})
        return args if isinstance(args, dict) else {}

def _get_tool_call_id(tool_call: Any) -> str:
    """Unified accessor for tool_call id, supporting both dict and ToolCall object."""
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    else:
        # Assume it's a ToolCall object or similar with .id attribute
        return getattr(tool_call, "id", "")

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

def _ai_message_with_tool_calls(message: AIMessage, tool_calls: List[Any]) -> AIMessage:
    """Return an AIMessage with updated tool calls while preserving provider metadata."""
    try:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"tool_calls": tool_calls})
        if hasattr(message, "copy"):
            return message.copy(update={"tool_calls": tool_calls})
    except Exception:
        pass

    kwargs = {
        "content": message.content,
        "tool_calls": tool_calls,
        "id": getattr(message, "id", None),
        "name": getattr(message, "name", None),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
        "usage_metadata": getattr(message, "usage_metadata", None),
    }
    return AIMessage(**{key: value for key, value in kwargs.items() if value is not None})

def _validated_tool_image_base64(image_str: Any) -> Optional[str]:
    """Return canonical pure base64 for a decodable image tool argument."""
    if not isinstance(image_str, str) or not image_str.strip():
        return None

    normalized = _normalize_to_data_url(image_str)
    if not isinstance(normalized, str) or not normalized.startswith("data:image"):
        return None

    try:
        encoded = normalized.split(",", 1)[1]
    except Exception:
        return None

    old_max_pixels = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = 1024 * 1024 * 1024
    try:
        cleaned = encoded.strip()
        cleaned += "=" * (-len(cleaned) % 4)
        image_data = base64.b64decode(cleaned, validate=True)
        with PILImage.open(io.BytesIO(image_data)) as img:
            img.verify()
    except Exception:
        return None
    finally:
        PILImage.MAX_IMAGE_PIXELS = old_max_pixels

    pure_b64 = to_pure_base64(normalized)
    if isinstance(pure_b64, str) and len(pure_b64) > 100:
        return pure_b64
    return None

def _count_tokens_for_node(
    msg_list: List[BaseMessage | Dict[str, str]],
    output_msg: AIMessage,
    node_name: str,
) -> Dict[str, Any]:
    """
    Count tokens for a model call and return update dict for state.

    Behavior:
    1) Prefer actual counts from QwenBatcher (output_msg.additional_kwargs).
    2) If unavailable, estimate tokens from sanitized messages:
       - Replace image_url payloads (especially data URLs) with short placeholders
       - Avoid counting base64 as text tokens.
    3) Also report num_images_in_prompt for visibility.
    """

    def _count_images_in_content(content: Any) -> int:
        """Count image items inside a multimodal content list/dict."""
        cnt = 0
        if isinstance(content, list):
            for it in content:
                if isinstance(it, dict):
                    if it.get("type") == "image_url":
                        cnt += 1
                    elif it.get("type") == "image":
                        cnt += 1
        elif isinstance(content, dict):
            # Rare case: single dict content
            if content.get("type") in ("image_url", "image"):
                cnt += 1
        return cnt

    def _sanitize_content_for_token_count(content: Any) -> Any:
        """
        Sanitize content so text-token estimation does NOT include huge base64/data URLs.
        Keeps structure but replaces image payloads with placeholders.
        """
        if isinstance(content, list):
            new_list = []
            for it in content:
                if isinstance(it, dict):
                    t = it.get("type")
                    if t == "image_url":
                        # Replace potentially huge data URL / long URL with placeholder
                        new_list.append({"type": "image_url", "image_url": {"url": "<IMAGE>"}})
                    elif t == "image":
                        new_list.append({"type": "text", "text": "<IMAGE_OBJ>"})
                    else:
                        # Keep other structured items (e.g., {"type":"text","text":...})
                        new_list.append(it)
                else:
                    new_list.append(it)
            return new_list
        return content

    def _to_langchain_messages_sanitized(msg_list_in: Any) -> List[BaseMessage]:
        """Convert input msg_list (dict/BaseMessage mix) into sanitized LC messages for estimation."""
        lc_msgs: List[BaseMessage] = []
        for item in msg_list_in:
            if isinstance(item, BaseMessage):
                role = item.__class__.__name__.lower()
                content = getattr(item, "content", "")
                content = _sanitize_content_for_token_count(content)
                if "system" in role:
                    lc_msgs.append(SystemMessage(content=content))
                else:
                    lc_msgs.append(HumanMessage(content=content))
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = _sanitize_content_for_token_count(item.get("content", ""))
                if role == "system":
                    lc_msgs.append(SystemMessage(content=content))
                else:
                    lc_msgs.append(HumanMessage(content=content))
            else:
                # Fallback: stringify unknown item
                lc_msgs.append(HumanMessage(content=str(item)))
        return lc_msgs

    num_images = 0
    for item in msg_list:
        if isinstance(item, dict):
            num_images += _count_images_in_content(item.get("content", None))
        elif isinstance(item, BaseMessage):
            num_images += _count_images_in_content(getattr(item, "content", None))

    input_tokens = None
    output_tokens = None
    if hasattr(output_msg, "additional_kwargs") and output_msg.additional_kwargs:
        ak = output_msg.additional_kwargs
        if isinstance(ak, dict):
            input_tokens = ak.get("input_tokens")
            output_tokens = ak.get("output_tokens")

    if input_tokens is None or output_tokens is None:
        # IMPORTANT: use sanitized messages to avoid counting base64/data URLs as text
        input_msgs = _to_langchain_messages_sanitized(msg_list)
        token_counts = count_input_output_tokens(input_msgs, output_msg)
        input_tokens = int(token_counts.get("input_tokens", 0))
        output_tokens = int(token_counts.get("output_tokens", 0))
    else:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)

    total_tokens = input_tokens + output_tokens

    if node_name == "reflector":
        # Show only small, safe diagnostics—no giant serialization
        print("[REFLECTOR DEBUG] token_count:")
        print(f"  text_input_tokens={input_tokens:,}, text_output_tokens={output_tokens:,}, total={total_tokens:,}")
        print(f"  num_images_in_prompt={num_images}")

    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "num_images_in_prompt": num_images,  # extra metric (harmless if unused)
        "token_count_by_node": [{
            "node": node_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "num_images_in_prompt": num_images,
        }],
    }


# ============ Nodes ============

async def template_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Template tools are invoked to provide preliminary information for the planner.
    
    This node:
    1. Extracts and stores raw image URL in state
    2. Calls keyword heuristic decision to generate general_template_analysis
    3. Calls counterfactual atomic candidate tool to generate counterfactual_template_analysis
    4. Sets system message
    5. Stores general_template_analysis and counterfactual_template_analysis in state
    """
    ctx = runtime.context
    system_message = ctx.system_prompt.format(system_time=_now_iso())
    
    # Extract and store raw image URL in state (single source of truth for original image)
    raw_image_url = None
    if not isinstance(state, dict):
        raw_image_url = getattr(state, "raw_image_url", None)
    else:
        raw_image_url = state.get("raw_image_url", None)
    
    # If not already set, extract from messages
    if not raw_image_url:
        raw_image_url, _ = extract_images_for_reasoner(state.messages)
        if raw_image_url:
            # Normalize to ensure it's a data URL
            normalized = _normalize_to_data_url(raw_image_url)
            if normalized:
                raw_image_url = normalized
    
    # Step 1: Call keyword heuristic decision to generate general_template_analysis (decision=False to get full reason report)
    try:
        predicted_label, score1, score2, score3, general_template_analysis = await _keyword_heuristic_decision(state, runtime, decision=False)
        # Print predicted label and scores
        print(f"[Template Tools] General template analysis - Predicted label: {predicted_label}, Score 1: {score1:.3f}, Score 2: {score2:.3f}, Score 3: {score3:.3f}")
    except Exception as e:
        # If keyword heuristic fails, set general_template_analysis to error message
        general_template_analysis = f"Error in general template analysis (keyword heuristic decision): {str(e)}"
        print(f"[Template Tools] General template analysis error: {str(e)}")
    
    # Step 2: Call counterfactual atomic candidate tool to generate counterfactual_template_analysis
    try:
        cf_top_k = getattr(runtime.context, "counterfactual_top_k", 3)
        counterfactual_template_analysis = await counterfactual_atomic_candidate_tool(state, runtime, k=int(cf_top_k))
        # Print summary statistics (similar to general_template_analysis)
        # Read statistics from state (per-sample data)
        if not isinstance(state, dict):
            stats = getattr(state, "counterfactual_stats", None)
        else:
            stats = state.get("counterfactual_stats", None)
        
        if stats and isinstance(stats.get("templates"), list):
            hard_violation = bool(stats.get("hard_rule_violation", False))
            print(
                "[Template Tools] Counterfactual template analysis - "
                f"templates={len(stats['templates'])}, hard_rule_violation={hard_violation}"
            )
            for template_stat in stats["templates"]:
                idx = template_stat.get("template_index", "?")
                matching_rule = template_stat.get("matching_rule", "?")
                strictness = template_stat.get("strictness", "?")
                status = template_stat.get("status", "ok")
                if matching_rule == "visual_check":
                    print(
                        "  "
                        f"Template {idx}: {strictness}/{matching_rule}, status={status}, "
                        f"pass={template_stat.get('num_pass', 0)}, "
                        f"fail={template_stat.get('num_fail', 0)}, "
                        f"unknown={template_stat.get('num_unknown', 0)}"
                    )
                elif matching_rule == "textual_matching":
                    avg_margin = template_stat.get("avg_margin")
                    avg_margin_str = (
                        f"{avg_margin:.3f}"
                        if avg_margin is not None and np.isfinite(avg_margin)
                        else "N/A"
                    )
                    print(
                        "  "
                        f"Template {idx}: {strictness}/{matching_rule}, status={status}, "
                        f"valid={template_stat.get('num_valid', 0)}/{template_stat.get('num_total', 0)}, "
                        f"avg_margin={avg_margin_str}"
                    )
        elif stats and stats.get('num_valid', 0) > 0:
            avg_margin_str = f"{stats['avg_margin']:.3f}" if stats['avg_margin'] is not None and np.isfinite(stats['avg_margin']) else "N/A"
            avg_evidence_str = f"{stats['avg_evidence_strength']:.3f}" if stats['avg_evidence_strength'] is not None and np.isfinite(stats['avg_evidence_strength']) else "N/A"
            avg_sim_anom_str = f"{stats['avg_max_sim_anom']:.3f}" if stats['avg_max_sim_anom'] is not None and np.isfinite(stats['avg_max_sim_anom']) else "N/A"
            avg_sim_norm_str = f"{stats['avg_max_sim_norm']:.3f}" if stats['avg_max_sim_norm'] is not None and np.isfinite(stats['avg_max_sim_norm']) else "N/A"
            print(f"[Template Tools] Counterfactual template analysis - Valid: {stats['num_valid']}/{stats['num_total']}, Avg margin: {avg_margin_str}, Avg evidence: {avg_evidence_str}, Avg sim_anom: {avg_sim_anom_str}, Avg sim_norm: {avg_sim_norm_str}")
        elif stats:
            print(f"[Template Tools] Counterfactual template analysis - No valid perspectives ({stats['num_total']} total)")
        else:
            print(f"[Template Tools] Counterfactual template analysis generated successfully")
    except Exception as e:
        # If counterfactual tool fails, set counterfactual_template_analysis to error message
        counterfactual_template_analysis = f"Error in counterfactual template analysis (atomic candidate tool): {str(e)}"
        print(f"[Template Tools] Counterfactual template analysis error: {str(e)}")
    
    if not isinstance(state, dict):
        counterfactual_stats = getattr(state, "counterfactual_stats", None)
        counterfactual_reports = getattr(state, "counterfactual_reports", None)
        hard_rule_violation = getattr(state, "hard_rule_violation", False)
    else:
        counterfactual_stats = state.get("counterfactual_stats", None)
        counterfactual_reports = state.get("counterfactual_reports", None)
        hard_rule_violation = state.get("hard_rule_violation", False)

    return {
        "messages": [SystemMessage(content=system_message)],
        "raw_image_url": raw_image_url,
        "general_template_analysis": general_template_analysis,
        "counterfactual_template_analysis": counterfactual_template_analysis,
        "counterfactual_stats": counterfactual_stats,
        "counterfactual_reports": counterfactual_reports,
        "hard_rule_violation": bool(hard_rule_violation),
        "tool_call_threshold": ctx.tool_call_threshold,  # Set tool_call_threshold from context
        "normal_tolerance": ctx.normal_tolerance,  # Set normal_tolerance from context
        "max_reasoner_calls": ctx.max_reasoner_calls,  # Set max_reasoner_calls from context
    }

async def planner(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Use tool-enabled VLM to produce:
      - potential_anomalies: str
      - heuristic_prompt:    str
    Prompt: runtime.context.prompt4planner
    """
    ctx = runtime.context
    system_message = ctx.system_prompt.format(system_time=_now_iso())

    model = ctx.planner_model.bind_tools(TOOLS)

    # Build user content with general_template_analysis if available (first pass only)
    # Branch prompt based on model type: API models use tool calling, local models use JSON
    user_content_parts = []
    general_template_analysis = getattr(state, "general_template_analysis", None)
    if general_template_analysis:
        user_content_parts.append(f"General Template Analysis (Keyword Heuristic Decision):\n{general_template_analysis}")
    
    # Use different prompts for API vs local models
    if ctx.planner_model_type == "api":
        user_content_parts.append(ctx.prompt4planner_api)  # Don't require action JSON, use tool calling
    else:
        user_content_parts.append(ctx.prompt4planner_local)  # Allow action JSON
    
    user_content = "\n\n".join(user_content_parts)

    msg_list: List[BaseMessage | Dict[str, str]] = [
        _sys_msg(system_message),
        *state.messages,
        {"role": "user", "content": user_content},
    ]

    # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
    if ctx.planner_model_type == "api":
        resp = await safe_api_invoke(model, msg_list, "planner", ctx)
    else:
        resp = cast(AIMessage, await model.ainvoke(msg_list))
    token_updates = _count_tokens_for_node(msg_list, resp, "planner")
    
    # Parse JSON to extract potential_anomalies and heuristic_prompt
    # This should be done regardless of tool_calls, because:
    # 1. Local VLM models return JSON with these fields
    # 2. API models may return JSON in content even when tool_calls exist
    # 3. API models that don't call tools should still provide these fields in JSON
    parsed = {}
    potential_anomalies = ""
    heuristic_prompt = ""
    
    # Always try to parse JSON from content (API models may include JSON even with tool_calls)
    resp_text = _get_text(resp.content)
    if resp_text:
        parsed = _parse_json_from_text(resp_text)
        potential_anomalies = parsed.get("potential_anomalies", "")
        heuristic_prompt = parsed.get("heuristic_prompt", "")

        if not isinstance(potential_anomalies, str):
            potential_anomalies = ""
        if not isinstance(heuristic_prompt, str):
            heuristic_prompt = ""
    
    # If no tool_calls but JSON has "action", create tool_calls from JSON action field
    # This handles local VLM models that return JSON format
    if not resp.tool_calls and parsed.get("action"):
        # Local VLM model returned JSON format, convert to tool_calls
        action = parsed.get("action", "")
        args = parsed.get("args", {})
        
        # Find the tool by name
        tool_dict = {tool.name: tool for tool in TOOLS}
        if action in tool_dict:
            # Try to use LangChain's ToolCall type if available, otherwise use dict
            try:
                from langchain_core.messages.tool import ToolCall
                tool_call = ToolCall(
                    name=action,
                    args=args if isinstance(args, dict) else {},
                    id=str(uuid.uuid4()),
                )
            except ImportError:
                # Fallback: use dict format
                tool_call = {
                    "name": action,
                    "args": args if isinstance(args, dict) else {},
                    "id": str(uuid.uuid4()),
                }
            
            # Try to update tool_calls in place if possible, otherwise reconstruct
            try:
                # Try to set tool_calls directly (if the object allows it)
                if hasattr(resp, "tool_calls"):
                    resp.tool_calls = [tool_call]
                else:
                    # Fallback: reconstruct AIMessage, preserving all metadata
                    resp = _ai_message_with_tool_calls(resp, [tool_call])
            except Exception:
                # Last resort: reconstruct with minimal fields
                resp = _ai_message_with_tool_calls(
                    resp,
                    [{
                        "name": action,
                        "args": args if isinstance(args, dict) else {},
                        "id": str(uuid.uuid4()),
                    }],
                )
    # IMPORTANT: Always preserve potential_anomalies and heuristic_prompt from JSON parsing,
    # even if tool_calls were created. These fields are valuable for timeline and reasoner clues.

    out: Dict[str, Any] = {
        "messages": [resp],
        # Only append non-empty strings - avoid cluttering list with empty items
        "potential_anomalies": [potential_anomalies] if potential_anomalies else [],
        "heuristic_prompt": [heuristic_prompt] if heuristic_prompt else [],
        "last_caller": "planner",  # Mark caller for tools_node identification
        **token_updates,  # Merge token count updates
    }

    # Only increment num_tool_calls if tool_calls actually occurred
    # Return only delta - LangGraph's reducer (operator.add) will merge with existing
    num_new_tool_calls = len(resp.tool_calls or [])
    out["num_tool_calls"] = num_new_tool_calls
    # Preserve class_name if it exists in state
    class_name = getattr(state, "class_name", None)
    if class_name is not None:
        out["class_name"] = class_name
    return out


async def reasoner(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Produce:
      - result ∈ {"anomalous", "normal", "uncertain"}
      - reason: str

    If num_tool_calls > tool_call_threshold OR reasoner_call_count >= max_reasoner_calls,
    use prompt4reasoner_final for final decision.
    If API call fails, fallback to heuristic decision as last resort.
    """
    ctx = runtime.context
    # exceeds_threshold triggers final decision prompt when:
    # 1. Tool calls exceed threshold, OR
    # 2. Reasoner calls reach max limit (prevents infinite loops when model doesn't call tools)
    # IMPORTANT: Use (current + 1) for reasoner_call_count because the increment happens AFTER
    # this function returns, but route_reasoner_output sees the updated value. We must use the
    # same effective value to avoid returning "uncertain" when we're about to hit the limit.
    current_reasoner_calls = state.reasoner_call_count
    reasoner_call_count_after = current_reasoner_calls + 1  # Value after this call
    exceeds_tool_threshold = state.num_tool_calls > state.tool_call_threshold
    exceeds_reasoner_limit = reasoner_call_count_after >= ctx.max_reasoner_calls
    exceeds_threshold = exceeds_tool_threshold or exceeds_reasoner_limit
    
    if exceeds_reasoner_limit and not exceeds_tool_threshold:
        print(f"[reasoner] Using final prompt: reasoner_call_count={current_reasoner_calls}+1={reasoner_call_count_after} >= max_reasoner_calls={ctx.max_reasoner_calls}")

    system_message = ctx.system_prompt.format(system_time=_now_iso())
    model = ctx.reasoner_model
    
    # Use strengthened acquisition rules (same as reflector for consistency)
    # 1.1 Raw image: Priority from first HumanMessage, fallback to scan all messages
    raw_image = None
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", None)
            if content:
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if url:
                                    raw_image = _normalize_to_data_url(url)
                                    break
                            elif item.get("type") == "image" and "image" in item:
                                try:
                                    pil_img = item.get("image")
                                    if pil_img and isinstance(pil_img, PILImage.Image):
                                        encoded = _encode_image_to_base64(pil_img)
                                        # Normalize to data URL format for multimodal API compatibility
                                        raw_image = _normalize_to_data_url(encoded)
                                        break
                                except Exception:
                                    pass
                if raw_image:
                    break
    
    # Fallback: if no raw image found in HumanMessage, scan all messages
    if not raw_image:
        raw_image, _ = extract_images_for_reasoner(state.messages)
    
    # Ensure raw_image is normalized to data URL format
    if raw_image:
        raw_image = _normalize_to_data_url(raw_image)
    
    # 1.2 Augmented image: Priority from state fields (single source of truth)
    augmented_image = None
    
    # Priority: Read from state.latest_augmented_image_url (single source of truth)
    if hasattr(state, "latest_augmented_image_url") and state.latest_augmented_image_url:
        augmented_image = state.latest_augmented_image_url
    
    # Secondary: Obtain from extract_images_for_reasoner's reverse ToolMessage scan
    if not augmented_image:
        _, augmented_image = extract_images_for_reasoner(state.messages)
    
    potential_anomalies_list = getattr(state, "potential_anomalies", []) or []
    heuristic_prompt_list = getattr(state, "heuristic_prompt", []) or []
    
    potential_anomalies_text = format_potential_anomalies(potential_anomalies_list)
    heuristic_prompt_text = format_heuristic_prompts(heuristic_prompt_list)
    
    executed_tool_calls = getattr(state, "executed_tool_calls", []) or []
    tool_calls_text = format_executed_tool_calls(executed_tool_calls)
    
    # Build the user content with all structured information
    user_content_parts = []
    
    # Determine if this is the first or second reasoner call
    prev_calls = getattr(state, "reasoner_call_count", 0)
    is_first_call = (prev_calls == 0)
    
    if is_first_call:
        general_template_analysis = getattr(state, "general_template_analysis", None)
        if general_template_analysis:
            user_content_parts.append(f"General Template Analysis (Keyword Heuristic Decision):\n{general_template_analysis}")

    counterfactual_template_analysis = getattr(state, "counterfactual_template_analysis", None)
    if counterfactual_template_analysis:
        user_content_parts.append(
            "Counterfactual Template Analysis (multi-template evidence; may include soft textual matching "
            f"and hard visual rule checks):\n{counterfactual_template_analysis}"
        )
    if getattr(state, "hard_rule_violation", False):
        user_content_parts.append(
            "Hard Rule Alert: the configured hard visual counterfactual rule-violation threshold was met. "
            "Treat this as a strong anomaly signal unless the report itself says the evidence is unreliable."
        )
    
    if potential_anomalies_text:
        user_content_parts.append(f"Potential Anomalies Identified:\n{potential_anomalies_text}")
    
    if heuristic_prompt_text:
        user_content_parts.append(f"Heuristic Prompts:\n{heuristic_prompt_text}")
    
    if tool_calls_text:
        user_content_parts.append(tool_calls_text)
    
    raw_and_aug_different = bool(raw_image and augmented_image and raw_image != augmented_image)
    image_source_instruction = ""
    if ctx.raw_to_reasoner and raw_and_aug_different:
        image_source_instruction = IMAGE_SOURCE_INSTRUCTION_TEXT
    
    tool_failure_instruction = ""
    if getattr(state, "tool_failed", False):
        tool_failure_instruction = TOOL_FAILURE_INSTRUCTION_TEXT
    
    # Use prompt4reasoner_final if threshold is exceeded, otherwise use prompt4reasoner
    if exceeds_threshold:
        reasoner_prompt = ctx.prompt4reasoner_final
    else:
        reasoner_prompt = ctx.prompt4reasoner
    reasoner_prompt = reasoner_prompt.replace("<<IMAGE_SOURCE_INSTRUCTION>>", image_source_instruction)
    reasoner_prompt = reasoner_prompt.replace("<<TOOL_FAILURE_INSTRUCTION>>", tool_failure_instruction)
    user_content_parts.append(reasoner_prompt)
    
    user_content = "\n\n".join(user_content_parts)
    print(f"reasoner user_content: length={len(user_content)}")
    
    # Build message list with images
    msg_list: List[BaseMessage | Dict[str, str]] = [
        _sys_msg(system_message),
    ]
    
    # Add images if available
    MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
    
    image_content_list = []
    raw_safe = None
    if ctx.raw_to_reasoner and raw_image:
        # Raw image: use directly without compression (assumed to be within limits)
        # Only log estimated bytes for debugging
        raw_est = estimate_bytes_from_data_url(raw_image)
        print(f"[INFO] Reasoner raw image est_bytes: {raw_est:,}")
        raw_safe = raw_image
    
    aug_safe = None
    if augmented_image:
        aug_est = estimate_bytes_from_data_url(augmented_image)
        # Use fit_image_to_budget for augmented images (boundary-hugging compression)
        # fit_image_to_budget uses real jpeg_bytes internally, no need for estimate here
        try:
            aug_safe = fit_image_to_budget(
                augmented_image,
                max_bytes=MAX_IMAGE_BYTES,
                jpeg_quality=85,
                scale_decay=0.9,
                decode_func=_decode_base64_image,
            )
            # Log fit result for verification (estimate only for logging, not judgment)
            fit_est = estimate_bytes_from_data_url(aug_safe)
            fit_ratio = fit_est / MAX_IMAGE_BYTES if MAX_IMAGE_BYTES > 0 else 0
            print(f"[INFO] Reasoner augmented image: original_est={aug_est:,}, fit_est={fit_est:,}, ratio={fit_ratio:.4f}")
        except (ValueError, RuntimeError) as e:
            print(f"[WARNING] fit_image_to_budget failed: {e}, falling back to raw_image")
            aug_safe = raw_safe
    
    # Check if last tool was bypassed (image unchanged) - use state flag set by tool_node
    last_tool_bypassed = getattr(state, "last_tool_bypassed", False)
    
    if raw_safe:
        image_content_list.append({"type": "text", "text": "Original Image:"})
        image_content_list.append({"type": "image_url", "image_url": {"url": raw_safe}})
    
    if aug_safe:
        image_content_list.append({"type": "text", "text": "Latest Processed Image:"})
        image_content_list.append({"type": "image_url", "image_url": {"url": aug_safe}})
    
    # If last tool was bypassed, inform the model
    if last_tool_bypassed:
        image_content_list.append({"type": "text", "text": "Note: The last image processing tool was bypassed and did not modify the image (e.g., image too large for upscaling). The processed image may be the same as before."})
    
    if image_content_list:
        image_content_list.append({"type": "text", "text": user_content})
        msg_list.append({"role": "user", "content": image_content_list})
    else:
        msg_list.append({"role": "user", "content": user_content})

    # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
    model_call_failed = False
    model_error_msg = ""
    try:
        if ctx.reasoner_model_type == "api":
            resp = await safe_api_invoke(model, msg_list, "reasoner", ctx)
        else:
            resp = cast(AIMessage, await model.ainvoke(msg_list))
    except Exception as e:
        # Local model call failed (exception)
        model_call_failed = True
        model_error_msg = str(e)
        # Create a dummy response for error case
        resp = AIMessage(content=f"[ERROR: {model_error_msg}]")
    
    # Count tokens for this model call
    token_updates = _count_tokens_for_node(msg_list, resp, "reasoner")
    
    # Check if API/model call failed (error message from safe_api_invoke or exception)
    resp_content = _get_text(resp.content)
    api_error_detected = False
    api_error_msg = ""
    
    if resp_content.startswith("[ERROR:") or "[ERROR:" in resp_content or model_call_failed:
        # API/model call failed
        api_error_detected = True
        if model_call_failed:
            api_error_msg = f"Model call failed: {model_error_msg}"
        else:
            api_error_msg = resp_content
        # Try to extract JSON if present, but prioritize error message
        parsed = _parse_json_from_text(resp_content)
    else:
        parsed = _parse_json_from_text(resp_content)

    result_raw = str(parsed.get("result", "")).strip().lower()
    # For final prompt (exceeds_threshold), force hard label (anomalous/normal)
    if exceeds_threshold:
        if result_raw in {"anomalous", "normal"}:
            result = result_raw
        else:
            result = "normal"
    else:
        result = result_raw if result_raw in {"anomalous", "normal", "uncertain"} else "uncertain"

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    
    # Determine this call index based on how many times reasoner has been called so far
    # (prev_calls was already defined above when building user_content)
    call_index = prev_calls + 1
    
    # If model call failed when threshold exceeded, fallback to heuristic (forced=True)
    use_heuristic_fallback = False
    if exceeds_threshold and api_error_detected:
        use_heuristic_fallback = True
    
    if use_heuristic_fallback:
        # Fallback to heuristic decision as last resort (model call failed)
        label, score1, score2, score3, _ = await _keyword_heuristic_decision(state, runtime, decision=True)
        avg_score = (score1 + score2 + score3) / 3.0
        
        resp = AIMessage(
            content=f"[reasoner:forced@{state.num_tool_calls}] result={label} (scores=[{score1:.3f}, {score2:.3f}, {score3:.3f}], avg={avg_score:.3f})"
        )
        judgment = {
            "call_index": call_index,
            "result": label,
            "reason": (
                f"Forced decision at reasoner (fallback to heuristic) because "
                f"tool_usage={state.num_tool_calls} > {state.tool_call_threshold} and model call failed. "
                f"Predicted label: {label} (avg_score={avg_score:.3f})"
            ),
            "forced": True,
            "score": float(avg_score),
            "scores": [float(score1), float(score2), float(score3)],
        }
        token_updates_forced = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "token_count_by_node": [{
                "node": "reasoner",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }],
        }
        
        out = {
            "messages": [resp],
            "result": label,
            "reason": (
                f"Forced decision at reasoner (fallback to heuristic) because "
                f"tool_usage={state.num_tool_calls} > {state.tool_call_threshold} and model call failed. "
                f"Predicted label: {label} (avg_score={avg_score:.3f})"
            ),
            "reasoner_call_count": 1,  # Will be added to existing count via operator.add
            "reasoner_judgments": [judgment],  # Will be merged with existing judgments
            **token_updates_forced,  # Merge token count updates (0 tokens for heuristic path)
        }
        # Preserve class_name if it exists in state
        class_name = getattr(state, "class_name", None)
        if class_name is not None:
            out["class_name"] = class_name
        return out
    
    # Normal path: model call succeeded
    judgment = {
        "call_index": call_index,
        "result": result,
        "reason": reason,
        "forced": False,
    }

    out = {
        "messages": [resp],
        "result": result,
        "reason": reason,
        "reasoner_call_count": 1,  # Will be added to existing count via operator.add
        "reasoner_judgments": [judgment],  # Will be merged with existing judgments
        **token_updates,  # Merge token count updates
    }
    # Preserve class_name if it exists in state
    class_name = getattr(state, "class_name", None)
    if class_name is not None:
        out["class_name"] = class_name
    return out


async def reflector(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Run a planner-like re-analysis (tool-enabled) with prompt4reflector.
    Always routes to tools node after analysis, regardless of state.result.
    The reflector is called when reasoner returns "normal" or "uncertain", and performs
    additional analysis with access to image processing tools before routing to tools.
    For "normal" cases, it verifies that no anomalies were missed.
    For "uncertain" cases, it attempts to resolve the uncertainty.
    
    Structured input includes:
    1. System message
    2. Raw image (extracted from messages)
    3. Planner/reflector output sequence (potential anomalies, heuristic prompts, tool calls, reasoner judgments)
    4. Latest augmented image (from tool outputs)
    5. Prompt4Reflector
    """
    ctx = runtime.context

    # Get the previous reasoner result
    previous_result = (getattr(state, "result", "") or "").lower()
    
    tool_suggestion = ""
    if previous_result == "normal":
        # Extract tool names from executed_tool_calls
        executed_tool_calls = getattr(state, "executed_tool_calls", [])
        if executed_tool_calls:
            used_tools = set()
            for tc in executed_tool_calls:
                tool_name = tc.get("tool_name", "")
                if tool_name and tool_name.startswith("image_"):
                    used_tools.add(tool_name)
            
            if used_tools:
                # Available image processing tools
                all_tools = {"image_denoising", "image_deblurring", "image_super_resolution", "image_zooming", "image_brightness_enhancement"}
                unused_tools = all_tools - used_tools
                
                if unused_tools:
                    tool_list = ", ".join(sorted(unused_tools))
                    tool_suggestion = f"\n\nIMPORTANT: The previous analysis used the following tools: {', '.join(sorted(used_tools))}. To ensure thorough verification, you SHOULD use DIFFERENT tools this time. Consider using: {tool_list}."
                else:
                    # All tools were used, suggest reusing with different focus
                    tool_suggestion = f"\n\nIMPORTANT: All image processing tools have been used previously: {', '.join(sorted(used_tools))}. You may reuse tools but focus on different aspects or regions of the image for verification."

    system_message = ctx.system_prompt.format(system_time=_now_iso())
    model = ctx.reflector_model.bind_tools(TOOLS)
    
    # 1.1 Raw image: Priority from state field (single source of truth)
    raw_image = None
    
    # Priority 1: Read from state.raw_image_url (set by planner, already normalized)
    if hasattr(state, "raw_image_url") and state.raw_image_url:
        raw_image = state.raw_image_url
    
    # Priority 2: Extract from state.messages (HumanMessage)
    if not raw_image:
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                content = getattr(msg, "content", None)
                if content:
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "image_url":
                                    url = item.get("image_url", {}).get("url", "")
                                    if url:
                                        raw_image = _normalize_to_data_url(url)
                                        break
                                elif item.get("type") == "image" and "image" in item:
                                    try:
                                        pil_img = item.get("image")
                                        if pil_img and isinstance(pil_img, PILImage.Image):
                                            encoded = _encode_image_to_base64(pil_img)
                                            # Normalize to data URL format for multimodal API compatibility
                                            raw_image = _normalize_to_data_url(encoded)
                                            break
                                    except Exception:
                                        pass
                    if raw_image:
                        break
    
    # Priority 3: Fallback to extract_images_for_reasoner
    if not raw_image:
        raw_image, _ = extract_images_for_reasoner(state.messages)
    
    # Ensure raw_image is normalized to data URL format
    if raw_image:
        raw_image = _normalize_to_data_url(raw_image)
    
    augmented_image = None
    augmented_image_hash = None
    
    if hasattr(state, "latest_augmented_image_url") and state.latest_augmented_image_url:
        augmented_image = state.latest_augmented_image_url
        augmented_image_hash = getattr(state, "latest_augmented_image_hash", None)
    
    if not augmented_image:
        _, augmented_image = extract_images_for_reasoner(state.messages)
    
    if not augmented_image:
        print("[WARNING] No augmented image found in reflector; proceed with raw only.")
    
    potential_anomalies_list = getattr(state, "potential_anomalies", []) or []
    heuristic_prompt_list = getattr(state, "heuristic_prompt", []) or []
    
    latest_potential_anomalies = potential_anomalies_list[-2:] if potential_anomalies_list else []
    latest_heuristic_prompt = heuristic_prompt_list[-2:] if heuristic_prompt_list else []
    
    executed_tool_calls = getattr(state, "executed_tool_calls", []) or []
    reasoner_judgments = getattr(state, "reasoner_judgments", []) or []
    num_tool_calls = getattr(state, "num_tool_calls", 0)
    
    analysis_sequence = format_planner_reflector_sequence(
        latest_potential_anomalies,
        latest_heuristic_prompt,
        executed_tool_calls,
        reasoner_judgments,
        num_tool_calls,
    )
    
    user_content_parts = []
    
    # Add general_template_analysis and counterfactual_template_analysis if available
    general_template_analysis = getattr(state, "general_template_analysis", None)
    if general_template_analysis:
        user_content_parts.append(f"General Template Analysis (Keyword Heuristic Decision):\n{general_template_analysis}")
    counterfactual_template_analysis = getattr(state, "counterfactual_template_analysis", None)
    if counterfactual_template_analysis:
        user_content_parts.append(f"Counterfactual Template Analysis (Atomic Candidate Matching):\n{counterfactual_template_analysis}")
    
    if analysis_sequence:
        user_content_parts.append(analysis_sequence)
    
    raw_and_aug_different = bool(raw_image and augmented_image and raw_image != augmented_image)
    tool_target_instruction = ""
    if ctx.raw_to_reflector and raw_and_aug_different:
        tool_target_instruction = TOOL_TARGET_INSTRUCTION_TEXT
    
    # Branch prompt based on model type: API models use tool calling, local models use JSON
    if ctx.reflector_model_type == "api":
        reflector_prompt = ctx.prompt4reflector_api  # Don't require action JSON, use tool calling
    else:
        reflector_prompt = ctx.prompt4reflector_local  # Allow action JSON
    reflector_prompt = reflector_prompt.replace("<<TOOL_TARGET_INSTRUCTION>>", tool_target_instruction)
    user_content_parts.append(reflector_prompt)
    
    if tool_suggestion:
        user_content_parts.append(tool_suggestion)
    
    user_content = "\n\n".join(user_content_parts)
    print(f"reflector user_content: length={len(user_content)}")
    
    msg_list: List[BaseMessage | Dict[str, str]] = [
        _sys_msg(system_message),
    ]
    
    content_list = []
    
    # Determine which images to include based on raw_to_reflector setting
    # If raw_to_reflector=False but no augmented_image is available, fall back to raw_image
    should_show_raw = ctx.raw_to_reflector or not augmented_image
    
    if should_show_raw and raw_image:
        # Verify image URL format (must be data:image format to be recognized as multimodal)
        if not raw_image.startswith("data:image"):
            print(f"[WARNING] Raw image URL does not start with 'data:image': {raw_image[:100]}...")
            normalized_raw = _normalize_to_data_url(raw_image)
            if normalized_raw:
                raw_image = normalized_raw
            else:
                print(f"[ERROR] Failed to normalize raw image URL, may cause token explosion!")
        content_list.append({"type": "text", "text": "Original Image (raw):"})
        content_list.append({"type": "image_url", "image_url": {"url": raw_image}})
    
    # Size limit for augmented image (10MB JPEG file size)
    MAX_AUGMENTED_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
    
    if augmented_image:
        aug_est = estimate_bytes_from_data_url(augmented_image)
        # Use fit_image_to_budget for augmented images (boundary-hugging compression)
        try:
            augmented_image = fit_image_to_budget(
                augmented_image,
                max_bytes=MAX_AUGMENTED_IMAGE_BYTES,
                jpeg_quality=85,
                scale_decay=0.9,
                decode_func=_decode_base64_image,
            )
            fit_est = estimate_bytes_from_data_url(augmented_image)
            fit_ratio = fit_est / MAX_AUGMENTED_IMAGE_BYTES if MAX_AUGMENTED_IMAGE_BYTES > 0 else 0
            print(f"[INFO] Reflector augmented image: original_est={aug_est:,}, fit_est={fit_est:,}, ratio={fit_ratio:.4f}")
        except (ValueError, RuntimeError) as e:
            print(f"[WARNING] fit_image_to_budget failed in reflector: {e}, falling back to raw_image")
            augmented_image = raw_image  # fallback to raw_image
        
        if augmented_image:
            content_list.append({"type": "text", "text": "Latest Processed Image (augmented):"})
            content_list.append({"type": "image_url", "image_url": {"url": augmented_image}})
    
    # Check if last tool was bypassed - use state flag set by tool_node
    # Check if last tool was bypassed - use state flag set by tool_node
    last_tool_bypassed = getattr(state, "last_tool_bypassed", False)
    if last_tool_bypassed:
        content_list.append({"type": "text", "text": "Note: The last image processing tool was bypassed and did not modify the image (e.g., image too large for upscaling). Consider using different tools."})
    
    if not augmented_image:
        user_content = "WARNING: No augmented image available; proceed with raw only.\n\n" + user_content
    
    content_list.append({"type": "text", "text": user_content})
    
    image_count = sum(1 for item in content_list if isinstance(item, dict) and item.get("type") == "image_url")
    
    # Expected image count based on raw_to_reflector and availability
    expected_image_count = 0
    if should_show_raw and raw_image:
        expected_image_count += 1
    if augmented_image:
        expected_image_count += 1
    
    if image_count == 0:  # No images, use text-only
        msg_list.append({"role": "user", "content": user_content})
    else:
        msg_list.append({"role": "user", "content": content_list})
    
    if image_count != expected_image_count:
        if image_count == 0:
            print(f"[WARNING] Reflector has 0 images (expected {expected_image_count}). Raw: {raw_image is not None}, Augmented: {augmented_image is not None}, raw_to_reflector: {ctx.raw_to_reflector}")
        elif image_count == 1:
            print(f"[WARNING] Reflector has 1 image (expected {expected_image_count}). Raw: {raw_image is not None}, Augmented: {augmented_image is not None}, raw_to_reflector: {ctx.raw_to_reflector}")
        else:
            print(f"[WARNING] Reflector has {image_count} images (expected {expected_image_count}). Possible duplication issue.")
    
    # 5.2 Base64 leak assertion: check analysis_sequence and formatter output (suggestion #4)
    # Check not just user_content, but also analysis_sequence (which contains tool args)
    leak_detected = False
    leak_sources = []
    leak_details = []
    
    # Check user_content
    if "data:image" in user_content:
        leak_detected = True
        leak_sources.append("user_content")
        # Find all occurrences
        import re
        matches = list(re.finditer(r'data:image[^"\s]+', user_content))
        leak_details.append(f"user_content: {len(matches)} occurrences, first at position {matches[0].start() if matches else 'N/A'}")
    if "base64," in user_content:
        leak_detected = True
        if "user_content" not in leak_sources:
            leak_sources.append("user_content")
        leak_details.append("user_content: contains 'base64,'")
    
    # Check analysis_sequence (contains formatted tool calls with args)
    if analysis_sequence:
        if "data:image" in analysis_sequence:
            leak_detected = True
            leak_sources.append("analysis_sequence")
            import re
            matches = list(re.finditer(r'data:image[^"\s]+', analysis_sequence))
            leak_details.append(f"analysis_sequence: {len(matches)} occurrences, first at position {matches[0].start() if matches else 'N/A'}")
        if "base64," in analysis_sequence:
            leak_detected = True
            if "analysis_sequence" not in leak_sources:
                leak_sources.append("analysis_sequence")
            leak_details.append("analysis_sequence: contains 'base64,'")
    
    if leak_detected:
        print(f"[ERROR] Base64 leak detected in reflector! Sources: {leak_sources}")
        for detail in leak_details:
            print(f"[ERROR] {detail}")
        print(f"[ERROR] This suggests tool args redaction may not be working correctly.")
        print(f"[ERROR] user_content length: {len(user_content)}, analysis_sequence length: {len(analysis_sequence) if analysis_sequence else 0}")
    
    # 5.3 Verify image URL format in content_list (Reason B check)
    # Check if images in content_list are properly formatted as data:image URLs
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url and not url.startswith("data:image"):
                print(f"[ERROR] Image URL in content_list is not data:image format! This may cause token explosion.")
                print(f"[ERROR] URL preview: {url[:200]}...")
                print(f"[ERROR] URL length: {len(url)}")
    
    # Validation 1: Print image lengths and count before token counting (debugging)
    print(f"[REFLECTOR DEBUG] Before _count_tokens_for_node:")
    print(f"  len(raw_image): {len(raw_image) if raw_image else 0:,} (None if not available)")
    print(f"  len(augmented_image): {len(augmented_image) if augmented_image else 0:,} (None if not available)")
    print(f"  image_count: {image_count}")
    # if raw_image and len(raw_image) > 100000:
    #     print(f"  [WARNING] raw_image is very long ({len(raw_image):,} chars), may be data URL being counted as text!")
    # if augmented_image and len(augmented_image) > 100000:
    #     print(f"  [WARNING] augmented_image is very long ({len(augmented_image):,} chars), may be data URL being counted as text!")
    
    # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
    if ctx.reflector_model_type == "api":
        resp = await safe_api_invoke(model, msg_list, "reflector", ctx)
    else:
        resp = cast(AIMessage, await model.ainvoke(msg_list))
    token_updates = _count_tokens_for_node(msg_list, resp, "reflector")
    
    # Parse JSON to extract potential_anomalies and heuristic_prompt
    # This should be done regardless of tool_calls, because:
    # 1. Local VLM models return JSON with these fields
    # 2. API models may return JSON in content even when tool_calls exist
    # 3. API models that don't call tools should still provide these fields in JSON
    parsed = {}
    addl_potential = ""
    addl_heuristic = ""
    image_target = ""
    
    # Always try to parse JSON from content (API models may include JSON even with tool_calls)
    resp_text = _get_text(resp.content)
    if resp_text:
        parsed = _parse_json_from_text(resp_text)
        addl_potential = parsed.get("potential_anomalies", "")
        addl_heuristic = parsed.get("heuristic_prompt", "")
        image_target = parsed.get("image_target", "")

        if not isinstance(addl_potential, str):
            addl_potential = ""
        if not isinstance(addl_heuristic, str):
            addl_heuristic = ""
        if not isinstance(image_target, str):
            image_target = ""
    
    image_target = image_target.strip().lower()
    image_target = image_target if image_target in {"raw", "augmented"} else ""
    
    # If no tool_calls but JSON has "action", create tool_calls from JSON action field
    # This handles local VLM models that return JSON format
    if not resp.tool_calls and parsed.get("action"):
        # Local VLM model returned JSON format, convert to tool_calls
        action = parsed.get("action", "")
        args = parsed.get("args", {})
        # Check if image_base64 needs injection using actual image decode validation.
        existing_b64 = args.get("image_base64") if isinstance(args, dict) else None
        validated_existing_b64 = _validated_tool_image_base64(existing_b64)
        b64_needs_injection = validated_existing_b64 is None
        if isinstance(args, dict) and validated_existing_b64 is not None:
            args["image_base64"] = validated_existing_b64
        if (
            ctx.raw_to_reflector
            and raw_image
            and augmented_image
            and raw_image != augmented_image
            and isinstance(args, dict)
            and b64_needs_injection
        ):
            if image_target == "raw":
                args["image_base64"] = _validated_tool_image_base64(raw_image)
            elif image_target == "augmented":
                args["image_base64"] = _validated_tool_image_base64(augmented_image)
        
        # Find the tool by name
        tool_dict = {tool.name: tool for tool in TOOLS}
        if action in tool_dict:
            # Try to use LangChain's ToolCall type if available, otherwise use dict
            try:
                from langchain_core.messages.tool import ToolCall
                tool_call = ToolCall(
                    name=action,
                    args=args if isinstance(args, dict) else {},
                    id=str(uuid.uuid4()),
                )
            except ImportError:
                # Fallback: use dict format
                tool_call = {
                    "name": action,
                    "args": args if isinstance(args, dict) else {},
                    "id": str(uuid.uuid4()),
                }
            
            # Try to update tool_calls in place if possible, otherwise reconstruct
            try:
                # Try to set tool_calls directly (if the object allows it)
                if hasattr(resp, "tool_calls"):
                    resp.tool_calls = [tool_call]
                else:
                    # Fallback: reconstruct AIMessage, preserving all metadata
                    resp = _ai_message_with_tool_calls(resp, [tool_call])
            except Exception:
                # Last resort: reconstruct with minimal fields
                resp = _ai_message_with_tool_calls(
                    resp,
                    [{
                        "name": action,
                        "args": args if isinstance(args, dict) else {},
                        "id": str(uuid.uuid4()),
                    }],
                )
    # IMPORTANT: Always preserve addl_potential and addl_heuristic from JSON parsing,
    # even if tool_calls were created. These fields are valuable for timeline and reasoner clues.
    # If tool_calls exist and image_target requests raw, inject raw image into args when missing.
    if (
        ctx.raw_to_reflector
        and raw_image
        and augmented_image
        and raw_image != augmented_image
        and image_target in {"raw", "augmented"}
        and resp.tool_calls
    ):
        target_image = raw_image if image_target == "raw" else augmented_image
        target_b64 = _validated_tool_image_base64(target_image)
        normalized_calls = []
        for tc in resp.tool_calls:
            if isinstance(tc, dict):
                args = tc.get("args", {})
                if isinstance(args, dict):
                    args = dict(args)
                    existing = args.get("image_base64")
                    validated_existing = _validated_tool_image_base64(existing)
                    if validated_existing is not None:
                        args["image_base64"] = validated_existing
                    elif target_b64 is not None:
                        args["image_base64"] = target_b64
                normalized_calls.append({**tc, "args": args})
            else:
                args = getattr(tc, "args", {})
                if isinstance(args, dict):
                    args = dict(args)
                    existing = args.get("image_base64")
                    validated_existing = _validated_tool_image_base64(existing)
                    if validated_existing is not None:
                        args["image_base64"] = validated_existing
                    elif target_b64 is not None:
                        args["image_base64"] = target_b64
                try:
                    # Try to reconstruct ToolCall if available
                    from langchain_core.messages.tool import ToolCall
                    normalized_calls.append(ToolCall(name=tc.name, args=args, id=tc.id))
                except Exception:
                    normalized_calls.append({"name": tc.name, "args": args, "id": tc.id})
        try:
            resp.tool_calls = normalized_calls
        except Exception:
            # If tool_calls is immutable, reconstruct message with updated calls
            resp = _ai_message_with_tool_calls(resp, normalized_calls)

    # Only return NEW items - LangGraph's reducer (operator.add) will merge with existing
    # Only append non-empty strings - avoid cluttering list with empty items
    out: Dict[str, Any] = {
        "messages": [resp],
        "potential_anomalies": [f"[reflector]\n{addl_potential}"] if addl_potential else [],
        "heuristic_prompt": [f"[reflector]\n{addl_heuristic}"] if addl_heuristic else [],
        "last_caller": "reflector",  # Mark caller for tools_node identification
        "reflector_image_target": image_target or None,
        **token_updates,
    }
    # Only increment num_tool_calls if tool_calls actually occurred
    # Return only delta - LangGraph's reducer (operator.add) will merge with existing
    num_new_tool_calls = len(resp.tool_calls or [])
    out["num_tool_calls"] = num_new_tool_calls
    class_name = getattr(state, "class_name", None)
    if class_name is not None:
        out["class_name"] = class_name
    return out

async def memory(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Summarize state.messages into a concise memory blob and attach as `memory_summary`.
    Uses the reasoner_model by default (can be swapped via Context).
    """
    ctx = runtime.context
    system_message = ctx.system_prompt.format(system_time=_now_iso())
    model = getattr(ctx, "reasoner_model", None) or ctx.reasoner_model

    store: BaseStore = cast(
        BaseStore,
        runtime.store if runtime.store is not None else short_term_memory_store,
    )

    memories = await store.asearch(
        ("short_term_memory",),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"\n<memories>\n{formatted}\n</memories>"

    msg_list: List[BaseMessage | Dict[str, str]] = [
        _sys_msg(system_message),
        *state.messages,
        {
            "role": "user",
            "content": ctx.summary_prompt + (formatted or ""),
        },
    ]
    # Use safe_api_invoke for API models (with retry and error handling), direct invoke for local models
    if ctx.reasoner_model_type == "api":
        resp = await safe_api_invoke(model, msg_list, "memory", ctx)
    else:
        resp = cast(AIMessage, await model.ainvoke(msg_list))
    
    token_updates = _count_tokens_for_node(msg_list, resp, "memory")
    summary_text = _get_text(resp.content)

    # Memory update disabled to save memory
    # ns = ("short_term_memory",)
    # key = f"summary_{_now_iso()}"
    # await short_term_memory_store.aput(ns, key, summary_text)

    out = {
        "messages": [resp],
        "memory_summary": summary_text,
        **token_updates,
    }
    class_name = getattr(state, "class_name", None)
    if class_name is not None:
        out["class_name"] = class_name
    return out


# ============ Routing ============
# def route_reasoner_output(state: State) -> Literal["reflection", "memory"]:
#     """
#     If the last reasoner result is:
#       - 'anomalous' -> go to memory to store results, then end
#       - 'normal' and this is the second time reasoner is called -> go to memory (final result)
#       - 'normal' or 'uncertain' (first time) -> go to reflector for re-analysis
#     """
#     result = (getattr(state, "result", "") or "").lower()
#     reasoner_call_count = getattr(state, "reasoner_call_count", 0)
    
#     if result == "anomalous":
#         return "memory"
    
#     # If this is the second time reasoner is called and result is normal, output as final result
#     if result == "normal" and reasoner_call_count >= 2:
#         return "memory"
    
#     return "reflection"

def route_reflector_output(state: State) -> Literal["tool usage", "memory"]:
    """
    Reflector always routes to tools for additional analysis.
    This function is currently not used (reflector has a direct edge to tools),
    but kept for potential future use.
    
    Logic: Reflector should always go to tools to perform additional analysis,
    regardless of the previous result. The tools will then route back to reasoner
    for re-evaluation.
    """
    # Reflector always goes to tools for additional analysis
    return "tool usage"

def route_reasoner_output(state: State) -> Literal["reflection", "end"]:
    """
    If the last reasoner result is:
      - 'anomalous' -> go to end
      - 'normal' and reasoner_call_count >= normal_tolerance -> go to end (final result)
      - exceeds_threshold (num_tool_calls > tool_call_threshold OR reasoner_call_count >= max_reasoner_calls) -> go to end
      - otherwise -> go to reflector for re-analysis
    """
    result = (getattr(state, "result", "") or "").lower()
    reasoner_call_count = getattr(state, "reasoner_call_count", 0)
    normal_tolerance = getattr(state, "normal_tolerance", 2)  # Default to 2 for backward compatibility
    num_tool_calls = getattr(state, "num_tool_calls", 0)
    tool_call_threshold = getattr(state, "tool_call_threshold", 5)
    max_reasoner_calls = getattr(state, "max_reasoner_calls", 6)  # Default to 6 for safety
    
    # exceeds_threshold now includes both conditions
    exceeds_tool_threshold = num_tool_calls > tool_call_threshold
    exceeds_reasoner_limit = reasoner_call_count >= max_reasoner_calls
    exceeds_threshold = exceeds_tool_threshold or exceeds_reasoner_limit
    
    if result == "anomalous":
        return "end"
    
    # If result is normal and we've reached the tolerance threshold, accept as final result
    if result == "normal" and reasoner_call_count >= normal_tolerance:
        return "end"
    
    # If tool_call_threshold exceeded OR max_reasoner_calls reached, force end
    # (reasoner should have already made a hard decision using prompt4reasoner_final)
    if exceeds_threshold:
        if exceeds_reasoner_limit:
            print(f"[route_reasoner_output] Forcing end: reasoner_call_count={reasoner_call_count} >= max_reasoner_calls={max_reasoner_calls}")
        else:
            print(f"[route_reasoner_output] Forcing end: num_tool_calls={num_tool_calls} > tool_call_threshold={tool_call_threshold}")
        return "end"
    
    return "reflection"



# ============ Custom Tool Node ============

# List of image processing tool names
IMAGE_PROCESSING_TOOLS = {"image_denoising", "image_deblurring", "image_super_resolution", "image_zooming", "image_brightness_enhancement"}

async def tools_node_with_image_injection(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Custom tool node that automatically injects image data from state.messages
    into image processing tool calls.
    
    This wrapper:
    1. Extracts image from state.messages
    2. Checks if tool calls are for image processing tools
    3. If image_base64 is missing, automatically injects it
    4. Uses ToolNode to execute tools
    """
    # Get the last AIMessage with tool calls
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        # Not an AIMessage, return empty
        return {"messages": []}
    
    if not last_message.tool_calls:
        # No tool calls in the last message
        # Check if this is due to API error or model decision
        content = _get_text(last_message.content) if hasattr(last_message, 'content') else ""
        is_error = isinstance(content, str) and content.startswith("[ERROR")
        
        # Determine which node made this call to log appropriately
        calling_node = "unknown"
        for msg in reversed(state.messages[:-1]):
            if isinstance(msg, AIMessage):
                msg_content = _get_text(msg.content) if hasattr(msg, 'content') else ""
                if isinstance(msg_content, str):
                    content_lower = msg_content.lower()
                    if any(pattern in content_lower for pattern in ["planner", "[planner", "potential_anomalies", "heuristic_prompt"]):
                        calling_node = "planner"
                        break
                    elif any(pattern in content_lower for pattern in ["reflector", "[reflector", "result=normal", "result=anomalous", "result=uncertain"]):
                        calling_node = "reflector"
                        break
        if calling_node == "unknown":
            calling_node = "planner"  # Default
        
        if is_error:
            # API call failed - this is an exception case
            print(f"[TOOL_CALL] {calling_node} -> (no tools called - API ERROR)")
        else:
            # Model decided not to use tools - this is normal
            print(f"[TOOL_CALL] {calling_node} -> (no tools called)")
        return {"messages": []}
    
    # Determine which node made the tool call
    # Simple rule: first tool call is from planner, all subsequent calls are from reflector
    # Graph flow: planner -> tools -> reasoner -> reflector -> tools
    # Use last_caller from state (set by planner/reflector) for accurate identification
    last_caller = getattr(state, "last_caller", None)
    if last_caller in ("planner", "reflector"):
        calling_node = last_caller
    else:
        # Fallback: infer from executed_tool_calls history
        executed_tool_calls = getattr(state, "executed_tool_calls", [])
        if executed_tool_calls and len(executed_tool_calls) > 0:
            calling_node = "reflector"
        else:
            calling_node = "planner"
    
    # Extract image from messages (once, reuse for all image processing tools)
    extracted_image = _extract_image_from_messages(state.messages)
    
    raw_image = getattr(state, "raw_image_url", None)
    augmented_image = getattr(state, "latest_augmented_image_url", None)
    preferred_image = None
    if calling_node == "reflector":
        target = getattr(state, "reflector_image_target", None)
        if target == "raw" and raw_image:
            preferred_image = raw_image
        elif target == "augmented" and augmented_image:
            preferred_image = augmented_image

    preferred_b64 = _validated_tool_image_base64(preferred_image)
    extracted_b64 = _validated_tool_image_base64(extracted_image)
    
    # Modify tool calls to inject image_base64 if needed
    modified_tool_calls = []
    tool_names = []
    pre_messages = []
    pre_executed_calls = []
    injection_failed = False
    for tool_call in last_message.tool_calls:
        tool_name = _get_tool_call_name(tool_call)
        tool_args = _get_tool_call_args(tool_call).copy()  # Make a copy to modify
        
        # Check if this is an image processing tool and needs image injection
        if tool_name in IMAGE_PROCESSING_TOOLS:
            # Replace tool-provided base64 unless it decodes into a valid image.
            existing_b64 = tool_args.get("image_base64")
            validated_existing_b64 = _validated_tool_image_base64(existing_b64)
            needs_injection = validated_existing_b64 is None
            if validated_existing_b64 is not None:
                tool_args["image_base64"] = validated_existing_b64
            if needs_injection:
                if existing_b64:
                    print(
                        "[WARNING] Replacing invalid tool-provided image_base64 "
                        f"(tool={tool_name}, node={calling_node}, len={len(str(existing_b64))})"
                    )
                if preferred_b64:
                    tool_args["image_base64"] = preferred_b64
                elif extracted_b64:
                    tool_args["image_base64"] = extracted_b64
                else:
                    # No valid image data available for injection
                    target = getattr(state, "reflector_image_target", None)
                    print(
                        "[WARNING] No valid image for tool injection "
                        f"(tool={tool_name}, node={calling_node}, target={target}, "
                        f"has_raw={bool(raw_image)}, has_aug={bool(augmented_image)}, "
                        f"has_extracted={bool(extracted_image)})"
                    )
                    # Mark as failed and return explicit ToolMessage without calling tool
                    injection_failed = True
                    tool_call_id = _get_tool_call_id(tool_call) or str(uuid.uuid4())
                    pre_messages.append(
                        ToolMessage(
                            content="Error: No valid image available for tool injection.",
                            tool_call_id=tool_call_id,
                        )
                    )
                    safe_args = _redact_tool_args(tool_args)
                    pre_executed_calls.append({
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "args": safe_args,
                        "node": calling_node,
                    })
                    continue
        
        # Create modified tool call (ensure valid id for LangChain ToolMessage association)
        tool_call_id = _get_tool_call_id(tool_call) or str(uuid.uuid4())
        modified_tool_call = {
            "name": tool_name,
            "args": tool_args,
            "id": tool_call_id,
        }
        modified_tool_calls.append(modified_tool_call)
        tool_names.append(tool_name)
    
    # Log tool calls with clear format
    if tool_names:
        print(f"[TOOL_CALL] {calling_node} -> {', '.join(tool_names)}")
    elif pre_messages:
        print(f"[TOOL_CALL] {calling_node} -> (skipped tool call due to missing image)")
    
    # Keep provider metadata (Gemini 3 thought signatures live in content blocks
    # and additional_kwargs) while injecting tool args.
    modified_message = _ai_message_with_tool_calls(last_message, modified_tool_calls)
    
    # Get existing executed_tool_calls from state (if any)
    existing_executed_calls = []
    if hasattr(state, "executed_tool_calls"):
        existing_executed_calls = getattr(state, "executed_tool_calls", [])
        if not isinstance(existing_executed_calls, list):
            existing_executed_calls = []
    
    # Create a temporary state dict with the modified message
    # ToolNode expects a dict-like state with 'messages' key
    temp_state_dict = {
        "messages": [*state.messages[:-1], modified_message],
    }
    # Copy other state attributes if they exist
    if hasattr(state, "__dict__"):
        for key, value in state.__dict__.items():
            if key != "messages":
                temp_state_dict[key] = value
    
    # Ensure executed_tool_calls is in temp_state_dict
    if "executed_tool_calls" not in temp_state_dict:
        temp_state_dict["executed_tool_calls"] = existing_executed_calls
    
    # Use ToolNode to execute the tools
    # ToolNode.ainvoke expects a dict, not a State object
    if not modified_tool_calls:
        return {
            "messages": pre_messages,
            "tool_failed": True,
            "executed_tool_calls": pre_executed_calls,
        }
    tool_node = ToolNode(TOOLS)
    result = await tool_node.ainvoke(temp_state_dict)
    if isinstance(result, dict) and pre_messages:
        result["messages"] = pre_messages + result.get("messages", [])

    def _is_tool_error_content(content: Any) -> bool:
        if isinstance(content, str):
            text = content.strip().lower()
            return text.startswith("error:") or text.startswith("[error") or "error:" in text
        if isinstance(content, dict):
            if any(key.lower() == "error" for key in content.keys()):
                return True
            return any(isinstance(val, str) and "error" in val.lower() for val in content.values())
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = str(item.get("text", "")).lower()
                        if "error" in text:
                            return True
                elif isinstance(item, str):
                    if "error" in item.lower():
                        return True
        return False
    
    # For image processing tools, parse JSON response and create multimodal ToolMessage
    # so that reasoner can see the augmented figures
    # IMPORTANT: To avoid tensor size explosion (INT_MAX limit), we only keep the LAST
    # processed image from image processing tools. This prevents accumulation of too many
    # images in messages when multiple image processing tools are called.
    image_tool_failed = False  # Track if any image processing tool returned invalid data
    if isinstance(result, dict) and "messages" in result:
        modified_messages = []
        # Track the last image processing tool result
        last_image_tool_msg = None
        last_image_tool_name = None
        last_normalized_image = None
        last_text_content = None
        last_tool_bypassed_flag = False
        
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                # Find the corresponding tool call
                tool_call_id = msg.tool_call_id if hasattr(msg, 'tool_call_id') else None
                corresponding_tool_call = None
                for tc in modified_tool_calls:
                    # modified_tool_calls contains dicts, but use unified accessor for consistency
                    if _get_tool_call_id(tc) == tool_call_id:
                        corresponding_tool_call = tc
                        break
                
                # If this is an image processing tool, try to extract image data from response
                if corresponding_tool_call and _get_tool_call_name(corresponding_tool_call) in IMAGE_PROCESSING_TOOLS:
                    try:
                        content_str = msg.content if hasattr(msg, 'content') else ""
                        if isinstance(content_str, str) and content_str.strip():
                            # Try to extract image data (supports multiple formats)
                            
                            image_data = None
                            text_content = ""
                            field_name = None

                            # If tool returned an error string, skip image parsing
                            if content_str.lstrip().lower().startswith("error"):
                                modified_messages.append(msg)
                                continue
                            
                            # Format 1: JSON with processed_image_base64 field (preferred)
                            try:
                                tool_response = json.loads(content_str)
                                if isinstance(tool_response, dict):
                                    # Try various possible field names
                                    for cand in ["processed_image_base64", "image_base64", "image", "result_image", "output_image"]:
                                        if cand in tool_response:
                                            image_data = tool_response[cand]
                                            field_name = cand
                                            break
                                    # Extract text content if available
                                    text_content = tool_response.get("text", tool_response.get("message", "Image processing completed."))
                            except (json.JSONDecodeError, ValueError):
                                # Format 2: Direct base64 string or data URL (not JSON)
                                # Try to normalize directly as image data
                                image_data = content_str
                                text_content = "Image processing completed."
                            
                            # If we found image data that can be normalized, store this message
                            if image_data:
                                tool_name = _get_tool_call_name(corresponding_tool_call)
                                
                                # Check for BYPASS: prefix (tool returned unchanged image)
                                tool_bypassed = False
                                if isinstance(image_data, str) and image_data.startswith("BYPASS:"):
                                    image_data = image_data[7:]  # Remove "BYPASS:" prefix
                                    tool_bypassed = True
                                    print(f"[TOOL_IMAGE] {tool_name}: BYPASSED (image unchanged)")
                                
                                image_len = len(image_data) if isinstance(image_data, str) else None
                                if not tool_bypassed:
                                    print(f"[TOOL_IMAGE] {tool_name}: field={field_name or 'raw'} len={image_len}")
                                normalized = _normalize_to_data_url(image_data)
                                if normalized:
                                    # Store this as the last image processing result
                                    last_image_tool_msg = msg
                                    last_image_tool_name = tool_name
                                    last_normalized_image = normalized
                                    last_text_content = text_content
                                    last_tool_bypassed_flag = tool_bypassed  # Track bypass status
                                    # Don't add it yet - we'll add only the last one
                                    continue
                                else:
                                    print(f"[WARNING] tool output invalid base64 (tool={tool_name}, len={image_len})")
                                    image_tool_failed = True  # Mark as failed
                        
                        # Fallback to original message if no valid image data found
                        modified_messages.append(msg)
                    except Exception as e:
                        print(f"[DEBUG tools_node] Error parsing tool response: {e}")
                        # Fallback to original message if processing fails
                        modified_messages.append(msg)
                        continue
                else:
                    # Fallback: Non-image-processing tool ToolMessage - must keep it
                    modified_messages.append(msg)
                    continue
            else:
                # For non-image-processing tools, keep original message
                modified_messages.append(msg)
        
        # Add only the LAST image processing tool result as multimodal message
        # This prevents tensor size explosion when multiple image processing tools are used
        # Also write to state fields as single source of truth (suggestion #3)
        if last_image_tool_msg is not None:
            try:
                # Use the normalized image and text we stored earlier
                normalized_image_url = last_normalized_image
                text_content = last_text_content or "Image processing completed."
                
                if normalized_image_url:
                    # Write to state fields as single source of truth
                    import hashlib
                    # Compute hash for deduplication/debugging
                    image_hash = hashlib.sha256(normalized_image_url.encode('utf-8')).hexdigest()[:16]
                    
                    # Unconditionally inject the normalized image
                    # Create multimodal content: text + image
                    new_content = [
                        {
                            "type": "text",
                            "text": text_content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": normalized_image_url
                            }
                        }
                    ]
                    
                    # Create new ToolMessage with processed image
                    tool_call_id = last_image_tool_msg.tool_call_id if hasattr(last_image_tool_msg, 'tool_call_id') else None
                    new_msg = ToolMessage(
                        content=new_content,
                        tool_call_id=tool_call_id,
                        id=last_image_tool_msg.id if hasattr(last_image_tool_msg, 'id') else None
                    )
                    modified_messages.append(new_msg)
                    
                    # Update state with latest augmented image (single source of truth)
                    result["latest_augmented_image_url"] = normalized_image_url
                    result["latest_augmented_image_hash"] = image_hash
                    # Track if the last tool was bypassed (image unchanged)
                    result["last_tool_bypassed"] = last_tool_bypassed_flag
                else:
                    # Failed to normalize image data, fallback to original message
                    modified_messages.append(last_image_tool_msg)
            except Exception as e:
                # Fallback: add original message
                modified_messages.append(last_image_tool_msg)
        
        # Replace messages with modified ones
        result["messages"] = modified_messages

    # Record whether any tool returned an error-like message
    # Start with injection_failed or image_tool_failed to ensure they propagate
    tool_failed = injection_failed or image_tool_failed
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, "content") else ""
                if _is_tool_error_content(content):
                    tool_failed = True
                    break
    if isinstance(result, dict):
        result["tool_failed"] = tool_failed
    
    # Save tool call information to state so it can be extracted later
    # Extract tool call info from modified_tool_calls before they're lost
    # Use _redact_tool_args to properly filter all image/base64 fields (prevent token explosion)
    
    executed_tool_calls = []
    if pre_executed_calls:
        executed_tool_calls.extend(pre_executed_calls)
    for tc in modified_tool_calls:
        args = tc.get("args", {})
        # Use unified redaction function to filter all image/base64/mask fields
        safe_args = _redact_tool_args(args)
        
        executed_tool_calls.append({
            "tool_name": tc.get("name", "unknown"),
            "tool_call_id": tc.get("id", "unknown"),
            "args": safe_args,
            "node": calling_node,
        })
    
    # IMPORTANT: Only return NEW tool calls, not the merged list
    # LangGraph's reducer will merge them with existing ones
    # If we return the merged list, reducer will merge again and cause duplication
    
    # Ensure result is a dict and add executed_tool_calls (only new ones)
    if not isinstance(result, dict):
        result = {"messages": result} if hasattr(result, "__iter__") else {}
    
    # Return only the NEW tool calls - LangGraph will merge with existing via reducer
    result["executed_tool_calls"] = executed_tool_calls
    
    return result

# Create the tool node (use custom wrapper)
tool_node = tools_node_with_image_injection


# ============ Graph Definition ============
builder = StateGraph(State, input_schema=InputState, context_schema=Context) # schema changes to State

builder.add_node("template_tools", template_tools)
builder.add_node("planner", planner)
builder.add_node("tools", tool_node)
builder.add_node("reasoner", reasoner)
builder.add_node("reflector", reflector)
builder.add_node("memory", memory)

# Flow: start -> template_tools -> planner -> tools -> reasoner -> reflector
builder.add_edge("__start__", "template_tools")
builder.add_edge("template_tools", "planner")
builder.add_edge("planner", "tools")
builder.add_edge("tools", "reasoner")

# builder.add_conditional_edges("reasoner", route_reasoner_output, {
#     "reflection": "reflector",
#     "memory": "memory"
# })

builder.add_conditional_edges("reasoner", route_reasoner_output, {
    "reflection": "reflector",
    "end": "__end__"
})

builder.add_edge("reflector", "tools")
# builder.add_edge("memory", "__end__")

graph = builder.compile(name="Agent_v1")


# TODO: 
# 1) control the output format of the LLMs
# 2) consider extending the long-term memory
# 2) consider extending the long-term memory
