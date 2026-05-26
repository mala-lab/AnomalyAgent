"""
Evaluate agent_v1 graph on MVTec dataset using WinCLIP evaluation metrics.

This script:
1. Loads MVTec test dataset
2. Runs agent_v1 graph on each image
3. Collects anomaly detection results
4. Evaluates using WinCLIP evaluation functions

Notes:
- Mean AUROC: consistently skip NaN when printing and saving.
- Scoring: configurable --uncertain_score (default 0.5); if agent state carries a
  numeric score/confidence, prefer that; recursion-limit fallback uses the graph's
  own heuristic function (_keyword_heuristic_decision).
- Tool calls tracking: This version tracks and records all tool calls made during execution.
- Reasoning timeline: This version (v6) includes reasoning_timeline showing chronological tool calls and judgments.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from tabulate import tabulate
from tqdm import tqdm


DEFAULT_MVTEC_DATA_PATH = os.environ.get("MVTEC_DATA_PATH", "./data/mvtec")
DEFAULT_QWEN_VL_MODEL_PATH = os.environ.get("QWEN_VL_MODEL_PATH", "./models/qwen3_vl_4b")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# === Import new agent graph and its heuristic ===
from react_agent.agent_v1_16 import graph, _keyword_heuristic_decision
from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.utils import generate_and_save_atomic_candidates
from utils.image_utils import resize_image
# from react_agent.utils import extract_reasoning_timeline_from_state


def load_mvtec_test_data(data_path: str) -> List[Dict]:
    """Load MVTec test dataset from meta.json.

    Args:
        data_path: Path to MVTec data directory.

    Returns:
        List of test samples, each containing img_path, mask_path, cls_name, anomaly label.
    """
    meta_path = os.path.join(data_path, "meta.json")
    with open(meta_path, "r") as f:
        meta_info = json.load(f)

    test_data = []
    for cls_name, samples in meta_info.get("test", {}).items():
        for sample in samples:
            sample["full_img_path"] = os.path.join(data_path, sample["img_path"])
            sample["full_mask_path"] = (
                os.path.join(data_path, sample["mask_path"])
                if sample.get("mask_path")
                else None
            )
            test_data.append(sample)

    return test_data


def try_extract_numeric_score_from_state(state_obj) -> Optional[float]:
    """Try to extract a numeric score/confidence from agent state if provided by the graph."""
    # Handle both dict and State object
    if isinstance(state_obj, dict):
        candidate_attrs = ["score", "confidence", "anomaly_score", "prob", "probability"]
        for name in candidate_attrs:
            if name in state_obj:
                try:
                    val = state_obj[name]
                    if isinstance(val, (int, float)):
                        # clamp to [0,1]
                        return float(np.clip(val, 0.0, 1.0))
                except Exception:
                    pass
    else:
        # Common attribute names that may appear
        candidate_attrs = ["score", "confidence", "anomaly_score", "prob", "probability"]
        for name in candidate_attrs:
            if hasattr(state_obj, name):
                try:
                    val = getattr(state_obj, name)
                    if isinstance(val, (int, float)):
                        # clamp to [0,1]
                        return float(np.clip(val, 0.0, 1.0))
                except Exception:
                    pass
    return None

def extract_tool_calls_from_state(state) -> List[Dict[str, any]]:
    """Extract all tool calls from state.
    
    First tries to get from state.executed_tool_calls (saved by tools_node),
    then falls back to extracting from state.messages.
    
    Args:
        state: State object or dict containing messages.
        
    Returns:
        List of tool call dictionaries, each containing:
        - tool_name: name of the tool
        - tool_call_id: unique ID for this tool call
        - args: arguments passed to the tool
        - node: which node made the call (planner/reflector)
    """
    tool_calls_list = []
    
    # First, try to get from state.executed_tool_calls (saved by tools_node)
    if isinstance(state, dict):
        executed_calls = state.get("executed_tool_calls", [])
    else:
        executed_calls = getattr(state, "executed_tool_calls", [])
    
    if executed_calls and isinstance(executed_calls, list) and len(executed_calls) > 0:
        # executed_calls should already be in the correct format
        tool_calls_list.extend(executed_calls)
        return tool_calls_list
    
    # Fallback: extract from messages (should not be needed if tools_node works correctly)
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    
    # Track which node we're in based on message patterns
    current_node = "unknown"
    
    # Also track tool calls from ToolMessage (they have tool_call_id)
    tool_call_id_to_name = {}
    
    for msg in messages:
        # Check if this is a ToolMessage - extract tool name from content
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", "") or ""
            # Try to infer tool name from content
            if "denoising" in content.lower():
                tool_name = "image_denoising"
            elif "deblurring" in content.lower():
                tool_name = "image_deblurring"
            elif "super-resolution" in content.lower() or "upscaled" in content.lower():
                tool_name = "image_super_resolution"
            elif "zooming" in content.lower() or "zoomed" in content.lower():
                tool_name = "image_zooming"
            elif "brightness" in content.lower() or "enhancement" in content.lower() or "clahe" in content.lower():
                tool_name = "image_brightness_enhancement"
            else:
                tool_name = "unknown_tool"
            
            if tool_call_id:
                tool_call_id_to_name[tool_call_id] = tool_name
        
        # Check if this is an AIMessage with tool_calls
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                # Try to infer which node made the call based on message content
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str):
                    content_lower = content.lower()
                    # Check for planner patterns
                    if any(pattern in content_lower for pattern in ["planner", "[planner", "potential_anomalies", "heuristic_prompt"]):
                        current_node = "planner"
                    # Check for reflector patterns
                    elif any(pattern in content_lower for pattern in ["reflector", "[reflector", "result=normal", "result=anomalous", "result=uncertain"]):
                        current_node = "reflector"
                    # Check for reasoner patterns
                    elif any(pattern in content_lower for pattern in ["reasoner", "[reasoner", "result="]):
                        current_node = "reasoner"
                
                # If still unknown, check previous messages to infer node
                if current_node == "unknown":
                    msg_index = messages.index(msg)
                    # Check previous messages for context
                    for prev_msg in reversed(messages[:msg_index]):
                        if isinstance(prev_msg, AIMessage):
                            prev_content = getattr(prev_msg, "content", "") or ""
                            if isinstance(prev_content, str):
                                prev_content_lower = prev_content.lower()
                                if any(pattern in prev_content_lower for pattern in ["planner", "[planner", "potential_anomalies"]):
                                    current_node = "planner"
                                    break
                                elif any(pattern in prev_content_lower for pattern in ["reflector", "[reflector", "result=normal", "result=anomalous"]):
                                    current_node = "reflector"
                                    break
                
                # Default to planner if still unknown (first tools call is usually from planner)
                if current_node == "unknown":
                    current_node = "planner"
                
                # Extract tool calls
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get("name", "unknown")
                        tool_call_id = tc.get("id", "unknown")
                        args = tc.get("args", {})
                    else:
                        tool_name = getattr(tc, "name", getattr(tc, "function", {}).get("name", "unknown") if hasattr(tc, "function") else "unknown")
                        tool_call_id = getattr(tc, "id", "unknown")
                        args = getattr(tc, "args", getattr(tc, "function", {}).get("arguments", {}) if hasattr(tc, "function") else {})
                        if isinstance(args, str):
                            try:
                                import json
                                args = json.loads(args)
                            except Exception:
                                args = {}
                    
                    tool_calls_list.append({
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "args": args,
                        "node": current_node,
                    })
    
    # Also add tool calls found from ToolMessage if we didn't find them in AIMessage
    if not tool_calls_list:
        for tool_call_id, tool_name in tool_call_id_to_name.items():
            tool_calls_list.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "args": {},
                "node": "unknown",
            })
    
    return tool_calls_list


def extract_reasoner_judgments_from_state(state) -> List[Dict[str, any]]:
    """Extract all reasoner judgments from state.
    
    Args:
        state: State object or dict containing reasoner_judgments.
        
    Returns:
        List of judgment dictionaries, each typically containing:
        - call_index: 1-based index of the reasoner call
        - result: reasoner output label
        - reason: textual explanation
        - forced: whether the decision was forced by heuristic/limits
    """
    if isinstance(state, dict):
        judgments = state.get("reasoner_judgments", [])
    else:
        judgments = getattr(state, "reasoner_judgments", [])

    if not isinstance(judgments, list):
        return []
    return judgments


## load an image and invoke the graph + lots of safety checks
async def run_agent_on_image(
    graph_instance,
    image_path: str,
    class_name: str,
    context: Context,
    show_token_num: bool = False,
    resize: bool = False,
    ) -> Tuple[str, str, Optional[float], List[Dict[str, any]], List[Dict[str, any]], List[Dict[str, any]]]:
    """Run agent graph on a single image.

    Returns:
        (result, reason, score, tool_calls, reasoner_judgments, reasoning_timeline) where:
        - result in {"anomalous","normal"}
        - score in [0,1] or None
        - tool_calls: list of tool call dictionaries
        - reasoner_judgments: list of reasoner judgment dictionaries
        # - reasoning_timeline: chronological timeline of tool calls and judgments
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    if resize:
        image = resize_image(image, 518)

    # Create input message with image (prefer PIL.Image directly, fallback to base64)
    import base64
    from io import BytesIO
    import warnings
    
    def _message_contains_image(msg: HumanMessage) -> bool:
        """Check if a HumanMessage contains image data."""
        content = msg.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        # Check if it has PIL.Image or valid image data
                        if "image" in item:
                            from PIL import Image
                            if isinstance(item["image"], Image.Image):
                                return True
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image") or url.startswith("http"):
                            return True
                else:
                    # Check if it's a PIL.Image object
                    try:
                        from PIL import Image
                        if isinstance(item, Image.Image):
                            return True
                    except ImportError:
                        pass
        return False
    
    try:
        # Primary method: pass PIL.Image directly
        user_message = HumanMessage(
            content=[
                {"type": "text", "text": f"Please analyze this {class_name} image for anomalies."},
                {"type": "image", "image": image},   # pass PIL.Image directly
            ]
        )
        if not _message_contains_image(user_message):
            warnings.warn(
                f"WARNING: Image may not be included in message for {image_path}. "
                f"Primary method created message but image validation failed."
            )
    except Exception as e1:
        # Fallback 1: try base64 encoding
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            user_message = HumanMessage(
                content=[
                    {"type": "text", "text": f"Please analyze this {class_name} image for anomalies."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ]
            )
            if not _message_contains_image(user_message):
                warnings.warn(
                    f"WARNING: Image may not be included in message for {image_path}. "
                    f"Fallback 1 (base64) created message but image validation failed."
                )
        except Exception as e2:
            # Fallback 2: pure-text reference (last resort, but this will not work well for vision models)
            warnings.warn(
                f"CRITICAL WARNING: Failed to include image in message for {image_path}. "
                f"Primary error: {e1}, Fallback error: {e2}. "
                f"Using text-only fallback - Qwen model will NOT see the image!"
            )
            user_message = HumanMessage(
                content=f"Please analyze this {class_name} image located at {image_path} for anomalies. "
                f"The image shows a {class_name} that may or may not contain defects."
            )

    # Initial state
    initial_state = InputState(messages=[user_message], class_name=class_name)

    # Run graph with config
    try:
        from react_agent.store import short_term_memory_store
        from langgraph.errors import GraphRecursionError

        config = {
            "recursion_limit": 20,
            "configurable": {"store": short_term_memory_store},
        }

        # Use ainvoke to get the final merged state (includes all merged fields)
        # This is better than astream because it returns the complete final state
        last_state = None
        try:
            result_state = await graph_instance.ainvoke(initial_state, config=config, context=context)
            # Check executed_tool_calls in final state
            if isinstance(result_state, dict):
                exec_calls = result_state.get("executed_tool_calls", [])
            elif hasattr(result_state, "executed_tool_calls"):
                exec_calls = getattr(result_state, "executed_tool_calls", [])
        except GraphRecursionError:
            # If recursion limit hit, use astream and get the last complete state
            async for chunk in graph_instance.astream(initial_state, config=config, context=context):
                if chunk:
                    # chunk is a dict like {"node_name": state}
                    # Each state in chunk is the merged state after that node executes
                    for node_name, node_state in chunk.items():
                        # Keep the last state (should be the final merged state)
                        last_state = node_state
            
            result_state = last_state if last_state else initial_state

            # Handle both dict and State object
            if isinstance(result_state, dict):
                final_result = result_state.get("result", "")
                reason = result_state.get("reason", "")
            else:
                final_result = getattr(result_state, "result", "")
                reason = getattr(result_state, "reason", "")

            valid_labels = {"anomalous", "normal", "uncertain"}
            final_result_lower = final_result.lower().strip() if isinstance(final_result, str) else ""

            # If result is uncertain or invalid, use embedding-based heuristic to force a decision
            if not isinstance(final_result, str) or final_result_lower not in valid_labels or final_result_lower == "uncertain":
                # Ensure class_name is in state (it might have been lost during graph execution)
                # result_state might be a dict or State object
                if isinstance(result_state, dict):
                    result_state["class_name"] = class_name
                elif not hasattr(result_state, "class_name") or result_state.class_name is None:
                    result_state.class_name = class_name
                
                # Use the **same** heuristic as the agent (imported from agent_v1)
                # Create a simple runtime wrapper for the heuristic function
                class MockRuntime:
                    def __init__(self, ctx):
                        self.context = ctx
                        self.store = None
                
                mock_runtime = MockRuntime(context)
                heuristic_label, score1, score2, score3, suffix = await _keyword_heuristic_decision(result_state, mock_runtime, decision=True)
                heuristic_score = (score1 + score2 + score3) / 3.0  # Calculate average score
                reason = f"Recursion limit reached. {suffix}"
                tool_calls = extract_tool_calls_from_state(result_state)
                reasoner_judgments = extract_reasoner_judgments_from_state(result_state)
                # reasoning_timeline = extract_reasoning_timeline_from_state(result_state)
                reasoning_timeline = None
                return heuristic_label, reason, heuristic_score, tool_calls, reasoner_judgments, reasoning_timeline
            else:
                # Valid result (anomalous or normal), return it directly
                tool_calls = extract_tool_calls_from_state(result_state)
                reasoner_judgments = extract_reasoner_judgments_from_state(result_state)
                # reasoning_timeline = extract_reasoning_timeline_from_state(result_state)
                reasoning_timeline = None
                return (
                    final_result_lower,
                    "Recursion limit reached. Using partial result.",
                    try_extract_numeric_score_from_state(result_state),
                    tool_calls,
                    reasoner_judgments,
                    reasoning_timeline,
                )

        # Normal path: extract result/score
        # Handle both dict and State object
        if isinstance(result_state, dict):
            final_result = result_state.get("result", "uncertain")
            reason = result_state.get("reason", "")
        else:
            final_result = getattr(result_state, "result", "uncertain")
            reason = getattr(result_state, "reason", "")

        # Normalize result
        if isinstance(final_result, str):
            final_result = final_result.lower().strip()
        else:
            final_result = "uncertain"

        if final_result not in {"anomalous", "normal", "uncertain"}:
            final_result = "uncertain"

        # If graph completed normally but result is still "uncertain", 
        # use embedding-based heuristic to force a decision
        # (This shouldn't happen normally, but handle it as a safety measure)
        heuristic_score = None
        if final_result == "uncertain":
            # Ensure class_name is in state (it might have been lost during graph execution)
            # result_state might be a dict or State object
            if isinstance(result_state, dict):
                result_state["class_name"] = class_name
            elif not hasattr(result_state, "class_name") or result_state.class_name is None:
                result_state.class_name = class_name
            
            class MockRuntime:
                def __init__(self, ctx):
                    self.context = ctx
                    self.store = None
            
            mock_runtime = MockRuntime(context)
            heuristic_label, score1, score2, score3, suffix = await _keyword_heuristic_decision(result_state, mock_runtime, decision=True)
            heuristic_score = (score1 + score2 + score3) / 3.0  # Calculate average score
            final_result = heuristic_label
            reason = f"Graph completed but result was uncertain, using heuristic-based decision. {suffix} Original reason: {reason}"

        # Try to get a numeric score from state (preferred if available)
        state_score = try_extract_numeric_score_from_state(result_state)
        # If state_score is None and we have a heuristic_score, use it
        if state_score is None and heuristic_score is not None:
            state_score = heuristic_score
        
        # Extract tool calls and reasoner judgments from state
        tool_calls = extract_tool_calls_from_state(result_state)
        reasoner_judgments = extract_reasoner_judgments_from_state(result_state)
        # reasoning_timeline = extract_reasoning_timeline_from_state(result_state)
        reasoning_timeline = None
        
        # Extract and print token statistics (only if show_token_num is True)
        if show_token_num:
            if isinstance(result_state, dict):
                total_input_tokens = result_state.get("total_input_tokens", 0)
                total_output_tokens = result_state.get("total_output_tokens", 0)
                total_tokens = result_state.get("total_tokens", 0)
                token_count_by_node = result_state.get("token_count_by_node", [])
            else:
                total_input_tokens = getattr(result_state, "total_input_tokens", 0)
                total_output_tokens = getattr(result_state, "total_output_tokens", 0)
                total_tokens = getattr(result_state, "total_tokens", 0)
                token_count_by_node = getattr(result_state, "token_count_by_node", [])
            
            print(f"\n[TOKEN STATS] {image_path}")
            print(f"  Total Input Tokens: {total_input_tokens:,}")
            print(f"  Total Output Tokens: {total_output_tokens:,}")
            print(f"  Total Tokens: {total_tokens:,}")
            if token_count_by_node:
                print(f"  Token breakdown by node:")
                for node_stats in token_count_by_node:
                    node_name = node_stats.get("node", "unknown")
                    node_input = node_stats.get("input_tokens", 0)
                    node_output = node_stats.get("output_tokens", 0)
                    node_total = node_stats.get("total_tokens", 0)
                    print(f"    - {node_name}: {node_total:,} tokens (input: {node_input:,}, output: {node_output:,})")

        return final_result, reason, state_score, tool_calls, reasoner_judgments, reasoning_timeline

    except Exception as e:
        import traceback

        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return "uncertain", f"Exception: {e}", None, [], [], []


def convert_result_to_score(result: str, uncertain_score: float = 0.5) -> float:
    """Map textual result to anomaly score (0-1)."""
    mapping = {
        "anomalous": 1.0,
        "normal": 0.0,
        "uncertain": float(np.clip(uncertain_score, 0.0, 1.0)),
    }
    return mapping.get(result.lower(), float(np.clip(uncertain_score, 0.0, 1.0)))


def evaluate_results(
    results: List[Dict],
    obj_list: List[str],
) -> Dict:
    """Evaluate detection results using WinCLIP-style metrics."""
    # Organize results by class
    class_results = {obj: {"gt": [], "pred": []} for obj in obj_list}

    for result in results:
        cls_name = result["cls_name"]
        if cls_name in class_results:
            class_results[cls_name]["gt"].append(result["gt_anomaly"])
            class_results[cls_name]["pred"].append(result["pred_score"])

    # Calculate metrics per class
    table_ls = []
    auroc_sp_ls = []
    ap_sp_ls = []
    f1_sp_ls = []

    for obj in obj_list:
        if obj not in class_results or not class_results[obj]["gt"]:
            continue

        gt_sp = np.array(class_results[obj]["gt"])
        pr_sp = np.array(class_results[obj]["pred"])

        # AUROC
        # Handle case where only one class is present or all predictions are identical (AUROC undefined)
        try:
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
        except ValueError:
            # Only one class present or all predictions identical, AUROC is undefined
            auroc_sp = np.nan

        # AP
        ap_sp = average_precision_score(gt_sp, pr_sp)

        # F1 (best over thresholds)
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        # # Original robust version (commented for WinCLIP consistency):
        # finite_f1 = f1_scores[np.isfinite(f1_scores)]
        # f1_sp = np.max(finite_f1) if len(finite_f1) > 0 else np.nan
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

        # Format (handle NaN for AUROC)
        auroc_str = "nan" if np.isnan(auroc_sp) else str(np.round(auroc_sp * 100, decimals=1))
        table_ls.append(
            [obj, auroc_str, str(np.round(f1_sp * 100, decimals=1)), str(np.round(ap_sp * 100, decimals=1))]
        )

        auroc_sp_ls.append(auroc_sp)
        ap_sp_ls.append(ap_sp)
        f1_sp_ls.append(f1_sp)

    # Mean row (skip NaN values for AUROC and F1)
    valid_auroc = [x for x in auroc_sp_ls if not np.isnan(x)]
    mean_auroc = np.mean(valid_auroc) if valid_auroc else np.nan
    mean_auroc_str = "nan" if np.isnan(mean_auroc) else str(np.round(mean_auroc * 100, decimals=1))
    valid_f1 = [x for x in f1_sp_ls if not np.isnan(x)]
    mean_f1 = np.mean(valid_f1) if valid_f1 else np.nan
    mean_f1_str = "nan" if np.isnan(mean_f1) else str(np.round(mean_f1 * 100, decimals=1))
    if auroc_sp_ls:  # only append if any class processed
        table_ls.append(
            [
                "mean",
                mean_auroc_str,
                mean_f1_str,
                str(np.round(np.mean(ap_sp_ls) * 100, decimals=1)) if ap_sp_ls else "nan",
            ]
        )

    return {
        "table": table_ls,
        "auroc_sp": auroc_sp_ls,
        "ap_sp": ap_sp_ls,
        "f1_sp": f1_sp_ls,
        "mean_auroc": mean_auroc,  # numeric (may be NaN, filtered)
        "mean_f1": mean_f1,  # numeric (may be NaN, filtered)
        "mean_ap": np.mean(ap_sp_ls) if ap_sp_ls else np.nan,
    }


async def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate agent_v1 on MVTec dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_MVTEC_DATA_PATH,
        help="Path to MVTec dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/agent_v1_eval",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default=DEFAULT_QWEN_VL_MODEL_PATH,
        help="Path to Qwen VL model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, None for auto)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent image processing tasks (default: 1, sequential)",
    )
    parser.add_argument(
        "--uncertain_score",
        type=float,
        default=0.5,
        help="Score assigned to 'uncertain' predictions if no numeric score is available",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum batch size for Qwen VL micro-batching (default: 16)",
    )
    parser.add_argument(
        "--max_wait_ms",
        type=int,
        default=50,
        help="Maximum wait time (milliseconds) before flushing incomplete batches (default: 50)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation for Qwen VL model (default: flash_attention_2). Options: flash_attention_2, sdpa, or None",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=None,
        help="Path to model for embeddings. If not specified, defaults to QWEN_TEXT_MODEL_PATH or ./models/qwen3_4b. If specified and matches qwen_model_path, uses VLM for embeddings.",
    )
    parser.add_argument(
        "--show_token_num",
        action="store_true",
        default=False,
        help="If set, print token statistics for each image (default: False)",
    )
    parser.add_argument(
        "--atomic_candidates_file",
        type=str,
        default=None,
        help="Name of existing atomic candidates file in save_path directory. If None (default), will generate new candidates.",
    )
    parser.add_argument(
        "--counterfactual_top_k",
        type=int,
        default=3,
        help="Top-k for counterfactual atomic candidate matching (default: 3).",
    )
    parser.add_argument(
        "--super_resolution_version",
        type=str,
        default="nongan",
        choices=["gan", "nongan"],
        help="Super-resolution model version. 'gan' uses RealESRGAN (better quality, slower), 'nongan' uses RealESRNet (faster, default). Default: 'nongan'",
    )
    parser.add_argument(
        "--planner_model_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Model type for planner node. 'local' uses local Qwen VL model, 'api' uses API model. Default: 'local'",
    )
    parser.add_argument(
        "--reflector_model_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Model type for reflector node. 'local' uses local Qwen VL model, 'api' uses API model. Default: 'local'",
    )
    parser.add_argument(
        "--reasoner_model_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Model type for reasoner node. 'local' uses local Qwen VL model, 'api' uses API model. Default: 'local'",
    )
    parser.add_argument(
        "--planner_api_model",
        type=str,
        default=None,
        help="API model identifier for planner (e.g., 'anthropic/claude-sonnet-4-5-20250929'). Required when --planner_model_type is 'api'",
    )
    parser.add_argument(
        "--reflector_api_model",
        type=str,
        default=None,
        help="API model identifier for reflector (e.g., 'anthropic/claude-sonnet-4-5-20250929'). Required when --reflector_model_type is 'api'",
    )
    parser.add_argument(
        "--reasoner_api_model",
        type=str,
        default=None,
        help="API model identifier for reasoner (e.g., 'anthropic/claude-sonnet-4-5-20250929'). Required when --reasoner_model_type is 'api'",
    )
    parser.add_argument(
        "--image_description_model_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Model type for image description tasks (e.g., _keyword_heuristic_decision, counterfactual_atomic_candidate_tool caption generation). 'local' uses local Qwen VL model, 'api' uses API model. Default: 'local'",
    )
    parser.add_argument(
        "--image_description_api_model",
        type=str,
        default=None,
        help="API model identifier for image description tasks (e.g., 'anthropic/claude-sonnet-4-5-20250929'). Required when --image_description_model_type is 'api'",
    )
    parser.add_argument(
        "--atomic_candidate_llm_model_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Model type for atomic candidate generation (text-only LLM). 'local' uses local model, 'api' uses API model. Default: 'local'",
    )
    parser.add_argument(
        "--atomic_candidate_llm_api_model",
        type=str,
        default=None,
        help="API model identifier for atomic candidate generation (e.g., 'anthropic/claude-sonnet-4-5-20250929'). Required when --atomic_candidate_llm_model_type is 'api'",
    )
    
    # API generation parameters
    parser.add_argument(
        "--api_temperature",
        type=float,
        default=None,
        help="Temperature for API model generation (0.0-2.0). Lower values for deterministic output, higher for creative output. If not specified, uses model default.",
    )
    parser.add_argument(
        "--api_max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate in API calls. If not specified, uses model default.",
    )
    parser.add_argument(
        "--api_top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter for API calls (0.0-1.0). If not specified, uses model default.",
    )
    parser.add_argument(
        "--api_frequency_penalty",
        type=float,
        default=None,
        help="Frequency penalty for API calls (-2.0 to 2.0). Reduces repetition. If not specified, uses model default.",
    )
    parser.add_argument(
        "--api_presence_penalty",
        type=float,
        default=None,
        help="Presence penalty for API calls (-2.0 to 2.0). Encourages new topics. If not specified, uses model default.",
    )
    parser.add_argument(
        "--api_max_concurrency",
        type=int,
        default=16,
        help="Maximum concurrency for API calls. Default: 16.",
    )
    parser.add_argument(
        "--use_responses_api",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="Whether to enable OpenAI Responses API mode for OpenAI API models. Default: False.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort for OpenAI reasoning models when using Responses API "
             "(e.g., none/low/medium/high). Default: None.",
    )
    parser.add_argument(
        "--tool_call_threshold",
        type=int,
        default=5,
        help="Maximum number of tool calls before forcing a decision. When num_tool_calls exceeds this threshold, the reasoner will skip API calls and use heuristic-based decision. Default: 5.",
    )
    parser.add_argument(
        "--raw_to_reasoner",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=True,
        help="Whether to include the raw (original) image in the reasoner node input. If True, reasoner sees both raw and augmented images. If False, reasoner only sees augmented images. Default: True.",
    )
    parser.add_argument(
        "--raw_to_reflector",
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=True,
        help="Whether to include the raw (original) image in the reflector node input. If True, reflector sees both raw and augmented images. If False, reflector only sees augmented images. Note: If False and no augmented image is available (e.g., planner didn't call tools), falls back to showing raw image. Default: True.",
    )
    parser.add_argument(
        "--normal_tolerance",
        type=int,
        default=2,
        help="Number of consecutive 'normal' judgments required before accepting 'normal' as final result. The reasoner must return 'normal' this many times (after reflection cycles) before the result is accepted. Default: 2.",
    )
    parser.add_argument(
        "--resize",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="Whether to resize all test images to 518x518 before inference. Default: False.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.planner_model_type == "api" and args.planner_api_model is None:
        parser.error("--planner_api_model is required when --planner_model_type is 'api'")
    if args.reflector_model_type == "api" and args.reflector_api_model is None:
        parser.error("--reflector_api_model is required when --reflector_model_type is 'api'")
    if args.reasoner_model_type == "api" and args.reasoner_api_model is None:
        parser.error("--reasoner_api_model is required when --reasoner_model_type is 'api'")
    if args.image_description_model_type == "api" and args.image_description_api_model is None:
        parser.error("--image_description_api_model is required when --image_description_model_type is 'api'")
    if args.atomic_candidate_llm_model_type == "api" and args.atomic_candidate_llm_api_model is None:
        parser.error("--atomic_candidate_llm_api_model is required when --atomic_candidate_llm_model_type is 'api'")

    # Responses API path: do not pass Chat Completions penalty params.
    if (
        args.use_responses_api
        and (
            args.api_frequency_penalty is not None
            or args.api_presence_penalty is not None
        )
    ):
        print(
            "[WARNING] --use_responses_api=True: ignoring "
            "--api_frequency_penalty and --api_presence_penalty."
        )
        args.api_frequency_penalty = None
        args.api_presence_penalty = None

    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # Load test data
    print(f"Loading test data from {args.data_path}...")
    test_data = load_mvtec_test_data(args.data_path)

    if args.max_samples:
        test_data = test_data[: args.max_samples]

    print(f"Loaded {len(test_data)} test samples")
    if args.resize:
        print("Resize preprocessing enabled: all test images will be resized to 518x518.")

    # Dynamic class list from meta.json (fallback to canonical list)
    derived_obj_set = sorted({s.get("cls_name") for s in test_data if "cls_name" in s})
    canonical = [
        "carpet",
        "bottle",
        "hazelnut",
        "leather",
        "cable",
        "capsule",
        "grid",
        "pill",
        "transistor",
        "metal_nut",
        "screw",
        "toothbrush",
        "zipper",
        "tile",
        "wood",
    ]
    obj_list = derived_obj_set if derived_obj_set else canonical
    print(f"Using {len(obj_list)} classes: {obj_list}")

    # Filter samples by class (defensive)
    filtered_samples = [sample for sample in test_data if sample.get("cls_name") in obj_list]

    # Initialize context (must match agent_v1.Context)
    # Set atomic_candidates_file path so counterfactual_atomic_candidate_tool can load from file
    # If user provided --atomic_candidates_file, use that; otherwise use default "atomic_candidates.json"
    # Convert to absolute path to avoid issues with relative paths and current working directory
    if args.atomic_candidates_file is not None:
        atomic_candidates_file = os.path.abspath(os.path.join(args.save_path, args.atomic_candidates_file))
    else:
        atomic_candidates_file = os.path.abspath(os.path.join(args.save_path, "atomic_candidates.json"))
    context = Context(
        qwen_model_path=args.qwen_model_path,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        attn_implementation=args.attn_implementation if args.attn_implementation != "None" else None,
        embedding_model_path=args.embedding_model_path,
        atomic_candidates_file=atomic_candidates_file,
        counterfactual_top_k=args.counterfactual_top_k,
        super_resolution_version=args.super_resolution_version,
        planner_model_type=args.planner_model_type,
        reflector_model_type=args.reflector_model_type,
        reasoner_model_type=args.reasoner_model_type,
        planner_api_model=args.planner_api_model,
        reflector_api_model=args.reflector_api_model,
        reasoner_api_model=args.reasoner_api_model,
        image_description_model_type=args.image_description_model_type,
        image_description_api_model=args.image_description_api_model,
        atomic_candidate_llm_model_type=args.atomic_candidate_llm_model_type,
        atomic_candidate_llm_api_model=args.atomic_candidate_llm_api_model,
        api_temperature=args.api_temperature,
        api_max_tokens=args.api_max_tokens,
        api_top_p=args.api_top_p,
        api_frequency_penalty=args.api_frequency_penalty,
        api_presence_penalty=args.api_presence_penalty,
        use_responses_api=args.use_responses_api,
        reasoning_effort=args.reasoning_effort,
        tool_call_threshold=args.tool_call_threshold,
        raw_to_reasoner=args.raw_to_reasoner,
        raw_to_reflector=args.raw_to_reflector,
        normal_tolerance=args.normal_tolerance,
    )

    # Log embedding model configuration
    from pathlib import Path
    if args.embedding_model_path:
        model_name = Path(args.embedding_model_path).name
        if "vl" in args.embedding_model_path.lower():
            print(f"Using VLM model for embeddings: {model_name}")
        else:
            print(f"Using text-only LLM model for embeddings: {model_name}")
    else:
        print("Using text-only LLM model for embeddings: QWEN_TEXT_MODEL_PATH or ./models/qwen3_4b")

    # Preload models to avoid concurrent loading issues
    print("Preloading models...")
    _ = context.embedding_model  # Preload embedding model
    _ = context.planner_model  # Preload planner model
    _ = context.reflector_model  # Preload reflector model
    _ = context.reasoner_model  # Preload reasoner model
    _ = context.image_description_model  # Preload image description model
    _ = context.atomic_candidate_llm_model  # Preload atomic candidate LLM model
    print("Models preloaded successfully.")
    
    # Generate and save atomic candidates for all classes
    # Note: atomic_candidates_file path is already set in context above
    if args.atomic_candidates_file is None:
        # Generate new candidates
        print(f"\nGenerating atomic candidates for {len(obj_list)} classes...")
        try:
            await generate_and_save_atomic_candidates(
                object_names=obj_list,
                context=context,
                output_file=atomic_candidates_file,
            )
            print(f"Atomic candidates saved to: {atomic_candidates_file}")
        except Exception as e:
            print(f"Warning: Failed to generate atomic candidates: {e}")
            print("Continuing with evaluation...")
    else:
        # Use existing file
        print(f"\nUsing existing atomic candidates file: {args.atomic_candidates_file}")
        try:
            await generate_and_save_atomic_candidates(
                object_names=obj_list,
                context=context,
                output_file=atomic_candidates_file,
                file_name=args.atomic_candidates_file,  # Check if file exists in save_path directory
            )
            # Note: The function will print whether it used existing file or generated new one
        except Exception as e:
            print(f"Warning: Failed to load atomic candidates from {args.atomic_candidates_file}: {e}")
            print("Continuing with evaluation...")
    
    # Process images
    print(f"\nRunning agent on test images (concurrent: {args.concurrent})...")

    # Helper to compute pred_score
    def compute_pred_score(result_label: str, reason: str, state_score: Optional[float]) -> float:
        if state_score is not None:
            return float(np.clip(state_score, 0.0, 1.0))

        # Check if this is a heuristic-based decision by looking for specific markers
        # (not just the word "heuristic" which may appear in normal reasoning)
        reason_lower = (reason or "").lower()
        is_heuristic_decision = (
            "forced decision" in reason_lower or 
            "heuristic-based decision" in reason_lower or
            "[reasoner:forced" in reason_lower
        )
        
        if is_heuristic_decision:
            if result_label == "anomalous":
                return 0.7
            if result_label == "normal":
                return 0.3
            # uncertain
            return float(np.clip(args.uncertain_score, 0.0, 1.0))

        # default mapping
        return convert_result_to_score(result_label, args.uncertain_score)

    if args.concurrent == 1:
        # Sequential processing
        print("Processing images sequentially...")
        results = []
        for sample in tqdm(filtered_samples, desc="Processing images"):
            result, reason, state_score, tool_calls, reasoner_judgments, reasoning_timeline = await run_agent_on_image(
                graph,
                sample["full_img_path"],
                sample["cls_name"],
                context,
                show_token_num=args.show_token_num,
                resize=args.resize,
            )
            pred_score = compute_pred_score(result, reason, state_score)
            results.append(
                {
                    "cls_name": sample["cls_name"],
                    "img_path": sample["img_path"],
                    "gt_anomaly": sample["anomaly"],
                    "pred_result": result,
                    "pred_score": pred_score,
                    "reason": reason,
                    "tool_calls": tool_calls,
                    "reasoner_judgments": reasoner_judgments,
                    # "reasoning_timeline": reasoning_timeline,
                }
            )
    else:
        # Parallel processing with semaphore to limit concurrency
        print(f"Processing images in parallel (max {args.concurrent} concurrent)...")
        semaphore = asyncio.Semaphore(args.concurrent)

        async def process_sample_with_semaphore(sample):
            async with semaphore:
                result, reason, state_score, tool_calls, reasoner_judgments, reasoning_timeline = await run_agent_on_image(
                    graph,
                    sample["full_img_path"],
                    sample["cls_name"],
                    context,
                    show_token_num=args.show_token_num,
                    resize=args.resize,
                )
                pred_score = compute_pred_score(result, reason, state_score)
                return {
                    "cls_name": sample["cls_name"],
                    "img_path": sample["img_path"],
                    "gt_anomaly": sample["anomaly"],
                    "pred_result": result,
                    "pred_score": pred_score,
                    "reason": reason,
                    "tool_calls": tool_calls,
                    "reasoner_judgments": reasoner_judgments,
                    # "reasoning_timeline": reasoning_timeline,
                }

        tasks = [process_sample_with_semaphore(sample) for sample in filtered_samples]

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"):
            result_item = await coro
            results.append(result_item)

    # Evaluate
    print("Evaluating results...")
    eval_results = evaluate_results(results, obj_list)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(
        tabulate(
            eval_results["table"],
            headers=["Object", "AUROC-SP", "F1-SP", "AP-SP"],
            tablefmt="pipe",
        )
    )

    # Mean metrics (already filtered in evaluate_results)
    mean_auroc_sp = float(eval_results["mean_auroc"]) if not np.isnan(eval_results["mean_auroc"]) else float("nan")
    mean_f1_sp = float(eval_results["mean_f1"]) if not np.isnan(eval_results["mean_f1"]) else float("nan")
    mean_ap_sp = float(eval_results["mean_ap"]) if not np.isnan(eval_results["mean_ap"]) else float("nan")

    # Save results
    results_path = os.path.join(args.save_path, "results.json")
    with open(results_path, "w") as f:
        # Convert args to dictionary, filtering out None values for cleaner output
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        json.dump(
            {
                "args": args_dict,
                "classes": obj_list,
                "results": results,
                "metrics": {
                    "mean_auroc_sp": mean_auroc_sp,
                    "mean_f1_sp": mean_f1_sp,
                    "mean_ap_sp": mean_ap_sp,
                },
                "table": eval_results["table"],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_path}")
    await context.aclose()


if __name__ == "__main__":
    asyncio.run(main())

# toy test: python evaluate_mvtec_v12.py --data_path ./data/mvtec --save_path ./results/test --max_samples 10 --qwen_model_path ./models/qwen3_vl_4b --device cuda --concurrent 2 --uncertain_score 0.5 --max_batch_size 4 --max_wait_ms 50 --show_token_num
# toy test: python evaluate_mvtec_v12.py --data_path ./data/mvtec --save_path ./results/test --max_samples 10 --qwen_model_path ./models/qwen3_vl_4b --device cuda --concurrent 2 --uncertain_score 0.5 --max_batch_size 4 --max_wait_ms 50 --show_token_num --atomic_candidates_file atomic_candidates.json
