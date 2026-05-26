"""Define the state structures for the agent."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Optional, Sequence, Annotated, Dict, Any

import torch
from torch import Tensor
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


def merge_tool_calls_list(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge two lists of tool calls by concatenating them."""
    if not isinstance(existing, list):
        existing = []
    if not isinstance(new, list):
        new = []
    return existing + new


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """

    class_name: Optional[str] = None
    """The class name of the sample (e.g., 'bottle', 'carpet', etc.)."""


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    sample_id: Optional[int] = None
    feature_map: Optional[Tensor] = None
    feature_map_aug: Optional[Tensor] = None
    is_train: bool = False
    y: Optional[int] = None
    seg: Optional[Tensor] = None
    potential_anomalies: Annotated[list[str], operator.add] = field(default_factory=list)
    heuristic_prompt: Annotated[list[str], operator.add] = field(default_factory=list)
    pred_y: Optional[int] = None
    reasoning_y: list[str] = field(default_factory=list)
    pred_seg: Optional[Tensor] = None
    reasoning_seg: list[str] = field(default_factory=list)
    num_tool_calls: Annotated[int, operator.add] = 0
    tool_call_threshold: int = 5
    executed_tool_calls: Annotated[list[dict], merge_tool_calls_list] = field(default_factory=list)
    """List of executed tool calls, each containing tool_name, tool_call_id, args, and node."""
    reasoner_call_count: Annotated[int, operator.add] = 0
    """Number of times the reasoner has been called."""
    reasoner_judgments: Annotated[list[dict], merge_tool_calls_list] = field(default_factory=list)
    """Ordered list of reasoner judgments, each containing at least 'call_index', 'result', and 'reason'."""
    result: Optional[str] = None
    """The final result of the agent's reasoning: 'anomalous', 'normal', or 'uncertain'."""
    reason: Optional[str] = None
    """The reason or explanation for the result."""
    
    # Token counting fields
    total_input_tokens: Annotated[int, operator.add] = 0
    """Total input tokens consumed across all model calls."""
    total_output_tokens: Annotated[int, operator.add] = 0
    """Total output tokens generated across all model calls."""
    total_tokens: Annotated[int, operator.add] = 0
    """Total tokens (input + output) consumed across all model calls."""
    token_count_by_node: Annotated[list[dict], merge_tool_calls_list] = field(default_factory=list)
    """List of token counts per node call, each containing 'node', 'input_tokens', 'output_tokens', 'total_tokens'."""
    
    # Raw image field (single source of truth for original image)
    raw_image_url: Optional[str] = None
    """Normalized data URL or file URL of the original/raw image from user input."""
    
    # Latest augmented image fields (single source of truth)
    latest_augmented_image_url: Optional[str] = None
    """Normalized data URL or file URL of the latest processed/augmented image from image processing tools."""
    latest_augmented_image_hash: Optional[str] = None
    """Hash of the latest augmented image content (for debug/deduplication/consistency)."""
    
    last_tool_bypassed: bool = False
    """Whether the last image processing tool was bypassed (returned unchanged image, e.g., scale=1)."""

    tool_failed: bool = False
    """Whether the latest tool execution produced an error-like response."""

    reflector_image_target: Optional[str] = None
    """Requested image target for reflector tool calls: 'raw' or 'augmented'."""
    
    last_caller: Optional[str] = None
    """Last node that called tools: 'planner' or 'reflector'. Used by tools_node to identify caller."""
    
    # General template analysis from keyword heuristic decision
    general_template_analysis: Optional[str] = None
    """General template analysis report from keyword heuristic decision in prior_tools node."""
    
    # Counterfactual template analysis from counterfactual_atomic_candidate_tool node
    counterfactual_template_analysis: Optional[str] = None
    """Counterfactual template analysis report from counterfactual_atomic_candidate_tool node."""
    
    # Statistics from counterfactual_atomic_candidate_tool (for printing)
    counterfactual_stats: Optional[Dict[str, Any]] = None
    """Statistics from counterfactual_atomic_candidate_tool: avg_margin, avg_evidence_strength, etc."""
    
    # Image captions from VLM (generated by keyword_heuristic_decision)
    image_captions: Optional[list[str]] = None
    """List of 3 perspective captions generated by VLM for the current image."""

    def __post_init__(self):
        # protection function
        if self.is_train:
            if self.y is None:
                raise ValueError("State: is_train=True requires non-None `y` (classification label).")
            if self.seg is None:
                raise ValueError("State: is_train=True requires non-None `seg` (segmentation label tensor).")
        else:
            if self.y is not None or self.seg is not None:
                self.y = None
                self.seg = None

        if self.feature_map is not None and not isinstance(self.feature_map, torch.Tensor):
            raise TypeError("`feature_map` must be a torch.Tensor or None.")
        if self.seg is not None and not isinstance(self.seg, torch.Tensor):
            raise TypeError("`seg` (seg label) must be a torch.Tensor or None.")
        if self.pred_seg is not None and not isinstance(self.pred_seg, torch.Tensor):
            raise TypeError("`pred_seg` must be a torch.Tensor or None.")

        def _to_str_list(x) -> list[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
                return [str(v) for v in x]
            raise TypeError("Expected str or Sequence[str].")

        self.potential_anomalies = _to_str_list(self.potential_anomalies)
        self.heuristic_prompt    = _to_str_list(self.heuristic_prompt)
        self.reasoning_y         = _to_str_list(self.reasoning_y)
        self.reasoning_seg       = _to_str_list(self.reasoning_seg)

    def set_supervision(self, y: int, seg: Tensor) -> None:
        if not isinstance(seg, torch.Tensor):
            raise TypeError("`seg` must be a torch.Tensor.")
        self.is_train = True
        self.y = int(y)
        self.seg = seg

    def set_predictions(
        self,
        pred_y: Optional[int] = None,
        pred_seg: Optional[Tensor] = None,
        reasoning_y: Optional[Sequence[str] | str] = None,
        reasoning_seg: Optional[Sequence[str] | str] = None,
        *,
        append: bool = True,
    ) -> None:

        if pred_y is not None:
            self.pred_y = int(pred_y)
        if pred_seg is not None:
            if not isinstance(pred_seg, torch.Tensor):
                raise TypeError("`pred_seg` must be a torch.Tensor.")
            self.pred_seg = pred_seg

        def _to_str_list(x) -> list[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
                return [str(v) for v in x]
            raise TypeError("Expected str or Sequence[str].")

        ry = _to_str_list(reasoning_y)
        rs = _to_str_list(reasoning_seg)

        if ry:
            if append:
                self.reasoning_y.extend(ry)
            else:
                self.reasoning_y = ry
        if rs:
            if append:
                self.reasoning_seg.extend(rs)
            else:
                self.reasoning_seg = rs