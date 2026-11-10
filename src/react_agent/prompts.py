"""Default prompts used by the agent for image anomaly detection."""

IMAGE_SOURCE_INSTRUCTION_TEXT = (
    "IMPORTANT: raw_to_reasoner is true and the latest augmented image differs from the raw image. "
    "You MUST consider BOTH the latest augmented image and the raw image together when making the judgment, "
    "and reconcile any differences explicitly."
)

TOOL_TARGET_INSTRUCTION_TEXT = (
    "IMPORTANT: raw_to_reflector is true and the latest augmented image differs from the raw image. "
    "If you decide to call a tool, you MUST explicitly choose whether it should operate on the "
    "latest augmented image or the raw image, and state that choice in your response. "
    "Do NOT generate or copy image_base64 yourself. Set image_target to 'raw' or 'augmented', and the "
    "system will inject the correct image automatically. "
    "When you call a tool, include an explicit 'image_target' field in your JSON with value 'raw' or 'augmented'."
)

TOOL_FAILURE_INSTRUCTION_TEXT = (
    "If image processing failed, do NOT infer anomalies from it."
)

from typing import List

SYSTEM_PROMPT = """You are an expert AI assistant specialized in image anomaly detection.

Your task is to analyze images and detect anomalies. You use reasoning approaches based on visual feature analysis, pattern recognition, and comparison with normal appearance patterns. You are also encouraged to use tools to help you analyze the image.

Key capabilities:
- Visual feature extraction and analysis
- Pattern recognition and deviation detection
- Anomaly classification and anomaly segmentation
- Detailed reasoning about visual anomalies
- Use tools to help you analyze the image
"""
## System time: {system_time}"""


PROMPT4PLANNER = """Your task is to analyze the provided image(s) and plan the anomaly-detection workflow.

You have access to a general template analysis from keyword heuristic decision (if available), which provides semantic evidence signals based on caption-prototype comparison. Use this information to inform your planning, but do not rely solely on it.

You MUST:
1. Examine the image for potential anomalies (color, texture, shape, patterns, structure).
2. Choose ONE image-processing tool based on the actual IMAGE CONDITION.
3. Output a JSON object containing: "action", "args", "potential_anomalies", "heuristic_prompt".

IMPORTANT:
- Tool choice MUST be based on FUNCTION, NOT list order. The tools are listed in arbitrary order.
- You may call tools WITHOUT providing image_base64; the image will be extracted automatically.
- You MUST call at least one tool.

Available tools (arbitrary order):
- image_denoising: reduce noise or grain
- image_deblurring: sharpen blurry or out-of-focus images
- image_super_resolution: enhance fine-grained details
- image_zooming: magnify specific regions
- image_brightness_enhancement: enhance brightness for dimly lit or overexposed images using CLAHE

Decision Guide:
- If noisy → image_denoising  
- If blurry → image_deblurring  
- If low-resolution → image_super_resolution  
- If specific region needs close inspection → image_zooming  
- If dimly lit or overexposed → image_brightness_enhancement
- If image looks already clear → use super-resolution or zoom for finer detail analysis  

Your response MUST be a JSON object of the form:
{
  "action": "tool_name",
  "args": {},
  "potential_anomalies": "Describe suspected or observed anomalies, with locations and visual characteristics. If none, describe normal appearance.",
  "heuristic_prompt": "Guidance for further analysis: regions to re-inspect, features to verify, or next-step strategy."
}

Return ONLY the JSON object.
"""

PROMPT4PLANNER_API = """Your task is to analyze the provided image(s) and plan the anomaly-detection workflow.

You have access to a general template analysis from keyword heuristic decision (if available), which provides semantic evidence signals based on caption-prototype comparison. Use this information to inform your planning, but do not rely solely on it.

You MUST:
1. Examine the image for potential anomalies (color, texture, shape, patterns, structure).
2. Decide whether additional image processing is needed.
3. Either:
   - Call exactly ONE image-processing tool using the tool calling mechanism, OR
   - Skip tool calling if you have already identified obvious anomalies.

IMPORTANT (CRITICAL PROTOCOL — MUST FOLLOW EXACTLY):

You MUST ALWAYS provide a JSON object in the assistant message content with the following structure:
{
  "decision": "call_tool" or "skip_tool",
  "potential_anomalies": "Describe suspected or observed anomalies, with locations and visual characteristics. If none, describe normal appearance.",
  "heuristic_prompt": "Guidance for further analysis: regions to re-inspect, features to verify, or next-step strategy."
}

This JSON is ALWAYS required, regardless of whether you call a tool or skip tool calling.

Tool calling rules:
- If you choose "decision": "call_tool":
  - Use the tool calling mechanism to invoke the selected tool.
  - You MUST include the required JSON in the SAME assistant message content (even if it's very short, just the JSON).
  - The JSON can be in the same message that contains the tool_calls.
- If you choose "decision": "skip_tool":
  - Do NOT call any tools.
  - Output ONLY the JSON object in your response content.

Additional rules:
- Tool choice MUST be based on FUNCTION, NOT list order.
- You may call tools WITHOUT providing image_base64; the image will be extracted automatically.
- Do NOT mention tool names if you skip tool calling.

Available tools (arbitrary order):
- image_denoising: reduce noise or grain
- image_deblurring: sharpen blurry or out-of-focus images
- image_super_resolution: enhance fine-grained details
- image_zooming: magnify specific regions
- image_brightness_enhancement: enhance brightness for dimly lit or overexposed images using CLAHE

Decision Guide:
- If noisy → image_denoising
- If blurry → image_deblurring
- If low-resolution → image_super_resolution
- If specific region needs close inspection → image_zooming
- If dimly lit or overexposed → image_brightness_enhancement
- If image looks already clear → use super-resolution or zoom for finer detail analysis
"""

PROMPT4PLANNER_LOCAL = """Your task is to analyze the provided image(s) and plan the anomaly-detection workflow.

You have access to a general template analysis from keyword heuristic decision (if available), which provides semantic evidence signals based on caption-prototype comparison. Use this information to inform your planning, but do not rely solely on it.

You MUST:
1. Examine the image for potential anomalies (color, texture, shape, patterns, structure).
2. Choose ONE image-processing tool based on the actual IMAGE CONDITION.
3. Output a JSON object containing: "action", "args", "potential_anomalies", "heuristic_prompt".

IMPORTANT:
- Tool choice MUST be based on FUNCTION, NOT list order. The tools are listed in arbitrary order.
- You may call tools WITHOUT providing image_base64; the image will be extracted automatically.
- You MUST call at least one tool.

Available tools (arbitrary order):
- image_denoising: reduce noise or grain
- image_deblurring: sharpen blurry or out-of-focus images
- image_super_resolution: enhance fine-grained details
- image_zooming: magnify specific regions
- image_brightness_enhancement: enhance brightness for dimly lit or overexposed images using CLAHE

Decision Guide:
- If noisy → image_denoising  
- If blurry → image_deblurring  
- If low-resolution → image_super_resolution  
- If specific region needs close inspection → image_zooming  
- If dimly lit or overexposed → image_brightness_enhancement
- If image looks already clear → use super-resolution or zoom for finer detail analysis  

Your response MUST be a JSON object of the form:
{
  "action": "tool_name",
  "args": {},
  "potential_anomalies": "Describe suspected or observed anomalies, with locations and visual characteristics. If none, describe normal appearance.",
  "heuristic_prompt": "Guidance for further analysis: regions to re-inspect, features to verify, or next-step strategy."
}

Return ONLY the JSON object.
"""

PROMPT4REASONER = """Your task is to make a FINAL judgment about whether the image contains anomalies.  

Use ALL available information, including:
- General template analysis from keyword heuristic decision (semantic evidence signals based on caption-prototype comparison)
- Counterfactual template analysis. This may contain multiple reports:
  - soft textual_matching reports from caption-to-candidate embedding matching
  - hard visual_check reports from prepared logical rule checks
- Potential anomalies identified in earlier stages
- Heuristic prompts
- Any processed/enhanced image results
- Your own analysis based on the current image

Hard-rule handling:
- A hard visual_check report is a rule-verification signal, not a generic similarity score.
- If a hard visual_check report has hard_violation=true after framework thresholding, treat it as a strong anomaly signal.
- Individual fail checks that do not meet the configured hard-violation threshold are warning signals, not decisive anomaly evidence by themselves.
- Unknown hard checks are not normal evidence; they mean the rule could not be verified from the image.
- Soft textual_matching reports remain advisory and should not alone force a decision.

Conservative decision policy for logical anomaly detection:
- Normal samples may contain viewpoint, illumination, scale, texture, and arrangement variations. Do not classify these as anomalous unless they create a concrete rule violation or visible structural defect.
- Use "anomalous" only when there is specific, localized, visually verifiable evidence such as missing/extra parts, broken/misaligned structures, invalid connections, wrong relative positions, or a hard visual_check report with hard_violation=true.
- If the evidence is only a weak suspicion, a single advisory soft score, an uncertain/unknown check, or a vague appearance difference, prefer "uncertain" over "anomalous".
- Use "normal" when the image is broadly consistent with the expected object/rule and no concrete violation is visible.

Respond with a JSON object containing exactly two fields:
{
    "result": "One of: 'anomalous', 'normal', or 'uncertain'.  
               Use 'anomalous' if you judge that anomalies exist.  
               Use 'normal' if the image appears consistent with expected appearance.  
               Use 'uncertain' if the evidence is mixed, ambiguous, or insufficient.",
    "reason": "A detailed explanation of your reasoning, including:  
               (1) what specific issues were found or why the image seems normal,  
               (2) how previous findings influenced your judgment,  
               (3) your confidence level,  
               (4) if uncertain, what information is still missing."
}

Guidelines:
- Be specific about which visual or analytical factors led to your conclusion.
- Integrate past findings with your current understanding.
- Only output the JSON object, with no extra text.
- <<IMAGE_SOURCE_INSTRUCTION>>
- <<TOOL_FAILURE_INSTRUCTION>>
"""

PROMPT4REASONER_FINAL = """Your task is to make a FINAL decision about whether the image contains anomalies. This is your LAST chance to make a judgment - you MUST provide a definitive answer.

Use ALL available information, including:
- General template analysis from keyword heuristic decision (semantic evidence signals based on caption-prototype comparison)
- Counterfactual template analysis. This may contain multiple reports:
  - soft textual_matching reports from caption-to-candidate embedding matching
  - hard visual_check reports from prepared logical rule checks
- Potential anomalies identified in earlier stages
- Heuristic prompts
- Any processed/enhanced image results
- Your own analysis based on the current image

Hard-rule handling:
- A hard visual_check report is a rule-verification signal, not a generic similarity score.
- If a hard visual_check report has hard_violation=true after framework thresholding, treat it as a strong anomaly signal.
- Individual fail checks that do not meet the configured hard-violation threshold are warning signals, not decisive anomaly evidence by themselves.
- Unknown hard checks are not normal evidence; they mean the rule could not be verified from the image.
- Soft textual_matching reports remain advisory and should not alone force a decision.

Conservative decision policy for logical anomaly detection:
- Normal samples may contain viewpoint, illumination, scale, texture, and arrangement variations. Do not classify these as anomalous unless they create a concrete rule violation or visible structural defect.
- Use "anomalous" only when there is specific, localized, visually verifiable evidence such as missing/extra parts, broken/misaligned structures, invalid connections, wrong relative positions, or a hard visual_check report with hard_violation=true.
- If the evidence is only a weak suspicion, a single advisory soft score, an uncertain/unknown check, or a vague appearance difference, choose "normal" for this forced final decision.
- Use "normal" when the image is broadly consistent with the expected object/rule and no concrete violation is visible.

Respond with a JSON object containing exactly two fields:
{
    "result": "One of: 'anomalous' or 'normal' ONLY.  
               Use 'anomalous' if you judge that anomalies exist.  
               Use 'normal' if the image appears consistent with expected appearance.  
               IMPORTANT: You CANNOT output 'uncertain' - you must make a definitive decision based on the available evidence.",
    "reason": "A detailed explanation of your reasoning, including:  
               (1) what specific issues were found or why the image seems normal,  
               (2) how previous findings influenced your judgment,  
               (3) your confidence level and why you made this definitive decision."
}

Guidelines:
- Be specific about which visual or analytical factors led to your conclusion.
- Integrate past findings with your current understanding.
- You MUST choose either 'anomalous' or 'normal' - no uncertainty allowed.
- Only output the JSON object, with no extra text.
- <<IMAGE_SOURCE_INSTRUCTION>>
- <<TOOL_FAILURE_INSTRUCTION>>
"""


PROMPT4REFLECTOR = """Your task is to RE-ANALYZE the case to verify or refine the previous conclusion.  

Use ALL available information:
- General template analysis from keyword heuristic decision (semantic evidence signals based on caption-prototype comparison)
- Counterfactual template analysis from atomic candidate matching (evidence report based on similarity matching between image captions and atomic candidates)
- Previous potential anomalies
- The heuristic prompt
- Which tools were used previously
- Any processed or enhanced visual results

You MUST:
1. Choose ONE tool that was NOT used previously.  
   (If all tools were used, you may reuse one but with a different purpose or region of focus.)
2. Provide refined potential anomalies and a refined heuristic prompt.
3. Output a JSON object containing: "action", "args", "potential_anomalies", "heuristic_prompt".

<<TOOL_TARGET_INSTRUCTION>>

IMPORTANT:
- Tool selection MUST be based on FUNCTION, NOT list order.
- You MUST call at least one tool.
- The goal is to gather NEW information from a DIFFERENT angle.

Available tools:
- image_denoising  
- image_deblurring  
- image_super_resolution  
- image_zooming
- image_brightness_enhancement  

Re-analysis goals:
- If previous result was "normal": verify that no subtle or overlooked anomalies exist.
- If previous result was "uncertain": focus on resolving ambiguity.
- If previous result contained potential anomalies: refine, validate, or challenge them.

Your JSON response MUST follow:
{
  "action": "tool_name",
  "args": {},
  "image_target": "raw" or "augmented",
  "potential_anomalies": "Refined/expanded findings from re-analysis. Include any subtle issues, confirmations, or contradictions.",
  "heuristic_prompt": "Guidance for the next step: what remains unclear, what to verify, or what information is still needed."
}

Return ONLY the JSON object.
"""

PROMPT4REFLECTOR_API = """Your task is to RE-ANALYZE the case to verify or refine the previous conclusion.

Use ALL available information:
- General template analysis from keyword heuristic decision (semantic evidence signals based on caption-prototype comparison)
- Counterfactual template analysis from atomic candidate matching (evidence report based on similarity matching)
- Previous potential anomalies
- The heuristic prompt
- Which tools were used previously
- Any processed or enhanced visual results

You MUST:
1. Decide whether new information is needed.
2. Either:
   - Call exactly ONE image-processing tool (prefer a tool NOT used previously; if all tools were used, you may reuse one with a different focus), OR
   - Skip tool calling if the existing evidence is sufficient.
3. Provide refined potential anomalies and a refined heuristic prompt.

IMPORTANT (CRITICAL PROTOCOL — MUST FOLLOW EXACTLY):

You MUST ALWAYS provide a JSON object in the assistant message content with the following structure:
{
  "decision": "call_tool" or "skip_tool",
  "image_target": "raw" or "augmented",
  "potential_anomalies": "Refined description of suspected or observed anomalies, with locations and visual characteristics.",
  "heuristic_prompt": "Refined guidance for further analysis: regions to re-inspect, features to verify, or next-step strategy."
}

This JSON is ALWAYS required, regardless of whether you call a tool or skip tool calling.

Tool calling rules:
- If you choose "decision": "call_tool":
  - Use the tool calling mechanism to invoke the selected tool.
  - You MUST include the required JSON in the SAME assistant message content (even if it's very short, just the JSON).
  - The JSON can be in the same message that contains the tool_calls.
- If you choose "decision": "skip_tool":
  - Do NOT call any tools.
  - Do NOT mention any tool names.
  - Output ONLY the JSON object in your response content.

<<TOOL_TARGET_INSTRUCTION>>

Additional rules:
- Tool selection MUST be based on FUNCTION, NOT list order.
- The goal is to gather NEW information from a DIFFERENT angle.
- You may call tools WITHOUT providing image_base64; the image will be extracted automatically.

Available tools:
- image_denoising: reduce noise or grain
- image_deblurring: sharpen blurry or out-of-focus images
- image_super_resolution: enhance fine-grained details
- image_zooming: magnify specific regions
- image_brightness_enhancement: enhance brightness for dimly lit or overexposed images using CLAHE

Re-analysis goals:
- If previous result was "normal": verify that no subtle or overlooked anomalies exist.
- If previous result was "uncertain": focus on resolving ambiguity.
- If previous result contained potential anomalies: refine, validate, or challenge them.
"""

PROMPT4REFLECTOR_LOCAL = """Your task is to RE-ANALYZE the case to verify or refine the previous conclusion.  

Use ALL available information:
- General template analysis from keyword heuristic decision (semantic evidence signals based on caption-prototype comparison)
- Counterfactual template analysis from atomic candidate matching (evidence report based on similarity matching between image captions and atomic candidates)
- Previous potential anomalies
- The heuristic prompt
- Which tools were used previously
- Any processed or enhanced visual results

You MUST:
1. Choose ONE tool that was NOT used previously.  
   (If all tools were used, you may reuse one but with a different purpose or region of focus.)
2. Provide refined potential anomalies and a refined heuristic prompt.
3. Output a JSON object containing: "action", "args", "potential_anomalies", "heuristic_prompt".

<<TOOL_TARGET_INSTRUCTION>>

IMPORTANT:
- Tool selection MUST be based on FUNCTION, NOT list order.
- You MUST call at least one tool.
- The goal is to gather NEW information from a DIFFERENT angle.

Available tools:
- image_denoising  
- image_deblurring  
- image_super_resolution  
- image_zooming
- image_brightness_enhancement  

Re-analysis goals:
- If previous result was "normal": verify that no subtle or overlooked anomalies exist.
- If previous result was "uncertain": focus on resolving ambiguity.
- If previous result contained potential anomalies: refine, validate, or challenge them.

Your JSON response MUST follow:
{
  "action": "tool_name",
  "args": {},
  "image_target": "raw" or "augmented",
  "potential_anomalies": "Refined/expanded findings from re-analysis. Include any subtle issues, confirmations, or contradictions.",
  "heuristic_prompt": "Guidance for the next step: what remains unclear, what to verify, or what information is still needed."
}

Return ONLY the JSON object.
"""


SUMMARY_PROMPT = """Summarize the image anomaly detection process and results into a concise memory.

Include:
1. The image(s) analyzed
2. Key anomalies detected (if any) or confirmation of normal appearance
3. The final classification result (anomalous/normal)
4. Core visual features or patterns that led to the conclusion
5. Any important context or reasoning

Keep the summary focused on:
- The detection outcome and confidence level
- Critical visual evidence
- Any tools used to help you analyze the image
- Key decision factors

Return accurate and concise plain text, less than 5 sentences."""


PROMPT4DESCRIPTION = """You are performing visual condition inspection for anomaly detection.

Your task is to generate MULTIPLE concise descriptions of the object's visual condition from DIFFERENT PERSPECTIVES.

Carefully examine the object in the image and produce EXACTLY THREE short captions, each focusing on a different aspect:

Perspective 1 (Overall Condition):
- Describe whether the object appears intact, normal, clean, or obviously damaged.

Perspective 2 (Surface & Texture):
- Describe surface-level details such as scratches, cracks, dents, contamination,missing parts, deformation, or unusual texture.

Perspective 3 (Anomaly-Oriented Judgment):
- Describe whether there are any visual signs that could indicate defects, damage, or abnormality, even if subtle.

Guidelines:
- Each caption MUST be a single short English sentence.
- Use objective visual observations only.
- Do NOT explain reasoning.
- Do NOT speculate beyond what is visible.
- Do NOT repeat the same wording across captions.

Your response MUST strictly follow this JSON format:

{
  "captions": [
    "Caption from Perspective 1",
    "Caption from Perspective 2",
    "Caption from Perspective 3"
  ]
}

Return ONLY the JSON object, with no additional text.
"""


PROMPT4CANDIDATE_GENERATION = """You are generating atomic candidate descriptions for anomaly detection.

Given a class name (e.g., "bottle"), generate two sets of atomic candidates in strict JSON format:

1. anomaly_candidates: 10-12 short phrases or sentences describing anomalies such as:
   - Surface issues (scratches, cracks, dents, stains)
   - Deformations (bent, warped, misshapen)
   - Missing parts (broken, incomplete)
   - Contamination (dirty, discolored, foreign objects)
   - Label issues (misaligned, damaged labels)

2. normal_candidates: The same quantity as anomaly_candidates, describing the object in complete and intact state.

Guidelines:
- Each candidate MUST be a short phrase or sentence (1-10 words).
- Candidates should be specific and concrete.
- Do NOT repeat similar descriptions.
- anomaly_candidates should focus on defect/anomaly characteristics.
- normal_candidates should focus on intact/complete characteristics.

Constraints:
- Return candidates grouped to cover: surface, shape, parts, contamination, label/printing (at least 2 each).
- If a category is not applicable for this class (e.g., carpet/grid/tile may have no label/parts), replace it with other plausible defect types, but keep total count unchanged.
- Do NOT use generic words like 'defective', 'damaged', 'abnormal' alone; always be specific (e.g., 'cracked surface', 'missing cap').
- Do NOT treat synonyms as different candidates (e.g., 'scratch' vs 'scratched surface' are redundant); avoid pseudo-diversity.

Your response MUST strictly follow this JSON format:

{
  "anomaly_candidates": [
    "candidate 1",
    "candidate 2",
    ...
  ],
  "normal_candidates": [
    "candidate 1",
    "candidate 2",
    ...
  ]
}

Return ONLY the JSON object, with no additional text.
"""


TOOL_USE_MEMORY_TEMPLATE = """The {num} most similar images have been retrieved from memory:
- Image A has a similarity of {similarity} to the current image and used tool {tool}.
- Image B has a similarity of {similarity} to the current image and used tool {tool}.
- ...
"""

IMAGE_LABEL_MEMORY_TEMPLATE = """The {num} most similar images have been retrieved from memory:
- Image A has a similarity of {similarity} to the current image, and its ground-truth label is {label}.
- Image B has a similarity of {similarity} to the current image, and its ground-truth label is {label}.
- ...
"""


def format_keyword_heuristic_report(
    tool_name: str,
    class_name: str,
    escaped_captions: List[str],
    score1_s: str,
    score2_s: str,
    score3_s: str,
    margin1_s: str,
    margin2_s: str,
    margin3_s: str,
    caption_status: List[str],
    agreement_pattern: str,
    top_perspectives: str,
    score_spread: float,
    margin_spread: float,
    uncertainty_note: str,
    status_text: str,
) -> str:
    """Format the keyword heuristic anomaly evidence report.
    
    Args:
        tool_name: Name of the tool (e.g., "keyword_heuristic")
        class_name: Name of the object class
        escaped_captions: List of 3 escaped caption strings
        score1_s, score2_s, score3_s: Formatted score strings
        margin1_s, margin2_s, margin3_s: Formatted margin strings
        caption_status: List of 3 status strings ("valid", "invalid", "missing")
        agreement_pattern: Agreement pattern description
        top_perspectives: Top perspectives description
        score_spread: Score spread value
        margin_spread: Margin spread value
        uncertainty_note: Uncertainty note description
        status_text: Status summary text
    
    Returns:
        Formatted report string
    """
    return f"""[{tool_name}] Anomaly evidence report (caption-prototype comparison) for class="{class_name}":

Method:

- For each caption, we compared its embedding against two text prototypes:

  NORMAL prototype vs ANOMALOUS prototype.

- anomaly_score in [0,1]: higher means closer to the ANOMALOUS prototype.

- margin = (similarity to anomalous prototype) - (similarity to normal prototype).

Evidence by perspective:

1) Perspective 1 - Overall condition

   - caption: {escaped_captions[0]}

   - anomaly_score: {score1_s}

   - margin (anom - norm): {margin1_s}

   - status: {caption_status[0]}

2) Perspective 2 - Surface & texture

   - caption: {escaped_captions[1]}

   - anomaly_score: {score2_s}

   - margin (anom - norm): {margin2_s}

   - status: {caption_status[1]}

3) Perspective 3 - Anomaly-oriented judgment

   - caption: {escaped_captions[2]}

   - anomaly_score: {score3_s}

   - margin (anom - norm): {margin3_s}

   - status: {caption_status[2]}

Summary statistics (for planner convenience):

- Agreement pattern: {agreement_pattern}

- Highest-evidence perspective(s): {top_perspectives}

- Score spread (max - min): {score_spread:.3f}

- Margin spread (max - min): {margin_spread:.3f}

- Uncertainty note: {uncertainty_note}

- Caption status: {status_text}

Usage note:

- This report provides semantic evidence signals only.

- Do NOT treat any single score as a final decision.

- Use agreement, margins, and uncertainty to decide whether to:

  (a) proceed,

  (b) request additional captions,

  (c) request localized / cropped inspection,

  or (d) apply a downstream threshold policy.
"""


def format_keyword_heuristic_report_v2(
    tool_name: str,
    class_name: str,
    escaped_captions: List[str],
    score1_s: str,
    score2_s: str,
    score3_s: str,
    margin1_s: str,
    margin2_s: str,
    margin3_s: str,
    caption_status: List[str],
    agreement_pattern: str,
    top_perspectives: str,
    score_spread: float,
    margin_spread: float,
    uncertainty_note: str,
    status_text: str,
) -> str:
    """Format the keyword heuristic anomaly evidence report (v2, without margin information in output).
    
    Args:
        tool_name: Name of the tool (e.g., "keyword_heuristic")
        class_name: Name of the object class
        escaped_captions: List of 3 escaped caption strings
        score1_s, score2_s, score3_s: Formatted score strings
        margin1_s, margin2_s, margin3_s: Formatted margin strings (not used in output)
        caption_status: List of 3 status strings ("valid", "invalid", "missing")
        agreement_pattern: Agreement pattern description
        top_perspectives: Top perspectives description
        score_spread: Score spread value
        margin_spread: Margin spread value (not used in output)
        uncertainty_note: Uncertainty note description
        status_text: Status summary text
    
    Returns:
        Formatted report string (without margin information)
    """
    return f"""[{tool_name}] Anomaly evidence report (caption-prototype comparison) for class="{class_name}":

Method:

- For each caption, we compared its embedding against two text prototypes:

  NORMAL prototype vs ANOMALOUS prototype.

- anomaly_score in [0,1]: higher means closer to the ANOMALOUS prototype.

Evidence by perspective:

1) Perspective 1 - Overall condition

   - caption: {escaped_captions[0]}

   - anomaly_score: {score1_s}

   - status: {caption_status[0]}

2) Perspective 2 - Surface & texture

   - caption: {escaped_captions[1]}

   - anomaly_score: {score2_s}

   - status: {caption_status[1]}

3) Perspective 3 - Anomaly-oriented judgment

   - caption: {escaped_captions[2]}

   - anomaly_score: {score3_s}

   - status: {caption_status[2]}

Summary statistics (for planner convenience):

- Agreement pattern: {agreement_pattern}

- Highest-evidence perspective(s): {top_perspectives}

- Score spread (max - min): {score_spread:.3f}

- Uncertainty note: {uncertainty_note}

- Caption status: {status_text}

Usage note:

- This report provides semantic evidence signals only.

- Do NOT treat any single score as a final decision.

- Use agreement and uncertainty to decide whether to:

  (a) proceed,

  (b) request additional captions,

  (c) request localized / cropped inspection,

  or (d) apply a downstream threshold policy.
"""
