#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
CONCURRENT="${CONCURRENT:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
MAX_WAIT_MS="${MAX_WAIT_MS:-50}"
UNCERTAIN_SCORE="${UNCERTAIN_SCORE:-0.5}"
QWEN_VL_MODEL_PATH="${QWEN_VL_MODEL_PATH:-./models/qwen3_vl_4b}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
SUPER_RESOLUTION_VERSION="${SUPER_RESOLUTION_VERSION:-nongan}"
COUNTERFACTUAL_TOP_K="${COUNTERFACTUAL_TOP_K:-3}"
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-16}"
TOOL_CALL_THRESHOLD="${TOOL_CALL_THRESHOLD:-5}"
RAW_TO_REASONER="${RAW_TO_REASONER:-true}"
RAW_TO_REFLECTOR="${RAW_TO_REFLECTOR:-true}"
NORMAL_TOLERANCE="${NORMAL_TOLERANCE:-2}"
RESIZE="${RESIZE:-false}"
USE_RESPONSES_API="${USE_RESPONSES_API:-false}"

PLANNER_MODEL_TYPE="${PLANNER_MODEL_TYPE:-local}"
REFLECTOR_MODEL_TYPE="${REFLECTOR_MODEL_TYPE:-local}"
REASONER_MODEL_TYPE="${REASONER_MODEL_TYPE:-local}"
IMAGE_DESCRIPTION_MODEL_TYPE="${IMAGE_DESCRIPTION_MODEL_TYPE:-local}"
ATOMIC_CANDIDATE_LLM_MODEL_TYPE="${ATOMIC_CANDIDATE_LLM_MODEL_TYPE:-local}"

MVTEC_DATA_PATH="${MVTEC_DATA_PATH:-./data/mvtec}"
MVTEC_SAVE_PATH="${MVTEC_SAVE_PATH:-./results/mvtec_eval}"
MVTEC_LOCO_DATA_PATH="${MVTEC_LOCO_DATA_PATH:-./data/mvtec_loco}"
MVTEC_LOCO_SAVE_PATH="${MVTEC_LOCO_SAVE_PATH:-./results/mvtecloco_eval}"
HEADCT_DATA_PATH="${HEADCT_DATA_PATH:-./data/headct/images}"
HEADCT_LABELS_CSV="${HEADCT_LABELS_CSV:-./data/headct/labels.csv}"
HEADCT_SAVE_PATH="${HEADCT_SAVE_PATH:-./results/headct_eval}"
HEADCT_TARGET_ANOMALY_DEFINITION="${HEADCT_TARGET_ANOMALY_DEFINITION:-Only intracranial hemorrhage or visible bleeding is anomalous. Head CT images without hemorrhage should be judged normal/target-negative, even if other non-hemorrhagic findings are visible.}"
LAG_DATA_PATH="${LAG_DATA_PATH:-./data/lag/test}"
LAG_SAVE_PATH="${LAG_SAVE_PATH:-./results/lag_eval}"
KAPUTT_PARQUET_PATH="${KAPUTT_PARQUET_PATH:-./data/kaputt/query-test.parquet}"
KAPUTT_CROP_ROOT="${KAPUTT_CROP_ROOT:-./data/kaputt/query-crop/data/test/query-data/crop}"
KAPUTT_QUERY_IMAGE_ROOT="${KAPUTT_QUERY_IMAGE_ROOT:-./data/kaputt/query-image/data/test/query-data}"
KAPUTT_IMAGE_SUBDIR="${KAPUTT_IMAGE_SUBDIR:-}"
KAPUTT_MODE="${KAPUTT_MODE:-both}"
KAPUTT_SAVE_PATH="${KAPUTT_SAVE_PATH:-./results/kaputt_eval}"

MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_NUM="${MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_NUM:-1}"
MVTEC_LOCO_COUNTERFACTUAL_CHECKS_FILE="${MVTEC_LOCO_COUNTERFACTUAL_CHECKS_FILE:-counterfactual_atomic_visual_checks.json}"
MVTEC_LOCO_HARD_RULE_FAIL_CONFIDENCE="${MVTEC_LOCO_HARD_RULE_FAIL_CONFIDENCE:-0.7}"
MVTEC_LOCO_HARD_RULE_MIN_FAIL_COUNT="${MVTEC_LOCO_HARD_RULE_MIN_FAIL_COUNT:-1}"
read -r -a MVTEC_LOCO_COUNTERFACTUAL_STRICTNESS <<< "${MVTEC_LOCO_COUNTERFACTUAL_STRICTNESS:-soft}"
read -r -a MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_SOURCE <<< "${MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_SOURCE:-none}"
read -r -a MVTEC_LOCO_COUNTERFACTUAL_MATCHING_RULE <<< "${MVTEC_LOCO_COUNTERFACTUAL_MATCHING_RULE:-textual_matching}"

COMMON_OPTIONAL_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "${EMBEDDING_MODEL_PATH:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--embedding_model_path "$EMBEDDING_MODEL_PATH")
fi
if [[ "${SHOW_TOKEN_NUM:-false}" == "true" ]]; then
  COMMON_OPTIONAL_ARGS+=(--show_token_num)
fi
if [[ -n "${ATOMIC_CANDIDATES_FILE:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--atomic_candidates_file "$ATOMIC_CANDIDATES_FILE")
fi
if [[ -n "${PLANNER_API_MODEL:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--planner_api_model "$PLANNER_API_MODEL")
fi
if [[ -n "${REFLECTOR_API_MODEL:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--reflector_api_model "$REFLECTOR_API_MODEL")
fi
if [[ -n "${REASONER_API_MODEL:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--reasoner_api_model "$REASONER_API_MODEL")
fi
if [[ -n "${IMAGE_DESCRIPTION_API_MODEL:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--image_description_api_model "$IMAGE_DESCRIPTION_API_MODEL")
fi
if [[ -n "${ATOMIC_CANDIDATE_LLM_API_MODEL:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--atomic_candidate_llm_api_model "$ATOMIC_CANDIDATE_LLM_API_MODEL")
fi
if [[ -n "${API_TEMPERATURE:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--api_temperature "$API_TEMPERATURE")
fi
if [[ -n "${API_MAX_TOKENS:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--api_max_tokens "$API_MAX_TOKENS")
fi
if [[ -n "${API_TOP_P:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--api_top_p "$API_TOP_P")
fi
if [[ -n "${API_FREQUENCY_PENALTY:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--api_frequency_penalty "$API_FREQUENCY_PENALTY")
fi
if [[ -n "${API_PRESENCE_PENALTY:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--api_presence_penalty "$API_PRESENCE_PENALTY")
fi
if [[ -n "${REASONING_EFFORT:-}" ]]; then
  COMMON_OPTIONAL_ARGS+=(--reasoning_effort "$REASONING_EFFORT")
fi

MVTEC_LOCO_OPTIONAL_ARGS=()
if [[ -n "${MVTEC_LOCO_COUNTERFACTUAL_RULES_FILE:-}" ]]; then
  MVTEC_LOCO_OPTIONAL_ARGS+=(--counterfactual_rules_file "$MVTEC_LOCO_COUNTERFACTUAL_RULES_FILE")
fi

echo "[1/5] Running MVTec"
python evaluate_mvtec_v12.py \
  --data_path "$MVTEC_DATA_PATH" \
  --save_path "$MVTEC_SAVE_PATH" \
  --qwen_model_path "$QWEN_VL_MODEL_PATH" \
  --device "$DEVICE" \
  --concurrent "$CONCURRENT" \
  --uncertain_score "$UNCERTAIN_SCORE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_wait_ms "$MAX_WAIT_MS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --counterfactual_top_k "$COUNTERFACTUAL_TOP_K" \
  --super_resolution_version "$SUPER_RESOLUTION_VERSION" \
  --planner_model_type "$PLANNER_MODEL_TYPE" \
  --reflector_model_type "$REFLECTOR_MODEL_TYPE" \
  --reasoner_model_type "$REASONER_MODEL_TYPE" \
  --image_description_model_type "$IMAGE_DESCRIPTION_MODEL_TYPE" \
  --atomic_candidate_llm_model_type "$ATOMIC_CANDIDATE_LLM_MODEL_TYPE" \
  --api_max_concurrency "$API_MAX_CONCURRENCY" \
  --use_responses_api "$USE_RESPONSES_API" \
  --tool_call_threshold "$TOOL_CALL_THRESHOLD" \
  --raw_to_reasoner "$RAW_TO_REASONER" \
  --raw_to_reflector "$RAW_TO_REFLECTOR" \
  --normal_tolerance "$NORMAL_TOLERANCE" \
  --resize "$RESIZE" \
  "${COMMON_OPTIONAL_ARGS[@]}"

echo "[2/5] Running MVTec LOCO"
python evaluate_mvtecloco_v27.py \
  --data_path "$MVTEC_LOCO_DATA_PATH" \
  --save_path "$MVTEC_LOCO_SAVE_PATH" \
  --qwen_model_path "$QWEN_VL_MODEL_PATH" \
  --device "$DEVICE" \
  --concurrent "$CONCURRENT" \
  --uncertain_score "$UNCERTAIN_SCORE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_wait_ms "$MAX_WAIT_MS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --counterfactual_top_k "$COUNTERFACTUAL_TOP_K" \
  --counterfactual_template_num "$MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_NUM" \
  --counterfactual_strictness "${MVTEC_LOCO_COUNTERFACTUAL_STRICTNESS[@]}" \
  --counterfactual_template_source "${MVTEC_LOCO_COUNTERFACTUAL_TEMPLATE_SOURCE[@]}" \
  --counterfactual_matching_rule "${MVTEC_LOCO_COUNTERFACTUAL_MATCHING_RULE[@]}" \
  --counterfactual_checks_file "$MVTEC_LOCO_COUNTERFACTUAL_CHECKS_FILE" \
  --hard_rule_fail_confidence "$MVTEC_LOCO_HARD_RULE_FAIL_CONFIDENCE" \
  --hard_rule_min_fail_count "$MVTEC_LOCO_HARD_RULE_MIN_FAIL_COUNT" \
  --super_resolution_version "$SUPER_RESOLUTION_VERSION" \
  --planner_model_type "$PLANNER_MODEL_TYPE" \
  --reflector_model_type "$REFLECTOR_MODEL_TYPE" \
  --reasoner_model_type "$REASONER_MODEL_TYPE" \
  --image_description_model_type "$IMAGE_DESCRIPTION_MODEL_TYPE" \
  --atomic_candidate_llm_model_type "$ATOMIC_CANDIDATE_LLM_MODEL_TYPE" \
  --api_max_concurrency "$API_MAX_CONCURRENCY" \
  --use_responses_api "$USE_RESPONSES_API" \
  --tool_call_threshold "$TOOL_CALL_THRESHOLD" \
  --raw_to_reasoner "$RAW_TO_REASONER" \
  --raw_to_reflector "$RAW_TO_REFLECTOR" \
  --normal_tolerance "$NORMAL_TOLERANCE" \
  --resize "$RESIZE" \
  "${COMMON_OPTIONAL_ARGS[@]}" \
  "${MVTEC_LOCO_OPTIONAL_ARGS[@]}"

echo "[3/5] Running HeadCT"
python evaluate_headCT_v12_target.py \
  --data_path "$HEADCT_DATA_PATH" \
  --labels_csv "$HEADCT_LABELS_CSV" \
  --save_path "$HEADCT_SAVE_PATH" \
  --qwen_model_path "$QWEN_VL_MODEL_PATH" \
  --device "$DEVICE" \
  --concurrent "$CONCURRENT" \
  --uncertain_score "$UNCERTAIN_SCORE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_wait_ms "$MAX_WAIT_MS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --target_anomaly_definition "$HEADCT_TARGET_ANOMALY_DEFINITION" \
  --counterfactual_top_k "$COUNTERFACTUAL_TOP_K" \
  --super_resolution_version "$SUPER_RESOLUTION_VERSION" \
  --planner_model_type "$PLANNER_MODEL_TYPE" \
  --reflector_model_type "$REFLECTOR_MODEL_TYPE" \
  --reasoner_model_type "$REASONER_MODEL_TYPE" \
  --image_description_model_type "$IMAGE_DESCRIPTION_MODEL_TYPE" \
  --atomic_candidate_llm_model_type "$ATOMIC_CANDIDATE_LLM_MODEL_TYPE" \
  --api_max_concurrency "$API_MAX_CONCURRENCY" \
  --use_responses_api "$USE_RESPONSES_API" \
  --tool_call_threshold "$TOOL_CALL_THRESHOLD" \
  --raw_to_reasoner "$RAW_TO_REASONER" \
  --raw_to_reflector "$RAW_TO_REFLECTOR" \
  --normal_tolerance "$NORMAL_TOLERANCE" \
  --resize "$RESIZE" \
  "${COMMON_OPTIONAL_ARGS[@]}"

echo "[4/5] Running LAG"
python evaluate_lag_v12_ori.py \
  --data_path "$LAG_DATA_PATH" \
  --save_path "$LAG_SAVE_PATH" \
  --qwen_model_path "$QWEN_VL_MODEL_PATH" \
  --device "$DEVICE" \
  --concurrent "$CONCURRENT" \
  --uncertain_score "$UNCERTAIN_SCORE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_wait_ms "$MAX_WAIT_MS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --counterfactual_top_k "$COUNTERFACTUAL_TOP_K" \
  --super_resolution_version "$SUPER_RESOLUTION_VERSION" \
  --planner_model_type "$PLANNER_MODEL_TYPE" \
  --reflector_model_type "$REFLECTOR_MODEL_TYPE" \
  --reasoner_model_type "$REASONER_MODEL_TYPE" \
  --image_description_model_type "$IMAGE_DESCRIPTION_MODEL_TYPE" \
  --atomic_candidate_llm_model_type "$ATOMIC_CANDIDATE_LLM_MODEL_TYPE" \
  --api_max_concurrency "$API_MAX_CONCURRENCY" \
  --use_responses_api "$USE_RESPONSES_API" \
  --tool_call_threshold "$TOOL_CALL_THRESHOLD" \
  --raw_to_reasoner "$RAW_TO_REASONER" \
  --raw_to_reflector "$RAW_TO_REFLECTOR" \
  --normal_tolerance "$NORMAL_TOLERANCE" \
  --resize "$RESIZE" \
  "${COMMON_OPTIONAL_ARGS[@]}"

echo "[5/5] Running Kaputt"
python evaluate_kaputt_v12.py \
  --parquet_path "$KAPUTT_PARQUET_PATH" \
  --crop_root "$KAPUTT_CROP_ROOT" \
  --query_image_root "$KAPUTT_QUERY_IMAGE_ROOT" \
  --image_subdir "$KAPUTT_IMAGE_SUBDIR" \
  --mode "$KAPUTT_MODE" \
  --save_path "$KAPUTT_SAVE_PATH" \
  --qwen_model_path "$QWEN_VL_MODEL_PATH" \
  --device "$DEVICE" \
  --concurrent "$CONCURRENT" \
  --uncertain_score "$UNCERTAIN_SCORE" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_wait_ms "$MAX_WAIT_MS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --counterfactual_top_k "$COUNTERFACTUAL_TOP_K" \
  --super_resolution_version "$SUPER_RESOLUTION_VERSION" \
  --planner_model_type "$PLANNER_MODEL_TYPE" \
  --reflector_model_type "$REFLECTOR_MODEL_TYPE" \
  --reasoner_model_type "$REASONER_MODEL_TYPE" \
  --image_description_model_type "$IMAGE_DESCRIPTION_MODEL_TYPE" \
  --atomic_candidate_llm_model_type "$ATOMIC_CANDIDATE_LLM_MODEL_TYPE" \
  --api_max_concurrency "$API_MAX_CONCURRENCY" \
  --use_responses_api "$USE_RESPONSES_API" \
  --tool_call_threshold "$TOOL_CALL_THRESHOLD" \
  --raw_to_reasoner "$RAW_TO_REASONER" \
  --raw_to_reflector "$RAW_TO_REFLECTOR" \
  --normal_tolerance "$NORMAL_TOLERANCE" \
  --resize "$RESIZE" \
  "${COMMON_OPTIONAL_ARGS[@]}"
