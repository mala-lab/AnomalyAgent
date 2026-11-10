#!/bin/bash
# agg_img_cand7_v2 와 같은 설정(이미지 29장, 후보 7+7, 기본 이상 정의)에서 에이전트의 캡션 프롬프트만 AGG 특화 3관점으로 바꿔 돌린다.
# 필요: 환경변수 ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID (이전 실행과 같은 것). 사본을 ~/AnomalyAgent/ 에 두고 실행한다.
cd ~/AnomalyAgent || exit 1
export AGG_CAPTION_PROMPT_FILE="$AGG_DATA_ROOT/exp_aa/outputs/aa_caption_tiles/prompt4description_agg.txt"
M=anthropic/claude-sonnet-5-5
$HOME/anomalyagent_venv/bin/python evaluate_agg_v12_target.py --unit image --save_path ./results/agg_img_cand7_aggcaption --atomic_candidates_file atomic_candidates_agg.json --attn_implementation sdpa \
  --planner_model_type api --planner_api_model $M --reflector_model_type api --reflector_api_model $M --reasoner_model_type api --reasoner_api_model $M \
  --image_description_model_type api --image_description_api_model $M --atomic_candidate_llm_model_type api --atomic_candidate_llm_api_model $M --concurrent 1 2>&1 | tee ./results/agg_img_cand7_aggcaption.log
