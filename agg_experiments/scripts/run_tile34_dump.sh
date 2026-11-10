#!/bin/bash
# run_tile34.sh 와 동일한 설정(타일 34개, 후보 7+7, 기본 이상 정의, Sonnet 5.5)에 프롬프트 덤프만 켠다.
# 필요: ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID (이전 실행과 같은 것)
cd "$(dirname "$0")/../.." || exit 1   # 저장소 루트
export DUMP_PROMPTS_DIR=./results/agg_tile34_dump/prompts
mkdir -p ./results/agg_tile34_dump
cp --update=none results/agg_tile34/atomic_candidates_agg.json ./results/agg_tile34_dump/  # 같은 7+7 후보를 재사용(재생성 방지)
M=anthropic/claude-sonnet-5-5
"${PYTHON:-$HOME/anomalyagent_venv/bin/python}" evaluate_agg_v12_target.py --unit tile --save_path ./results/agg_tile34_dump --atomic_candidates_file atomic_candidates_agg.json --attn_implementation sdpa \
  --planner_model_type api --planner_api_model $M --reflector_model_type api --reflector_api_model $M --reasoner_model_type api --reasoner_api_model $M \
  --image_description_model_type api --image_description_api_model $M --atomic_candidate_llm_model_type api --atomic_candidate_llm_api_model $M --concurrent 1 2>&1 | tee ./results/agg_tile34_dump.log
"${PYTHON:-$HOME/anomalyagent_venv/bin/python}" agg_experiments/render_prompt_dump.py ./results/agg_tile34_dump/prompts ./results/agg_tile34_dump/results.json
