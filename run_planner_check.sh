#!/bin/bash
cd ~/AnomalyAgent
M=anthropic/claude-sonnet-5-5
$HOME/anomalyagent_venv/bin/python evaluate_agg_v12_target.py --unit image --max_samples 2 --save_path ./results/agg_img_planner_check --atomic_candidates_file atomic_candidates_agg.json --attn_implementation sdpa \
  --planner_model_type api --planner_api_model $M --reflector_model_type api --reflector_api_model $M --reasoner_model_type api --reasoner_api_model $M \
  --image_description_model_type api --image_description_api_model $M --atomic_candidate_llm_model_type api --atomic_candidate_llm_api_model $M --concurrent 1 2>&1 \
  | grep -E "planner|\[ERROR\]|Error|Evaluation Results|Results saved"
