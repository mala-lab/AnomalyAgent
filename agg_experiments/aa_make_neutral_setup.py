"""aa_make_neutral_setup: 밝기 방향에 중립적인 후보 문장 파일과 실행 스크립트(타일 34개 재실행용)를 만든다.

근거: 사용자가 표시한 AGG 15곳을 주변 띠와 비교하면 어두움 9 / 밝음 3 / 비슷 3 이었고(aa_region_polarity.py),
      상자 안 질감도 바깥보다 매끈하지 않았다(평균 1.07). 그래서 지난 후보("darker mottled" 쪽으로 쏠림)와 정의("평평하고 매끈한 영역")를 바꾼다.
이 후보는 캡션 집계가 아니라 위 측정을 보고 직접 쓴 문장이다. 지난번과 달라진 점은 이상 후보에 밝은 얼룩을 넣은 것,
"옅은 얼룩은 정상"이라고 말하던 정상 후보를 뺀 것, 가장자리·조명 띠를 정상 후보에 넣은 것이다.
출력: outputs/aa_caption_tiles/atomic_candidates_agg_neutral.json, ~/AnomalyAgent/run_tile34_v2.sh (WSL)
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/aa_caption_tiles"

CANDIDATES = {
    "metal_sample_surface": {
        "anomaly_candidates": [
            "A soft-edged patch on the surface is slightly darker than the area around it.",
            "A soft-edged patch on the surface is slightly lighter than the area around it.",
            "A localized area differs in brightness from the surrounding fine granular texture.",
            "A small blotch with the same granular texture looks faintly darker or brighter than its surroundings.",
        ],
        "normal_candidates": [
            "The surface has a uniform fine-grained texture with no distinct regions.",
            "The surface is uniformly granular with consistent speckle density and brightness.",
            "Brightness changes gradually across the surface without any localized patch.",
            "The dark band and bright haze along the sample edge come from the sample boundary and lighting.",
        ],
    }
}

DEFINITION = (
    "A region is anomalous (AGG) when a localized area of the sample surface, with a soft and gradual edge, looks darker or lighter than the surrounding surface "
    "while keeping a similar fine granular texture; the patch may be faint and low in contrast. "
    "The following are not anomalous: the dark background and sample edges, lighting bands or haze that change gradually across the whole surface, "
    "scratches, fibers, dust, and stains. A surface without such a localized darker or lighter patch should be judged normal."
)

json.dump(CANDIDATES, open(OUT / "atomic_candidates_agg_neutral.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
(OUT / "target_anomaly_definition_neutral.txt").write_text(DEFINITION, encoding="utf-8")

run = """#!/bin/bash
cd ~/AnomalyAgent
M=anthropic/claude-sonnet-5-5
DEF="$(cat "$AGG_DATA_ROOT/exp_aa/outputs/aa_caption_tiles/target_anomaly_definition_neutral.txt")"
$HOME/anomalyagent_venv/bin/python evaluate_agg_v12_target.py --unit tile --save_path ./results/agg_tile34_v2 --atomic_candidates_file atomic_candidates_agg_neutral.json --attn_implementation sdpa \\
  --target_anomaly_definition "$DEF" \\
  --planner_model_type api --planner_api_model $M --reflector_model_type api --reflector_api_model $M --reasoner_model_type api --reasoner_api_model $M \\
  --image_description_model_type api --image_description_api_model $M --atomic_candidate_llm_model_type api --atomic_candidate_llm_api_model $M --concurrent 1 2>&1 | tee ./results/agg_tile34_v2.log
"""
q = os.path.expanduser("~/AnomalyAgent/run_tile34_v2.sh")
open(q, "w", encoding="utf-8", newline="\n").write(run)
os.chmod(q, 0o755)
print("ok", q)
