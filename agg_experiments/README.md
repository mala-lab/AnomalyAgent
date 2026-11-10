# AGG 과제용 수정 사항 (AnomalyAgent 포크)

원 저장소(mala-lab/AnomalyAgent) 구조를 그대로 두고, AGG(금속 시편 현미경 이미지) 실험에 맞게 바꾸거나 추가한 부분만 이 브랜치에 담았다. 이미지, 결과(`results/`), 캡션 원문은 올리지 않았다.

## 원 저장소에서 바뀐 파일
| 파일 | 변경 |
|---|---|
| `src/react_agent/context.py`, `context_v1_27.py` | (1) Anthropic 사용 시 환경변수 `ANTHROPIC_WORKSPACE_ID` 가 있으면 요청 헤더에 넣는다. (2) 환경변수 `AGG_CAPTION_PROMPT_FILE` 이 있으면 이미지 설명 프롬프트를 그 파일로 바꾼다. |
| `src/react_agent/utils.py` | Anthropic 호출 때 system 메시지를 하나로 합친다 (연속되지 않은 system 메시지를 거부함). 모든 API 호출 입출력을 `prompt_dump.py` 로 덤프한다. |

## 새로 추가한 파일
| 파일 | 내용 |
|---|---|
| `src/react_agent/tools.py`, `tools_v1_27.py`, `prompts.py` | **재구성한 구현이며 저자 원본이 아니다.** 공개 저장소에 없는 비주얼 툴(zoom, 밝기, 노이즈 제거, 디블러, 초해상) 5종, general template 근거(`_keyword_heuristic_decision`), counterfactual atomic candidate 도구를 논문 설명을 보고 다시 만든 것이다. 결과가 논문과 다를 수 있다. `prompts.py` 는 `prompts_v1_27.py` 와 내용이 같은 복사본이다. `tools_v1_27.py` 의 MVTec LOCO 용 visual_check 규칙은 구현하지 않았다. |
| `src/react_agent/prompt_dump.py`, `render_prompt_dump.py` | 에이전트가 실제로 보낸 프롬프트와 응답(플래너 출력 포함)을 저장하고 보여 준다. 환경변수 `DUMP_PROMPTS_DIR` 를 줄 때만 동작하고(없으면 아무 일도 하지 않음), 덤프에는 모델에 보낸 이미지도 저장되므로 결과 폴더를 올리지 않는다. |
| `evaluate_agg_v12_target.py` | AGG 평가 스크립트 (이미지 단위/타일 단위). 환경변수 `AGG_DATA_ROOT` 가 가리키는 폴더(기본값은 자리표시자 `/path/to/AGG_labeled_only`) 아래 `exp_aa/outputs/aa_caption_tiles/` 의 후보 파일을 읽는다. |
| `run_tile34.sh`, `run_tile34_v2.sh`, `run_tile34_dump.sh`, `run_image_cand7_v2.sh`, `run_image_cand7_agg_caption.sh`, `run_planner_check.sh` | 실행에 쓴 명령. `AGG_DATA_ROOT`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` 를 환경변수로 준다. |

## 후보 문장 생성 (`agg_experiments/`)
1. `aa_caption_tiles.py`: AGG1 표시 타일 17개와 AGG0 대조 타일 17개를 이미지 설명 모델에 넣어 타일마다 3관점 캡션을 만든다 (라벨 미공개).
2. `aa_make_candidates.py`: 캡션에서 두 그룹이 갈리는 표현을 후보 문장으로 쓰고 정규식으로 센다. 자기 그룹 타일 수 - 반대 그룹 타일 수 >= 2 인 것만 남기고, 남은 것을 바꿔 써서 이상 7개 + 정상 7개로 만든다.
3. `aa_make_neutral_setup.py`: 밝기 방향에 중립인 후보(v2)와 이상 정의를 만든다.
4. `candidates/`: 만들어진 후보 문장 json, 중립 이상 정의, AGG 특화 캡션 프롬프트.

주의: 후보를 만든 타일 34개가 타일 단위 실행의 평가 타일과 같아서, 그 실행의 counterfactual 값은 독립 성능이 아니다.
