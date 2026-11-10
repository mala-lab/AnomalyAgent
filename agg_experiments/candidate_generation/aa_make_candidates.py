"""aa_make_candidates: aa_caption_tiles 의 캡션(AGG1 표시 타일 17개, AGG0 대조 타일 17개)에서 AnomalyAgent 의 반사실 후보 도구가 쓰는 후보 문장을 정리한다.

후보는 캡션에서 두 그룹이 갈리는 표현을 직접 읽고 짧은 한 문장으로 쓴 것이고, 후보마다 근거(그 표현이 나온 캡션 수, 정규식)를 센다.
이상 후보(anomaly_candidates): AGG1 표시 타일 쪽에 많았던 표현.  정상 후보(normal_candidates): AGG0 대조 타일 쪽에 많았던 표현.
출력: outputs/aa_caption_tiles/atomic_candidates_agg.json (에이전트 후보 파일 형식), candidate_evidence.csv (후보마다 근거 개수)
주의: 후보를 만든 타일과 같은 타일로 평가하면 성능이 부풀려진다 (AGG2 이미지와 표시하지 않은 AGG1 타일로 평가해야 함).
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/aa_caption_tiles"
CLASS = "metal sample surface"
MIN_GAP = 2     # 17타일 중 자기 그룹에서 나온 타일 수 - 반대 그룹 타일 수 가 이 값 이상인 후보만 파일에 넣는다

# (후보 문장, 근거 정규식)  -- 근거 정규식은 캡션에서 그 표현이 나왔는지 세기 위한 것
ANOMALY = [
    ("The surface has a few faint darker mottled patches.", r"darker[^.]{0,40}(mottled )?(patch|patches|blotch|blotches|area)|(patch|patches|blotch|blotches)[^.]{0,30}darker"),
    ("Slightly darker, cloudy patches appear on the surface.", r"cloudy patch|darker, (cloudy|diffuse|blurry)|hazy|hazier"),
    ("A faint lighter, hazy blotch appears on the surface.", r"lighter[^.]{0,20}(blotch|patch|hazy)|hazier patch|pale"),
    ("The surface has uneven brightness with faint local patches.", r"unevenly lit|uneven brightness|local variations? in brightness|slight local"),
    ("A few isolated dark pits or spots are visible on the surface.", r"dark pits or spots|isolated dark|darker pits"),
]
NORMAL = [
    ("The surface has a uniform fine-grained texture with no distinct regions.", r"no distinct regions"),
    ("Small bright specks are spread evenly across the field.", r"(spread|scattered|dotted) evenly|evenly across|spread evenly"),
    ("There are no distinct scratches, cracks, or other irregularities.", r"no distinct (scratches|cracks|defects)|no distinct regions, scratches"),
    ("The only variation is slight mottling or shifts in brightness from the granular roughness.", r"slight (mottled|shifts)|only variation|gradual shifts|shifts in brightness"),
    ("The surface is uniformly granular with consistent speckle density and brightness.", r"consistent throughout|uniformly granular|uniform, fine-grained"),
]

# 근거가 있는 후보(포함=예)의 바꿔쓰기. 같은 뜻을 다른 단어로 써서, 임베딩이 특정 단어에만 반응하는 것을 줄인다.
# (캡션 개수는 부모 후보의 것을 따르며, 새 근거가 아니다)
PARAPHRASES = [
    ("The surface has a few faint darker mottled patches.", "anomaly", "A few patches on the surface look slightly darker and mottled."),
    ("The surface has a few faint darker mottled patches.", "anomaly", "Faint dark mottled areas stand out against the surrounding texture."),
    ("Slightly darker, cloudy patches appear on the surface.", "anomaly", "Hazy, diffuse darker regions are scattered on the surface."),
    ("The surface has uneven brightness with faint local patches.", "anomaly", "Brightness is uneven, with faint local patches that differ from the rest."),
    ("The surface has a uniform fine-grained texture with no distinct regions.", "normal", "The surface texture is uniform and finely grained across the whole field."),
    ("The surface has a uniform fine-grained texture with no distinct regions.", "normal", "No distinct region stands out from the fine granular texture."),
    ("The only variation is slight mottling or shifts in brightness from the granular roughness.", "normal", "Brightness varies only slightly, following the granular roughness of the surface."),
    ("The surface is uniformly granular with consistent speckle density and brightness.", "normal", "The granular surface looks consistent in speckle density and brightness throughout."),
]


def main():
    caps = json.load(open(OUT / "captions.json", encoding="utf-8"))["captions"]
    # 안쪽 타일(가장자리 배경 영향이 없는 타일)만 따로도 센다
    rows = []
    for kind, lst in (("anomaly", ANOMALY), ("normal", NORMAL)):
        for sent, pat in lst:
            rx = re.compile(pat)
            row = dict(종류=kind, 후보=sent)
            for g, gname in (("AGG1표시", "AGG1표시"), ("AGG0대조", "AGG0대조")):
                cs = [c for c in caps if c["label"] == g]
                row[f"{gname} 타일 수(17개 중)"] = len({c["n"] for c in cs if rx.search(c["caption"].lower())})
                ci = [c for c in cs if c["mask_frac"] >= 0.999]
                row[f"{gname} 안쪽 타일 수(9개 중)"] = len({c["n"] for c in ci if rx.search(c["caption"].lower())})
            own, other = ("AGG1표시", "AGG0대조") if kind == "anomaly" else ("AGG0대조", "AGG1표시")
            row["포함"] = "예" if row[f"{own} 타일 수(17개 중)"] - row[f"{other} 타일 수(17개 중)"] >= MIN_GAP else "아니오(두 그룹에서 비슷하게 나옴)"
            rows.append(row)
    for r in rows:
        r["출처"] = "캡션 집계"
    for parent, kind, sent in PARAPHRASES:
        base = next(r for r in rows if r["후보"] == parent)
        rows.append({**base, "종류": kind, "후보": sent, "출처": "바꿔쓰기: " + parent[:40] + "...", "포함": base["포함"]})
    keep = lambda kind: [r["후보"] for r in rows if r["종류"] == kind and r["포함"] == "예"]
    json.dump({CLASS.replace(" ", "_"): {"anomaly_candidates": keep("anomaly"), "normal_candidates": keep("normal")}},
              open(OUT / "atomic_candidates_agg.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(OUT / "candidate_evidence.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["종류"][:3], "|", r["후보"][:60].ljust(60), "| 표시", r["AGG1표시 타일 수(17개 중)"], "/ 대조", r["AGG0대조 타일 수(17개 중)"], "| 안쪽 표시", r["AGG1표시 안쪽 타일 수(9개 중)"], "/ 대조", r["AGG0대조 안쪽 타일 수(9개 중)"])


if __name__ == "__main__":
    main()
