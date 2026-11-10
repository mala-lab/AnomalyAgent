"""aa_caption_tiles: AGG1 에서 직접 표시한 AGG 타일 17개와 AGG0 이미지의 타일 17개를 Claude API 로 캡션 생성하고, 반복되는 표현을 비교한다.

aa_caption_phrases.py 와 같은 방식 (타일마다 관점 3가지 캡션, 라벨 정보는 모델에 알리지 않음)이고, 보내는 단위가 사진 전체가 아니라 224px 타일이다 (448px 로 확대해서 전송).
타일 선택:
  - AGG1(표시) : exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv 에서 has_agg == 1 인 17개 전부.
  - AGG0(대조) : AGG0 이미지의 분석 타일 중 17개. 표시 타일의 안쪽(mask_frac=1)/가장자리 개수와 같게 뽑고 (가장자리는 mask_frac 0.75 이상만), 같은 이미지에서 2개 이상 안 뽑는다 (seed 고정).
실행: ANTHROPIC_API_KEY (필요하면 ANTHROPIC_WORKSPACE_ID) 를 환경변수로 설정한 뒤  python aa_caption_tiles.py [--dry-run]
출력: outputs/aa_caption_tiles/captions.json, phrases.csv, tiles/<그룹>_<번호>.png (보낸 타일 확인용, 로컬 전용), README.txt
주의: 타일 이미지 34장을 외부 API 로 전송한다.
"""
import argparse
import collections
import csv
import io
import json
import os
import random
from pathlib import Path

from PIL import Image

import aa_caption_phrases as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAB = ROOT / "exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv"
OUT = HERE / "outputs/aa_caption_tiles"
T = 224


def select(seed=0):
    g00 = {}
    for r in csv.DictReader(open(base.G00, encoding="utf-8-sig")):
        if r["selected"] == "1":
            g00[(r["image_id"], int(r["x0"]), int(r["y0"]))] = r
    marked = []
    for r in csv.DictReader(open(LAB, encoding="utf-8-sig")):
        if r["has_agg"] == "1":
            k = (r["image_id"], int(r["x0"]), int(r["y0"])); marked.append(dict(group="AGG1표시", **_t(g00[k])))
    inner = lambda t: t["mask_frac"] >= 0.999999
    n_in = sum(1 for t in marked if inner(t)); n_ed = len(marked) - n_in
    rng = random.Random(seed)
    pool = [_t(r) for r in g00.values() if r["label"] == "0"]
    ctrl, used = [], set()
    for want, flag in ((n_in, True), (n_ed, False)):
        c = [t for t in pool if inner(t) == flag and t["mask_frac"] >= 0.75]; rng.shuffle(c); got = 0      # 표시 타일의 가장자리 타일은 mask_frac 0.79 이상이라 대조도 0.75 이상만
        for t in c:
            if got == want:
                break
            if t["image_id"] in used:
                continue
            used.add(t["image_id"]); ctrl.append(dict(group="AGG0대조", **t)); got += 1
        assert got == want, "AGG0 이미지가 부족함"
    return marked, ctrl


def _t(r):
    return dict(image_id=r["image_id"], tile_id=r["tile_id"], x0=int(r["x0"]), y0=int(r["y0"]), mask_frac=round(float(r["mask_frac"]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()
    marked, ctrl = select()
    tiles = marked + ctrl
    for t in tiles:
        print(t["group"], t["image_id"].split(" - ")[0], t["tile_id"], f"mask_frac={t['mask_frac']}")
    print(f"AGG1 표시 {len(marked)}개 (안쪽 {sum(t['mask_frac'] >= 0.999999 for t in marked)}), AGG0 대조 {len(ctrl)}개 (안쪽 {sum(t['mask_frac'] >= 0.999999 for t in ctrl)})")
    if a.dry_run:
        return
    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    caps = []
    for n, t in enumerate(tiles, 1):
        im = Image.open(ROOT / (t["image_id"] + ".jpg")).convert("RGB").crop((t["x0"], t["y0"], t["x0"] + T, t["y0"] + T)).resize((448, 448), Image.LANCZOS)
        im.save(OUT / "tiles" / f"{t['group']}_{n:02d}.png")
        b = io.BytesIO(); im.save(b, "JPEG", quality=92)
        import base64
        data = base64.standard_b64encode(b.getvalue()).decode()
        for k, q in enumerate(base.PERSPECTIVES, 1):
            msg = client.messages.create(model=a.model, max_tokens=200, system=base.SYSTEM, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, {"type": "text", "text": q}]}])
            caps.append(dict(label=t["group"], n=n, image_id=t["image_id"], tile_id=t["tile_id"], mask_frac=t["mask_frac"], perspective=k, caption=msg.content[0].text.strip()))
        print(" ", n, t["group"], t["tile_id"], flush=True)
    json.dump(dict(model=a.model, captions=caps), open(OUT / "captions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    groups = ["AGG1표시", "AGG0대조"]
    cnt = {g: collections.Counter() for g in groups}
    ncap = {g: sum(1 for c in caps if c["label"] == g) for g in groups}
    for c in caps:
        cnt[c["label"]].update(base.ngrams(c["caption"]))
    allg = set().union(*[set(v) for v in cnt.values()])
    rows = sorted(((g, cnt[groups[0]][g], cnt[groups[1]][g]) for g in allg if cnt[groups[0]][g] + cnt[groups[1]][g] >= 4), key=lambda r: -(r[1] + r[2]))
    with open(OUT / "phrases.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(["표현", f"AGG1 표시 타일 캡션 수 (전체 {ncap[groups[0]]})", f"AGG0 대조 타일 캡션 수 (전체 {ncap[groups[1]]})"]); w.writerows(rows)
    (OUT / "README.txt").write_text(__doc__.strip() + f"\n\n모델: {a.model}\n", encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
