"""Render a DUMP_PROMPTS_DIR (calls.jsonl + images/) into one HTML page.

Usage: python render_prompt_dump.py <dump_dir> [results.json] [out.html]
Groups calls by sample (tile) and shows, per call, every message part the node received and its response.
"""

import html
import json
import sys
from collections import OrderedDict
from pathlib import Path

NODE_ORDER = {"image_description": 0, "planner": 1, "reasoner": 2, "reflector": 3}


def esc(s):
    return html.escape(str(s))


def render_parts(parts):
    out = []
    for p in parts:
        if p["type"] == "image":
            if p.get("file"):
                out.append(f'<a href="{esc(p["file"])}" target="_blank"><img src="{esc(p["file"])}" title="{p.get("bytes", 0):,} bytes"></a>')
            else:
                out.append(f'<div class="muted">[image: {esc(p.get("url", ""))}]</div>')
        else:
            out.append(f'<pre>{esc(p.get("text", ""))}</pre>')
    return "".join(out)


def render_call(rec):
    msgs = "".join(
        f'<div class="msg {esc(m["role"])}"><div class="role">{esc(m["role"])}</div>{render_parts(m["parts"])}'
        + (f'<pre class="tc">tool_calls: {esc(json.dumps(m["tool_calls"], ensure_ascii=False))}</pre>' if m.get("tool_calls") else "")
        + "</div>"
        for m in rec["messages"]
    )
    r = rec["response"]
    resp = render_parts(r["parts"])
    if r.get("tool_calls"):
        resp += f'<pre class="tc">tool_calls: {esc(json.dumps(r["tool_calls"], ensure_ascii=False))}</pre>'
    usage = r.get("usage") or {}
    return (
        f'<details class="call {esc(rec["node"])}"><summary><b>#{rec["seq"]} {esc(rec["node"])}</b> '
        f'<span class="muted">{esc(rec.get("model"))} · in {usage.get("input_tokens", "?")} / out {usage.get("output_tokens", "?")} tok</span></summary>'
        f'<h4>Input to model</h4>{msgs}<h4>Response</h4><div class="msg resp">{resp}</div></details>'
    )


def main():
    d = Path(sys.argv[1])
    results = {}
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        for r in json.load(open(sys.argv[2], encoding="utf-8"))["results"]:
            results[Path(r["img_path"]).name] = r
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else d / "prompt_dump.html"

    groups = OrderedDict()
    for line in open(d / "calls.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        groups.setdefault(rec.get("sample") or "(no sample)", []).append(rec)

    body = []
    for sample, recs in groups.items():
        r = results.get(sample)
        badge = ""
        if r:
            ok = (r["pred_result"] == "anomalous") == bool(r["gt_anomaly"])
            badge = (f' <span class="{"ok" if ok else "bad"}">GT={"anom" if r["gt_anomaly"] else "norm"} '
                     f'pred={esc(r["pred_result"])}</span>')
        body.append(f'<section><h2>{esc(sample)}{badge} <span class="muted">{len(recs)} calls</span></h2>'
                    + "".join(render_call(x) for x in recs) + "</section>")

    css = """
    body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
    section{border:1px solid #ccc;border-radius:8px;margin:16px 0;padding:8px 14px}
    details.call{margin:6px 0;border-left:4px solid #999;padding-left:10px}
    details.planner{border-color:#2a7} details.reasoner{border-color:#37c} details.reflector{border-color:#c73} details.image_description{border-color:#aaa}
    .msg{margin:6px 0;padding:6px 10px;background:#f6f6f6;border-radius:6px}.msg.resp{background:#eef6ff}
    .role{font-size:11px;text-transform:uppercase;color:#666} pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:12.5px/1.45 ui-monospace,monospace}
    img{max-height:200px;border:1px solid #ccc;margin:2px} .muted{color:#888;font-weight:normal;font-size:12px} .tc{color:#a40}
    .ok{color:#080}.bad{color:#c00}
    """
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>Prompt dump</title><style>{css}</style>'
                   f'<h1>Planner / Reasoner inputs</h1>{"".join(body)}', encoding="utf-8")
    print(f"wrote {out} ({len(groups)} samples)")


if __name__ == "__main__":
    main()
