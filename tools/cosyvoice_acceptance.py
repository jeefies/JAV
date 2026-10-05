#!/usr/bin/env python3
"""《公示期》配音验收工具（wants.md §8）：技术验收自动 + 听感验收清单。

对每个音色、每句剧本台词跑「默认合成 / 带表演指令」A/B 对照，下载产物并
自动校验技术项（mono PCM16、采样率、时长、非静音 RMS、削波、sha256 回执、
实际推理模式），生成 markdown 报告供人工听感检查。

用法（JAV 空闲时运行——单 runtime 互斥）:
  python3 tools/cosyvoice_acceptance.py [--voices demo-zh-f,demo-zh-m]
  可选: --extra 自定义 JSON {voice_id: [台词...]}
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = os.getenv("JAV_BASE_TOKEN_URL", "http://127.0.0.1:8765")
TERMINAL = ("completed", "failed", "cancelled")

# wants.md §8 指定验收台词
SCRIPT_LINES = [
    "这场报了三个人，记录得对一下。",
    "饭先吃。有人问公示什么时候完。",
    "那之前公布的两年呢？",
    "但不会公开来说。学院老师会做他的工作。",
]
# A/B 对照用的表演指令（与默认合成同一 seed，唯一变量是 instruction）
INSTRUCTION = "像日常对话一样自然地说，语气克制、略带疲惫，句尾轻收"


def token() -> str:
    t = os.getenv("JAV_API_TOKEN", "")
    if t:
        return t
    env = Path.home() / ".config/jav/env"
    for line in env.read_text().splitlines():
        if line.startswith("JAV_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no JAV_API_TOKEN (env or ~/.config/jav/env)")


def spec(voice: str, text: str, instructed: bool, ref: str, seed: int) -> dict:
    inputs = {"text": text, "voice_id": voice}
    if instructed:
        inputs["instruction"] = INSTRUCTION
    return {"provider": "cosyvoice", "workflow": "t2a", "inputs": inputs,
            "generation": {"seed": seed, "speed": 1.0, "sample_rate": 48000},
            "client_ref": ref}


def local_metrics(wav: bytes) -> dict:
    """PCM16 本地实测（旧缓存产物 meta 缺声学字段时兜底，非静音/削波仍可验）。"""
    import array
    import io
    import math
    import wave
    try:
        w = wave.open(io.BytesIO(wav))
        raw = w.readframes(w.getnframes())
        s = array.array("h", raw)
        if not s:
            return {}
        peak = max(abs(min(s)), max(s)) / 32768.0
        rms = math.sqrt(sum(x * x for x in s) / len(s)) / 32768.0
        return {"peak_dbfs": round(20 * math.log10(peak), 2) if peak > 0 else -120.0,
                "rms_dbfs": round(20 * math.log10(rms), 2) if rms > 0 else -120.0,
                "clipped_samples": sum(1 for x in s if abs(x) >= 32700)}
    except Exception:
        return {}


def tech_checks(d: dict, wav: bytes, expect_mode: str) -> list[str]:
    """§8「有音频、非静音、时长正常」技术验收 + §7 可剪辑干声项。"""
    problems = []
    o = d["outputs"][0]
    meta = dict(o.get("meta") or {})
    if "rms_dbfs" not in meta:
        meta = {**local_metrics(wav), **{k: v for k, v in meta.items() if v is not None}}
        meta["mode"] = o.get("meta", {}).get("mode")  # 模式仍需回执
    dur = o.get("duration_s")
    if not (dur and 0.3 < dur < 60):
        problems.append(f"时长异常 dur={dur}")
    b = wav
    if len(b) <= 44 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        problems.append("非法 WAV")
        return problems
    ch = int.from_bytes(b[22:24], "little")
    sr = int.from_bytes(b[24:28], "little")
    bits = int.from_bytes(b[34:36], "little")
    if ch != 1:
        problems.append(f"非单声道 ch={ch}")
    if sr != 48000:
        problems.append(f"采样率 {sr}!=48000")
    if bits != 16:
        problems.append(f"非 PCM16 bits={bits}")
    if meta.get("clipped_samples"):
        problems.append(f"削波 {meta['clipped_samples']} 样本")
    if meta.get("rms_dbfs", -120) < -55:
        problems.append(f"疑似静音 rms={meta.get('rms_dbfs')}dBFS")
    if meta.get("mode") != expect_mode:
        problems.append(f"实际模式 {meta.get('mode')} != {expect_mode}")
    if len(o.get("sha256") or "") != 64:
        problems.append("缺文件哈希")
    if meta.get("overflow"):
        problems.append(f"溢出 duration_limit（台词超时段，未截断仅报告）")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="demo-zh-f,demo-zh-m")
    ap.add_argument("--seed", type=int, default=42,
                    help="42=可复用缓存；验收新代码时换 seed 强制全量生成")
    ap.add_argument("--out", default="data/reports")
    args = ap.parse_args()
    H = {"Authorization": f"Bearer {token()}"}
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, batch_jobs = [], []
    for voice in voices:
        for i, line in enumerate(SCRIPT_LINES):
            for instructed in (False, True):
                ref = f"acc-{run}-{voice}-{i}-{'ab' if instructed else 'aa'}"
                batch_jobs.append((voice, i, line, instructed, ref,
                                   spec(voice, line, instructed, ref, args.seed)))
    # 一次批量提交：一条失败不影响其余（§6）
    b = requests.post(f"{BASE}/v1/jobs/batch", headers=H, timeout=30,
                      json={"jobs": [j[5] for j in batch_jobs],
                            "client_ref": f"acc-batch-{run}"})
    if b.status_code != 201:
        raise SystemExit(f"batch submit failed {b.status_code}: {b.text[:300]}")
    ids = b.json()["jobs"]
    print(f"batch {b.json()['batch_id']}: {len(ids)} jobs")

    deadline = time.time() + 1800
    results = {}
    for (voice, i, line, instructed, ref, _), jid in zip(batch_jobs, ids):
        d = requests.get(f"{BASE}/v1/jobs/{jid}", timeout=10).json()
        while d["status"] not in TERMINAL and time.time() < deadline:
            time.sleep(3)
            d = requests.get(f"{BASE}/v1/jobs/{jid}", timeout=10).json()
        results[(voice, i, instructed)] = (jid, d)

    md = [f"# cosyvoice.t2a 验收报告 {run}", "",
          f"- voices: {', '.join(voices)}  lines: {len(SCRIPT_LINES)}  "
          f"A/B: 默认 vs 指令（seed=42 speed=1.0 48kHz）", "",
          "| voice | line | mode | dur | rms dBFS | peak | clip | tech |",
          "|---|---|---|---|---|---|---|---|"]
    fails = 0
    for (voice, i, line, instructed, ref, _), jid in zip(batch_jobs, ids):
        jid, d = results[(voice, i, instructed)]
        if d["status"] != "completed":
            md.append(f"| {voice} | {i} | {'instruct2' if instructed else 'zero_shot'} "
                      f"| - | - | - | - | **FAILED {d['status']}** |")
            fails += 1
            continue
        wav = requests.get(BASE + d["outputs"][0]["url"], timeout=60).content
        p = out / f"acc_{run}_{voice}_{i}_{'instruct' if instructed else 'plain'}.wav"
        p.write_bytes(wav)
        problems = tech_checks(d, wav, "instruct2" if instructed else "zero_shot")
        m = d["outputs"][0].get("meta") or {}
        if m.get("rms_dbfs") is None:
            m = {**local_metrics(wav), **{k: v for k, v in m.items() if v is not None}}
        md.append(f"| {voice} | {i} | {m.get('mode')} | {m.get('duration_s')}s "
                  f"| {m.get('rms_dbfs')} | {m.get('peak_dbfs')} "
                  f"| {m.get('clipped_samples')} | {'PASS' if not problems else '**'+'; '.join(problems)+'**'} |")
        fails += bool(problems)
    md += ["", "## 人工听感检查清单（技术验收 ≠ 听感验收，§8）",
           "- [ ] 各角色跨句跨场景音色身份稳定（同 voice_id 全部条目像同一个人）",
           "- [ ] 同 seed 默认/指令 A/B：声线不变，仅表演（节奏/轻重/语气）变化",
           "- [ ] 短句无吞字、句尾完整、无突然截断；数字/专名读法正确",
           "- [ ] 普通对话与电话/正式场景的节奏差异符合剧情",
           "- [ ] 参考音频逐字稿与表演说明**没有**被念进任何产物",
           f"", f"产物目录: {out}/acc_{run}_*.wav"]
    report = out / f"acceptance_{run}.md"
    report.write_text("\n".join(md))
    print(f"report: {report}  tech-fails={fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
