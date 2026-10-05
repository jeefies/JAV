#!/usr/bin/env python3
"""Live smoke for cosyvoice.t2a: registers nothing, runs 3 production-shaped
lines through /v1/jobs, downloads the wavs, validates 48kHz mono PCM +
duration metadata. Run while JAV is idle (single runtime mutex!).

usage: JAV_API_TOKEN=... python3 tools/cosyvoice_smoke.py [--out DIR]
(token falls back to ~/.config/jav/env)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.getenv("JAV_BASE_URL", "http://127.0.0.1:8765")
TERMINAL = ("completed", "failed", "cancelled")


def token() -> str:
    t = os.getenv("JAV_API_TOKEN", "")
    if t:
        return t
    env = Path.home() / ".config/jav/env"
    for line in env.read_text().splitlines():
        if line.startswith("JAV_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no JAV_API_TOKEN (env or ~/.config/jav/env)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/kilo/cv3_smoke")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = {"Authorization": f"Bearer {token()}"}

    cases = [
        ("plain", {"provider": "cosyvoice", "workflow": "t2a",
                   "inputs": {"text": "明天上午九点，老地方见。", "voice_id": "demo-zh-f"},
                   "generation": {"seed": 42}, "client_ref": "cv3-smoke-plain"}),
        ("instruct", {"provider": "cosyvoice", "workflow": "t2a",
                      "inputs": {"text": "但不会公开来说。", "voice_id": "demo-zh-f",
                                 "instruction": "用平常的语气说，克制，带一点疲惫"},
                      "generation": {"seed": 7}, "client_ref": "cv3-smoke-instruct"}),
        ("speed-inline", {"provider": "cosyvoice", "workflow": "t2a",
                          "inputs": {"text": "你好，测试语速与内联参考音频。",
                                     "reference_audio": None, "reference_text":
                                     "希望你以后能够做的比我还好呦。"},
                          "generation": {"speed": 0.9, "seed": 11},
                          "client_ref": "cv3-smoke-inline"}),
    ]
    ref_wav = Path("/mnt/data/AV/CosyVoice/asset/zero_shot_prompt.wav")
    up = requests.post(f"{BASE}/v1/assets?kind=audio", data=ref_wav.read_bytes(),
                       headers={**H, "x-filename": ref_wav.name}, timeout=30)
    up.raise_for_status()
    cases[2][1]["inputs"]["reference_audio"] = up.json()["id"]

    jobs = []
    for tag, body in cases:
        r = requests.post(f"{BASE}/v1/jobs", json=body, headers=H, timeout=30)
        if r.status_code not in (200, 201):   # 200 = client_ref 幂等重放
            raise SystemExit(f"[{tag}] submit failed {r.status_code}: {r.text[:200]}")
        jobs.append((tag, r.json()["id"]))
        print(f"[{tag}] queued {r.json()['id']}")

    fails = 0
    for tag, jid in jobs:
        deadline = time.time() + 420
        st = None
        while time.time() < deadline:
            d = requests.get(f"{BASE}/v1/jobs/{jid}", timeout=10).json()
            st = d["status"]
            if st in TERMINAL:
                break
            time.sleep(2)
        d = requests.get(f"{BASE}/v1/jobs/{jid}", timeout=10).json()
        if d["status"] != "completed":
            print(f"[{tag}] FAIL {d['status']} error={d.get('error')}")
            fails += 1
            continue
        o = d["outputs"][0]
        wav = requests.get(BASE + o["url"], timeout=30)
        p = out / f"{tag}.wav"
        p.write_bytes(wav.content)
        dur = o.get("duration_s")
        # RIFF sanity: fmt sr/channels at fixed offsets for PCM mono
        b = wav.content
        sr = int.from_bytes(b[24:28], "little") if len(b) > 28 else 0
        ch = int.from_bytes(b[22:24], "little") if len(b) > 24 else 0
        ok = (dur or 0) > 0 and sr == 48000 and ch == 1 and len(b) > 44
        print(f"[{tag}] done dur={dur}s file={p} sr={sr} ch={ch} bytes={len(b)} "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            fails += 1
    if fails:
        sys.exit(1)
    print("COSYVOICE-SMOKE-PASS")


if __name__ == "__main__":
    main()
