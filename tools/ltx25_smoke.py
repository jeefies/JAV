"""LTX-2.5 Fast real smoke, production path:

Enqueue a tiny ltx25.t2v job directly into the live JAV SQLite store
(bypassing the API capability gate ONCE — the very validation run that
earns the capability flag). The live service's own scheduler claims it,
boots ComfyUI via the supervisor, generates, and — on success — marks
ltx25.t2v validated so the public API opens it for everyone.

usage: python3 tools/ltx25_smoke.py [wait_s]
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/mnt/data/AV/JAV")

from jav import providers
from jav.store import Store

BASE = "http://127.0.0.1:8765"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as f:
        return json.loads(f.read())


if __name__ == "__main__":
    horizon = float(sys.argv[1]) if len(sys.argv) > 1 else 2400
    store = Store()
    payload = providers.normalize("ltx25", "t2v",
                                  {"prompt": "A small red ball bounces once on a green field, static camera."},
                                  {"width": 512, "height": 288, "duration": 1, "fps": 24, "seed": 7})
    job = store.create_job(provider="ltx25", workflow="t2v", runtime_profile="ltx25",
                           payload=payload, assets=list(payload["assets"].values()),
                           client_ref="ltx25-real-smoke")
    jid = job["id"]
    print(f"enqueued {jid} (live scheduler picks it up within ~2s)", flush=True)
    last = None
    deadline = time.time() + horizon
    while time.time() < deadline:
        d = get(f"/v1/jobs/{jid}")
        if d["status"] != last:
            rt = get("/v1/runtime")
            print(f"[{time.strftime('%H:%M:%S')}] {d['status']} "
                  f"runtime={rt['state']}/{rt['active_profile']} rss={rt['rss_mb']}MB "
                  f"headroom={rt['mem']['admission_headroom_mb']}MB "
                  f"vram={rt['vram']['used_by_managed_pids_mb']}MB", flush=True)
            last = d["status"]
        if d["status"] == "completed":
            url = d["outputs"][0]["url"]
            with urllib.request.urlopen(BASE + url, timeout=180) as f:
                data = f.read()
            open("/tmp/kilo/ltx25_smoke.mp4", "wb").write(data)
            print(f"VIDEO_SAVED {len(data)} bytes", flush=True)
            caps = get("/v1/capabilities")
            print("caps t2v now:", json.dumps(caps["ltx25"]["workflows"]["t2v"]), flush=True)
            break
        if d["status"] in ("failed", "cancelled"):
            print("FAILED:", (d.get("error") or "")[:800], flush=True)
            sys.exit(1)
        time.sleep(10)
    else:
        print("SMOKE_TIMEOUT", flush=True)
        sys.exit(2)
    print("LTX25_SMOKE_OK", flush=True)
