"""MH3 fl2va real smoke, production path (same pattern as ltx25_smoke.py).

usage: python3 tools/mh3_smoke.py [workflow t2v|i2v] [wait_s]
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
    wf = sys.argv[1] if len(sys.argv) > 1 else "t2v"
    horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
    store = Store()
    inputs = {"prompt": "A paper boat drifting down a quiet stream at dusk, slow camera."}
    generation = {"duration": 2, "fps": 24, "seed": 7}
    if wf in ("i2v", "fl2v"):
        aid = json.loads(urllib.request.urlopen(urllib.request.Request(
            BASE + "/v1/assets?kind=image", data=open("/tmp/kilo/i2i_input.png", "rb").read(),
            headers={"x-filename": "mh3_seed.png", "content-type": "image/png"}), timeout=60).read())["id"]
        inputs["first_frame"] = aid
    if wf == "fl2v":
        inputs["last_frame"] = inputs["first_frame"]
    payload = providers.normalize("mh3", wf, inputs, generation)
    job = store.create_job(provider="mh3", workflow=wf, runtime_profile="mh3.fl2va",
                           payload=payload, assets=list(payload["assets"].values()),
                           client_ref="mh3-real-smoke")
    jid = job["id"]
    print(f"enqueued {jid}", flush=True)
    last = None
    deadline = time.time() + horizon
    while time.time() < deadline:
        d = get(f"/v1/jobs/{jid}")
        if d["status"] != last:
            rt = get("/v1/runtime")
            print(f"[{time.strftime('%H:%M:%S')}] {d['status']} runtime={rt['state']}/{rt['active_profile']} "
                  f"headroom={rt['mem']['admission_headroom_mb']}MB", flush=True)
            last = d["status"]
        if d["status"] == "completed":
            with urllib.request.urlopen(BASE + d["outputs"][0]["url"], timeout=180) as f:
                data = f.read()
            open(f"/tmp/kilo/mh3_{wf}_smoke.mp4", "wb").write(data)
            print(f"VIDEO_SAVED {len(data)} bytes -> /tmp/kilo/mh3_{wf}_smoke.mp4", flush=True)
            caps = get("/v1/capabilities")
            print("caps mh3:", json.dumps({k: v["available"] for k, v in
                                           caps["mh3"]["workflows"].items()}), flush=True)
            break
        if d["status"] in ("failed", "cancelled"):
            print("FAILED:", (d.get("error") or "")[:800], flush=True)
            sys.exit(1)
        time.sleep(15)
    else:
        print("SMOKE_TIMEOUT", flush=True)
        sys.exit(2)
    print("MH3_SMOKE_OK", flush=True)
