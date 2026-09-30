"""One-shot first validation: run a single job through the REAL production
code path (Store + Supervisor + ComfyBackend) in-process, without requiring
the HTTP service to be running. On success marks the capability validated.

usage: python3 tools/first_validation.py <provider> <workflow> <json-spec-file>
  spec: {"inputs": {...}, "generation": {...}, "assets": {slot: "/path.png"}}
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/data/AV/JAV")

from jav import capabilities, providers
from jav.runtime.supervisor import Supervisor
from jav.scheduler import Scheduler
from jav.store import Store


async def main():
    provider, workflow, spec_file = sys.argv[1], sys.argv[2], sys.argv[3]
    store = Store()
    spec = json.load(open(spec_file))

    def _upload(p: str) -> str:
        data = open(p, "rb").read()
        import hashlib
        import os as _os
        from jav import config
        sha = hashlib.sha256(data).hexdigest()
        ext = _os.path.splitext(p)[1] or ".bin"
        kind = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
                ".mp4": "video", ".mov": "video", ".wav": "audio", ".mp3": "audio"}.get(ext, "file")
        d = config.ASSETS_DIR / sha[:2]
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{sha[:24]}{ext}"
        if not dest.exists():
            dest.write_bytes(data)
        return store.put_asset(sha256=sha, kind=kind, path=str(dest), size=len(data))["id"]

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("/") and os.path.exists(obj):
            return _upload(obj)
        return obj

    spec["inputs"] = _walk(spec["inputs"])

    for stale in store.list_jobs(status="queued,starting_runtime",
                                 provider=provider, limit=50)["jobs"]:
        store.set_status(stale["id"], "cancelled", error="superseded by first_validation")

    sup = Supervisor(store)
    sched = Scheduler(store, sup)
    profile = providers.resolve_profile(provider, workflow)
    payload = providers.normalize(provider, workflow, spec["inputs"], spec["generation"])
    job = store.create_job(provider=provider, workflow=workflow,
                           runtime_profile=profile, payload=payload,
                           assets=list(payload["assets"].values()),
                           client_ref=f"first-validation-{workflow}")
    jid = job["id"]
    print(f"enqueued {jid} profile={profile}", flush=True)

    claimed = store.claim_next(profile)
    assert claimed and claimed["id"] == jid
    t0 = time.time()
    await sched.run_job(claimed)
    done = store.get_job(jid)
    el = time.time() - t0
    if done["status"] == "completed":
        print(f"COMPLETED in {el:.0f}s", flush=True)
        for o in store.get_outputs(jid):
            ext = os.path.splitext(o["path"])[1]
            out = f"/tmp/kilo/{provider}_{workflow}_validation{ext}"
            with open(o["path"], "rb") as src, open(out, "wb") as dst:
                dst.write(src.read())
            print("output:", out, flush=True)
        caps = capabilities.capabilities()
        print("caps now:", json.dumps(
            {k: v["available"] for k, v in caps[provider]["workflows"].items()}), flush=True)
        print("FIRST_VALIDATION_OK", flush=True)
    else:
        print(f"{done['status']} after {el:.0f}s:", done.get("error"), flush=True)
    await sup.shutdown("validation_done")
    if done["status"] != "completed":
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
