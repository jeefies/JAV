"""Batch first-validation: run several specs through the real production
code path (Store+Supervisor+ComfyBackend) in ONE process so the ComfyUI
cold start is paid once for the whole batch. Marks each validated capability.

usage: python3 tools/batch_first_validation.py spec1.json spec2.json ...
  each spec: {"provider":"ltx25","workflow":"union_control",
              "inputs":{... absolute paths auto-uploaded ...},
              "generation":{...}}
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/data/AV/JAV")

from first_validation import _guard_single_runtime  # reuse the anti-dual-runtime lock

from jav import capabilities, providers
from jav.runtime.supervisor import Supervisor
from jav.scheduler import Scheduler
from jav.store import Store

OUT = Path("/tmp/kilo/batch_val")


async def run_one(store, sched, sup, spec):
    provider, workflow = spec["provider"], spec["workflow"]

    def _upload(p: str) -> str:
        import hashlib
        import os as _os
        from jav import config
        from jav.models import EXT_KIND
        data = Path(p).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        ext = _os.path.splitext(p)[1] or ".bin"
        kind = EXT_KIND.get(ext.lower(), "file")
        d = config.ASSETS_DIR / sha[:2]
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{sha[:24]}{ext}"
        if not dest.exists():
            dest.write_bytes(data)
        return store.put_asset(sha256=sha, kind=kind, path=str(dest),
                               size=len(data))["id"]

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("/") and os.path.exists(obj):
            return _upload(obj)
        return obj

    spec = {**spec, "inputs": _walk(dict(spec["inputs"]))}
    profile = providers.resolve_profile(provider, workflow)
    payload = providers.normalize(provider, workflow, spec["inputs"],
                                  spec.get("generation", {}))
    job = store.create_job(provider=provider, workflow=workflow,
                           runtime_profile=profile, payload=payload,
                           assets=list(payload["assets"].values()),
                           client_ref=f"batch-validation-{workflow}")
    jid = job["id"]
    print(f"enqueued {provider}.{workflow} -> {jid}", flush=True)
    claimed = store.claim_next(profile)
    assert claimed and claimed["id"] == jid, "claim mismatch (queue not empty?)"
    t0 = time.time()
    await sched.run_job(claimed)
    done = store.get_job(jid)
    el = time.time() - t0
    status = done["status"]
    if status == "completed":
        capabilities.mark_validated(provider, workflow)
        for o in store.get_outputs(jid):
            OUT.mkdir(parents=True, exist_ok=True)
            tgt = OUT / f"{provider}_{workflow}{Path(o['path']).suffix}"
            tgt.write_bytes(Path(o["path"]).read_bytes())
        print(f"COMPLETED {provider}.{workflow} in {el:.0f}s -> {tgt}", flush=True)
        return True
    print(f"{status.upper()} {provider}.{workflow} after {el:.0f}s: "
          f"{done.get('error')}", flush=True)
    return False


async def main():
    specs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    from jav import config
    config.ensure_dirs()
    _guard_single_runtime()
    store = Store()
    sup = Supervisor(store)
    sched = Scheduler(store, sup)
    results = {}
    try:
        for spec in specs:
            key = f"{spec['provider']}.{spec['workflow']}" \
                  + (f"[{spec['inputs'].get('mode','reference')}]"
                     if spec.get("workflow") == "ic_lora" else "")
            results[key] = await run_one(store, sched, sup, spec)
    finally:
        await sup.shutdown("batch_done")
    print("BATCH_RESULT", json.dumps(results), flush=True)
    if not all(results.values()):
        sys.exit(1)
    print("ALL_VALIDATED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
