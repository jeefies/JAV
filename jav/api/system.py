"""System API: capabilities / runtime / queue / health / voices."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from .. import capabilities, config, voices as voices_mod
from ..runtime.supervisor import mem_available_mb, vram_pids

router = APIRouter(prefix="/v1")

VERSION = "0.1.0"


@router.get("/capabilities")
async def get_capabilities():
    return capabilities.capabilities()


@router.get("/runtime")
async def get_runtime(request: Request):
    ctx = request.app.ctx
    sup = ctx.sup
    backend = sup.backend
    rss_mb = None
    if backend and backend.pid:
        try:
            with open(f"/proc/{backend.pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS"):
                        rss_mb = int(line.split()[1]) // 1024
                        break
        except OSError:
            pass
    kb = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            kb[k] = int(v.strip().split()[0]) // 1024
    # nvidia-smi subprocess can take 100ms-10s under driver contention; never
    # run it inline on the event loop (supervisor moves these probes off-loop).
    vram_by_pid = await asyncio.to_thread(vram_pids) if backend and backend.pid else None
    return {
        "active_profile": sup.active_profile,
        "state": sup.state,
        "pid": backend.pid if backend else None,
        "rss_mb": rss_mb,
        "mem": {"ram_available_mb": kb.get("MemAvailable", 0),
                "swap_free_mb": kb.get("SwapFree", 0),
                "admission_headroom_mb": mem_available_mb()},
        "vram": {"used_by_managed_pids_mb": (vram_by_pid or {}).get(backend.pid, 0)
                 if backend and backend.pid else 0},
        "streak": ctx.sched.streak,
        "last_switch": sup.last_switch,
        "events": ctx.store.recent_events(15),
    }


@router.post("/runtime/keepalive")
async def keepalive_runtime(request: Request, ttl_s: int | None = None):
    """Refresh the idle-unload timer to keep the active runtime resident.
    Optional ?ttl_s= overrides the remaining window for this touch."""
    ctx = request.app.ctx
    backend = ctx.sup.backend
    if not backend or not backend.alive():
        return {"kept_alive": None,
                "note": "no active runtime; next job will cold-start"}
    ctx.sup.job_finished()
    if ttl_s:
        # clamp both ways: a negative window would make stop_if_idle fire the
        # very next tick — the exact opposite of keepalive
        ctx.sup.keepalive_ttl_s = max(1, min(int(ttl_s), 7200))
    else:
        ctx.sup.keepalive_ttl_s = None
    profile = ctx.sup.profiles[ctx.sup.active_profile]
    window = ctx.sup.keepalive_ttl_s or profile.idle_unload_s
    return {"kept_alive": ctx.sup.active_profile,
            "state": ctx.sup.state, "idle_unload_in_s": window}


@router.post("/runtime/unload")
async def unload_runtime(request: Request):
    """Force-release the active runtime (RAM/VRAM) without waiting for idle."""
    ctx = request.app.ctx
    # Never SIGTERM the GPU process out from under a live job (documented
    # "tasks are never interrupted mid-flight" guarantee; explicit cancel
    # first if you really want the outputs dropped).
    busy = ctx.store.list_jobs(status="starting_runtime,running", limit=1)["total"]
    if busy:
        raise HTTPException(409, detail=f"{busy} job(s) in flight; cancel them first")
    prev = ctx.sup.active_profile
    await ctx.sup.shutdown("manual_unload")
    ctx.store.log_event(prev, None, "manual_unload")
    ctx.sched.wake()
    return {"unloaded": prev, **ctx.sched.snapshot()}


@router.get("/queue")
async def get_queue(request: Request):
    return request.app.ctx.sched.snapshot()


@router.get("/voices")
async def list_voices(request: Request):
    try:
        registry = voices_mod.load_voices()
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    return {"voices": voices_mod.public_view(registry)}


@router.post("/voices", status_code=201)
async def register_voice(request: Request):
    """注册/替换一个角色音色：prompt_asset 必须是已上传的 kind=audio 干声
    资产，prompt_text 为其逐字稿（CosyVoice3 zero-shot 的两个输入）。"""
    ctx = request.app.ctx
    body = await request.json()
    entry = {"id": str(body.get("id", "")).strip(),
             "name": str(body.get("name", "")).strip(),
             "asset": str(body.get("prompt_asset", "")).strip(),
             "prompt_text": str(body.get("prompt_text", "")).strip(),
             "description": str(body.get("description", "")).strip()}
    asset = ctx.store.get_asset(entry["asset"]) if entry["asset"] else None
    if asset is None:
        raise HTTPException(400, detail=f"unknown prompt_asset: {entry['asset']!r}")
    if asset["kind"] != "audio":
        raise HTTPException(400, detail=f"prompt_asset must be kind=audio, got {asset['kind']}")
    try:
        v = voices_mod.upsert_voice({k: v for k, v in entry.items() if k != "name" or v})
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    return {"id": v["id"], "name": v["name"], "registered": True,
            "source": "asset", "description": v["description"]}


@router.get("/health")
async def health(request: Request):
    ctx = request.app.ctx
    return {
        "status": "healthy", "service": "JAV", "version": VERSION,
        "auth": "bearer" if config.API_TOKEN else "disabled",
        "state": ctx.sup.state, "active_profile": ctx.sup.active_profile,
        "queued": ctx.store.queued_count(),
        "scheduler_running": not ctx.sched.stopped.is_set(),
    }
