"""System API: capabilities / runtime / queue / health."""
from __future__ import annotations

from fastapi import APIRouter, Request

from .. import capabilities, config
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
    return {
        "active_profile": sup.active_profile,
        "state": sup.state,
        "pid": backend.pid if backend else None,
        "rss_mb": rss_mb,
        "mem": {"ram_available_mb": kb.get("MemAvailable", 0),
                "swap_free_mb": kb.get("SwapFree", 0),
                "admission_headroom_mb": mem_available_mb()},
        "vram": {"used_by_managed_pids_mb":
                 (vram_pids() or {}).get(backend.pid, 0) if backend and backend.pid else 0},
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
        ctx.sup.keepalive_ttl_s = min(int(ttl_s), 7200)
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
    prev = ctx.sup.active_profile
    await ctx.sup.shutdown("manual_unload")
    ctx.store.log_event(prev, None, "manual_unload")
    ctx.sched.wake()
    return {"unloaded": prev, **ctx.sched.snapshot()}


@router.get("/queue")
async def get_queue(request: Request):
    return request.app.ctx.sched.snapshot()


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
