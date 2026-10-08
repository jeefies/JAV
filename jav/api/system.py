"""System API: capabilities / runtime / queue / health / voices."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

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

    def _rss(pid):
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) // 1024
        except OSError:
            pass
        return None

    rss_mb = _rss(backend.pid) if backend and backend.pid else None
    kb = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            kb[k] = int(v.strip().split()[0]) // 1024
    # nvidia-smi subprocess can take 100ms-10s under driver contention; never
    # run it inline on the event loop (supervisor moves these probes off-loop).
    vram_by_pid = await asyncio.to_thread(vram_pids) if backend and backend.pid else None
    cpu_b = sup.cpu_backend
    cpu_rss_mb = _rss(cpu_b.pid) if cpu_b and cpu_b.pid else None
    return {
        "active_profile": sup.active_profile,
        "state": sup.state,
        "cpu_active_profile": sup.cpu_active_profile,
        "cpu_state": sup.cpu_state,
        "cpu_pid": cpu_b.pid if cpu_b else None,
        "cpu_rss_mb": cpu_rss_mb,
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
    prev = ctx.sup.active_profile or ctx.sup.cpu_active_profile
    await ctx.sup.shutdown("manual_unload", lane="both")
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


def _reference_probe(ctx, v: dict) -> dict | None:
    """Audition-file facts (duration/sr/channels/size/sha) without leaking paths."""
    path = None
    facts: dict = {}
    if v["asset"]:
        asset = ctx.store.get_asset(v["asset"])
        if not asset:
            return None
        path = asset["path"]
        facts.update({"bytes": asset["size"], "sha256": asset["sha256"],
                      "asset_id": v["asset"]})
        raw_meta = asset.get("meta")
        if isinstance(raw_meta, str):
            try:
                import json
                raw_meta = json.loads(raw_meta)
            except Exception:
                raw_meta = None
        dur = (raw_meta or {}).get("duration_s")
        if dur is not None:
            facts["duration_s"] = dur
    else:
        path = v["path"]
    if not path or not Path(path).exists():
        return {**facts, "available": False} if facts else None
    try:
        import soundfile as sf
        info = sf.info(path)
        facts.setdefault("duration_s", round(info.duration, 3))
        facts["sample_rate"] = info.samplerate
        facts["channels"] = info.channels
        facts["format"] = info.format
    except Exception:
        facts.setdefault("duration_s", None)
    try:
        facts["bytes"] = Path(path).stat().st_size
    except OSError:
        pass
    facts["available"] = True
    return facts


@router.get("/voices/{voice_id}")
async def voice_detail(voice_id: str, request: Request):
    """音色详情与版本历史：不含路径/逐字稿。"""
    ctx = request.app.ctx
    try:
        registry = voices_mod.load_voices()
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    v = registry.get(voice_id)
    if v is None:
        raise HTTPException(404, detail="voice not found")
    return {**voices_mod.detail_view(v), "reference": _reference_probe(ctx, v),
            "preview_url": f"/v1/voices/{voice_id}/preview"}


@router.post("/voices", status_code=201)
async def register_voice(request: Request):
    """注册一个角色音色（重复注册默认 409，不默默覆盖；
    显式 replace=true 替换并把旧版记入 history，version+1）。

    prompt_asset 必须是已上传的 kind=audio 干声资产，prompt_text 为其逐字稿
    （CosyVoice3 zero-shot 的两个输入）。kind/license/provenance 记录来源与
    许可；model 声明兼容模型，与服务部署不符时 422 拒绝（不静默入库）。"""
    ctx = request.app.ctx
    body = await request.json()
    model = str(body.get("model", "")).strip()
    if model and not model.startswith(config.COSYVOICE_MODEL_NAME.split("-")[0]):
        raise HTTPException(422, detail=(
            f"voice declares model {model!r}, this deployment runs "
            f"{config.COSYVOICE_MODEL_NAME!r}; register it on a compatible service"))
    entry = {"id": str(body.get("id", "")).strip(),
             "name": str(body.get("name", "")).strip(),
             "asset": str(body.get("prompt_asset", "")).strip(),
             "prompt_text": str(body.get("prompt_text", "")).strip(),
             "description": str(body.get("description", "")).strip(),
             "kind": str(body.get("kind", "local")).strip().lower(),
             "role": str(body.get("role", "")).strip(),
             "tags": body.get("tags") or [],
             "license": str(body.get("license", "")).strip(),
             "provenance": str(body.get("provenance", "")).strip(),
             "model": model or config.COSYVOICE_MODEL_NAME}
    if not isinstance(entry["tags"], list):
        raise HTTPException(400, detail="tags must be a list of strings")
    asset = ctx.store.get_asset(entry["asset"]) if entry["asset"] else None
    if asset is None:
        raise HTTPException(400, detail=f"unknown prompt_asset: {entry['asset']!r}")
    if asset["kind"] != "audio":
        raise HTTPException(400, detail=f"prompt_asset must be kind=audio, got {asset['kind']}")
    try:
        v, created = voices_mod.register_voice(
            {k: val for k, val in entry.items() if k != "name" or val},
            replace=bool(body.get("replace")), note=str(body.get("note", "")))
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    return JSONResponse(status_code=201 if created else 200, content={
        "id": v["id"], "name": v["name"], "registered": True,
        "replaced": not created, "version": v["version"],
        "source": "asset", "kind": v["kind"], "description": v["description"],
        "model": v["model"]})


_SAMPLE_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                ".ogg": "audio/ogg", ".m4a": "audio/mp4"}


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, request: Request):
    """删除注册（历史产物与资产不受影响；后续引用该 id 的任务提交即报错）。"""
    try:
        voices_mod.delete_voice(voice_id)
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    return {"id": voice_id, "deleted": True}


@router.get("/voices/{voice_id}/preview")
async def voice_preview(voice_id: str, request: Request):
    """参考干声试听：前端音色表/文档用它播放，不暴露路径（只按注册表解析）。"""
    ctx = request.app.ctx
    try:
        registry = voices_mod.load_voices()
    except voices_mod.VoiceError as e:
        raise HTTPException(e.status, detail=str(e))
    v = registry.get(voice_id)
    if v is None:
        raise HTTPException(404, detail="voice not found")
    path = None
    if v["asset"]:
        asset = ctx.store.get_asset(v["asset"])
        path = asset["path"] if asset else None
    else:
        path = v["path"]
    if not path or not Path(path).exists():
        raise HTTPException(410, detail="reference audio file missing")
    media = _SAMPLE_MIME.get(Path(path).suffix.lower()) or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=Path(path).name)


# 旧名 /sample 保留为别名（已上线的调用方不断档）
@router.get("/voices/{voice_id}/sample")
async def voice_sample(voice_id: str, request: Request):
    return await voice_preview(voice_id, request)


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


@router.get("/openapi.json")
async def openapi_json(request: Request):
    """Public OpenAPI 3.1 contract (internal endpoints stripped, per-process cache)."""
    from .. import openapi as openapi_mod
    return JSONResponse(openapi_mod.cached_spec(request.app))
