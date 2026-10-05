"""Job + batch API (JAV-DESIGN 3.1-3.3, 5)."""
from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .. import capabilities, config, providers
from ..models import BatchSpec, JobSubmit
from ..store import TERMINAL
from ..providers import ProviderError

router = APIRouter(prefix="/v1")

FILE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
             ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
             ".mkv": "video/x-matroska", ".mp3": "audio/mpeg", ".wav": "audio/wav",
             ".flac": "audio/flac", ".ogg": "audio/ogg"}


def _public(ctx, job: dict, outs: list[dict] | None = None,
            qpos: int | None | bool = False) -> dict:
    if outs is None:
        outs = ctx.store.get_outputs(job["id"])
    if qpos is False:
        qpos = ctx.store.queue_position(job["id"])
    return {
        "id": job["id"], "provider": job["provider"], "workflow": job["workflow"],
        "runtime_profile": job["runtime_profile"], "status": job["status"],
        "batch_id": job["batch_id"], "client_ref": job["client_ref"],
        "created_at": job["created_at"], "started_at": job["started_at"],
        "finished_at": job["finished_at"], "error": job["error"],
        "error_type": job["error_type"], "retry_count": job["retry_count"],
        "queue_position": qpos,
        "outputs": [_public_output(job["id"], o) for o in outs],
    }


def _public_output(job_id: str, o: dict) -> dict:
    meta = o.get("asset_meta") or {}
    out = {"id": o["id"], "kind": o["kind"], "asset_id": o["asset_id"], "role": o["role"],
           "url": f"/v1/jobs/{job_id}/output?asset_id={o['id']}"}
    if o.get("asset_sha256"):
        out["sha256"] = o["asset_sha256"]
    if o.get("asset_size") is not None:
        out["size_bytes"] = o["asset_size"]
    dur = meta.get("duration_s")
    if dur is not None:
        out["duration_s"] = dur
    # 回执注明实际推理模式/表演参数/声学指标（wants.md §3/§7）：worker meta
    # 全量透出；provider 私有字段本就只含 audio 需要的内容。
    if o["kind"] == "audio" and meta:
        out["meta"] = meta
    return out


class _Prepared:
    __slots__ = ("profile", "payload", "slots", "cache_key")

    def __init__(self, profile, payload, slots, ck):
        self.profile, self.payload, self.slots, self.cache_key = profile, payload, slots, ck


def _prepare(ctx, req: JobSubmit) -> _Prepared:
    ok, reason = capabilities.is_available(req.provider, req.workflow)
    if not ok:
        raise HTTPException(415, detail=f"{req.provider}.{req.workflow} unavailable: {reason}")
    try:
        profile = providers.resolve_profile(req.provider, req.workflow)
        payload = providers.normalize(req.provider, req.workflow, req.inputs, req.generation)
    except ProviderError as e:
        raise HTTPException(e.status, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    slots = providers.asset_slots(payload)
    for slot, asset_id in slots.items():
        asset = ctx.store.get_asset(asset_id)
        if not asset:
            raise HTTPException(400, detail=f"unknown asset for {slot}: {asset_id}")
    return _Prepared(profile, payload, slots, providers.cache_key(payload))


def _enqueue(ctx, req: JobSubmit, pr: _Prepared, batch_id: str | None = None) -> dict:
    ck = pr.cache_key
    if ck:
        hit = ctx.store.find_cache_hit(ck)
        if hit:
            job = ctx.store.create_job(
                provider=req.provider, workflow=req.workflow, runtime_profile=pr.profile,
                payload={**pr.payload, "_cached_from": hit["id"]},
                assets=list(pr.slots.values()), cache_key=ck, status="completed",
                client_ref=req.client_ref, priority=req.priority, batch_id=batch_id)
            ctx.store.update_job(job["id"], started_at=job["created_at"],
                                 finished_at=job["created_at"])
            for o in ctx.store.get_outputs(hit["id"]):
                ctx.store.add_output(job["id"], o["kind"], o["asset_id"], o["path"], o["role"])
            return job

    job = ctx.store.create_job(
        provider=req.provider, workflow=req.workflow, runtime_profile=pr.profile,
        payload=pr.payload, assets=list(pr.slots.values()), cache_key=ck,
        client_ref=req.client_ref, priority=req.priority, batch_id=batch_id)
    return job


@router.post("/jobs", status_code=201)
async def create_job(req: JobSubmit, request: Request):
    ctx = request.app.ctx
    # wants.md §6 幂等：client_ref 是客户端请求编号；重复提交同一 ref 返回既有
    # job（不重复生成/计费），响应丢失后可凭 ref 复查是否已创建。
    if req.client_ref:
        existing = ctx.store.get_job_by_client_ref(req.client_ref)
        if existing:
            return JSONResponse(status_code=200,
                                content={**_public(ctx, existing), "idempotent_replay": True})
    if ctx.store.queued_count() >= config.QUEUE_DEPTH_LIMIT:
        raise HTTPException(429, detail="queue depth limit reached")
    pr = _prepare(ctx, req)
    job = _enqueue(ctx, req, pr)
    ctx.sched.wake()
    return _public(ctx, ctx.store.get_job(job["id"]))


@router.post("/jobs/batch", status_code=201)
async def create_batch(req: BatchSpec, request: Request):
    ctx = request.app.ctx
    if not req.jobs:
        raise HTTPException(400, detail="empty batch")
    # 批量幂等（wants.md §6）：同一请求编号的 batch 重复提交返回既有批次。
    if req.client_ref:
        existing = ctx.store.get_batch_by_client_ref(req.client_ref)
        if existing:
            return JSONResponse(status_code=200, content={
                "batch_id": existing["id"], "jobs": existing["jobs"],
                "counts": existing["counts"], "idempotent_replay": True})
    if len(req.jobs) > config.BATCH_MAX_JOBS:
        raise HTTPException(400, detail=f"batch exceeds {config.BATCH_MAX_JOBS} jobs")
    if ctx.store.queued_count() + len(req.jobs) > config.QUEUE_DEPTH_LIMIT:
        raise HTTPException(429, detail="queue depth limit would be exceeded")
    # validate ALL first (atomicity), then enqueue
    prepared, errors = [], {}
    for i, raw in enumerate(req.jobs):
        try:
            sub = JobSubmit(**req.merged(raw))
            prepared.append((sub, _prepare(ctx, sub)))
        except HTTPException as e:
            errors[i] = e.detail
        except Exception as e:
            errors[i] = str(e)
    if errors:
        raise HTTPException(422, detail={"invalid_jobs": errors})
    batch = ctx.store.create_batch(req.client_ref, {"size": len(prepared)})
    jobs = [_enqueue(ctx, sub, pr, batch_id=batch["id"]) for sub, pr in prepared]
    ctx.sched.wake()
    return {"batch_id": batch["id"], "jobs": [j["id"] for j in jobs],
            "queue_positions": {j["id"]: ctx.store.queue_position(j["id"]) for j in jobs}}


@router.get("/jobs")
async def list_jobs(request: Request, status: str | None = None,
                    runtime_profile: str | None = None, batch_id: str | None = None,
                    provider: str | None = None, client_ref: str | None = None,
                    limit: int = 100, offset: int = 0):
    ctx = request.app.ctx
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    res = ctx.store.list_jobs(status=status, runtime_profile=runtime_profile,
                              batch_id=batch_id, provider=provider,
                              client_ref=client_ref, limit=limit, offset=offset)
    ids = [j["id"] for j in res["jobs"]]
    outs_map = ctx.store.outputs_for(ids)
    qpos_map = ctx.store.queue_positions_for(ids)
    return {"total": res["total"],
            "jobs": [_public(ctx, j, outs_map.get(j["id"], []),
                             qpos_map.get(j["id"], None)) for j in res["jobs"]]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    ctx = request.app.ctx
    job = ctx.store.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return _public(ctx, job)


async def _cancel_one(ctx, jid: str) -> tuple[str, bool]:
    """-> (status, already_terminal). queued -> immediate cancelled; in-flight
    -> persisted soft cancel that the scheduler finalizes at its next
    checkpoint (never silently dropped). already_terminal distinguishes a job
    that was/just became terminal (caller: 409) from a freshly accepted cancel."""
    job = ctx.store.get_job(jid)
    if not job:
        return "not_found", False
    if job["status"] in TERMINAL:
        return job["status"], True
    if ctx.store.cancel_queued(jid):
        ctx.bus.publish(jid, {"status": "cancelled"})
        return "cancelled", False
    ctx.store.request_cancel(jid)
    if job["status"] == "running" and ctx.sup.backend:
        await ctx.sup.backend.cancel(jid)
    return "cancelling", False


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    ctx = request.app.ctx
    job = ctx.store.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    if job["status"] in TERMINAL:
        raise HTTPException(409, detail=f"job already {job['status']}")
    result, already_terminal = await _cancel_one(ctx, job_id)
    if result == "not_found":
        raise HTTPException(404, detail="job not found")
    # already_terminal: raced to a terminal state between the pre-check and the
    # re-read (completed/failed/cancelled) -> genuine 409. A freshly accepted
    # queued-cancel returns ("cancelled", False) and MUST answer 200.
    if already_terminal:
        raise HTTPException(409, detail=f"job already {result}")
    return {"id": job_id, "status": result}


@router.get("/jobs/{job_id}/outputs")
async def job_outputs(job_id: str, request: Request):
    ctx = request.app.ctx
    if not ctx.store.get_job(job_id):
        raise HTTPException(404, detail="job not found")
    return {"job_id": job_id, "outputs": [
        {**o, "url": f"/v1/jobs/{job_id}/output?asset_id={o['id']}"}
        for o in ctx.store.get_outputs(job_id)]}


@router.get("/jobs/{job_id}/output")
async def job_output_file(job_id: str, request: Request, asset_id: str | None = None):
    ctx = request.app.ctx
    if not ctx.store.get_job(job_id):
        raise HTTPException(404, detail="job not found")
    outs = ctx.store.get_outputs(job_id)
    if asset_id:
        outs = [o for o in outs if o["id"] == asset_id]
    if not outs:
        raise HTTPException(404, detail="output not found")
    o = outs[0]
    p = Path(o["path"])
    if not p.exists():
        raise HTTPException(410, detail="output file missing (purged?)")
    media = FILE_MIME.get(p.suffix.lower()) or \
        {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}.get(o["kind"])
    return FileResponse(str(p), media_type=media, filename=p.name)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    ctx = request.app.ctx
    job = ctx.store.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")

    async def stream():
        q = ctx.bus.subscribe(job_id)
        try:
            cur = ctx.store.get_job(job_id)
            yield f"event: status\ndata: {json.dumps({'status': cur['status']})}\n\n"
            if cur["status"] in TERMINAL:
                return
            waited = 0.0
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=5)
                except asyncio.TimeoutError:
                    waited += 5
                    if waited > 1800:
                        break
                    yield ": keepalive\n\n"
                    continue
                waited = 0.0
                yield f"event: status\ndata: {json.dumps(ev, default=str)}\n\n"
                if ev.get("status") in TERMINAL:
                    break
        finally:
            ctx.bus.unsubscribe(job_id, q)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str, request: Request):
    ctx = request.app.ctx
    b = ctx.store.get_batch(batch_id)
    if not b:
        raise HTTPException(404, detail="batch not found")
    return b


@router.delete("/batches/{batch_id}")
async def cancel_batch(batch_id: str, request: Request):
    ctx = request.app.ctx
    b = ctx.store.get_batch(batch_id)
    if not b:
        raise HTTPException(404, detail="batch not found")
    result = {}
    for jid in b["jobs"]:
        result[jid], _ = await _cancel_one(ctx, jid)
    return {"batch_id": batch_id, "result": result}


# ---------------- internal worker callbacks ----------------
# Shared-secret gated: only the supervisor-spawned worker knows the value
# (env JAV_CALLBACK_SECRET). These endpoints read/move files on behalf of a
# job, so they must never be callable by API clients — including anyone
# reaching this port through a tunnel.
def _check_callback(request: Request):
    if not hmac.compare_digest(request.headers.get("x-jav-callback", ""),
                               config.CALLBACK_SECRET):
        raise HTTPException(403, detail="invalid worker callback secret")


@router.post("/internal/task_complete")
async def task_complete(request: Request):
    _check_callback(request)
    ctx = request.app.ctx
    payload = await request.json()
    return {"handled": ctx.sup.deliver(payload)}


@router.post("/internal/pipeline_status")
async def pipeline_status(request: Request):
    _check_callback(request)
    ctx = request.app.ctx
    payload = await request.json()
    return {"handled": ctx.sup.pipeline_status(payload)}
