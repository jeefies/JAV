"""Scheduler: one persistent SQLite queue + runtime_profile affinity with
anti-starvation, RAM admission backoff, idle unload, crash retry (JAV-DESIGN 4)."""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
from datetime import datetime
from datetime import timezone as datetime_timezone
from pathlib import Path

from . import capabilities, config, providers
from .models import EXT_KIND
from .runtime.base import BackendCrash
from .runtime.supervisor import AdmissionDenied


class EventBus:
    def __init__(self):
        self._subs: dict[str, set] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        self._subs.get(job_id, set()).discard(q)

    def publish(self, job_id: str, event: dict):
        for q in list(self._subs.get(job_id, set())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


def _age_s(iso_ts: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime_timezone.utc)
        return (datetime.now(datetime_timezone.utc) - dt).total_seconds()
    except Exception:
        return 0.0


RETRYABLE_ERRORS = {"runtime_crash", "timeout", "start_error",
                    "internal_error", "ingest_error", "oom_error"}

# GPU 被其他家族占用时，这些 profile 的排队任务改走 CPU 兜底通道并行出片。
# （值 = profile.name，其 Profile.gpu 必须为 False；supervisor.ensure 按此路由。）
CPU_FALLBACK_ROUTES = {"cosyvoice": "cosyvoice-cpu"}


class Scheduler:
    def __init__(self, store, supervisor, bus: EventBus | None = None,
                 max_same=None, max_other_wait=None):
        self.store = store
        self.sup = supervisor
        self.bus = bus or EventBus()
        self.max_same = config.MAX_SAME_RUNTIME_JOBS if max_same is None else max_same
        self.max_other_wait = config.MAX_OTHER_WAIT_S if max_other_wait is None else max_other_wait
        self.streak = 0
        self.last_profile: str | None = None
        self._backoff: dict[str, float] = {}
        self._backoff_attempts: dict[str, int] = {}
        self._wake = asyncio.Event()
        self._wake_cpu = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._cpu_task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

    def wake(self):
        self._wake.set()
        self._wake_cpu.set()

    async def start(self):
        self._task = asyncio.create_task(self._loop(), name="jav-scheduler")
        self._cpu_task = asyncio.create_task(self._cpu_loop(), name="jav-scheduler-cpu")

    async def _loop(self):
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                while True:
                    progress = await self.process_once()
                    if not progress:
                        break
            except Exception as e:  # scheduler must never die
                self.store.log_event(None, None, f"scheduler_error: {e}", ok=False)
            await self.sup.stop_if_idle()
        if self.sup.backend or self.sup.cpu_backend:
            await self.sup.shutdown("exit", lane="both")

    async def stop(self):
        self.stopped.set()
        self.wake()
        for t in (self._task, self._cpu_task):
            if t:
                try:
                    await asyncio.wait_for(t, timeout=90)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    t.cancel()

    async def _cpu_loop(self):
        """GPU 被占用时把 cosyvoice 排队任务引到 CPU 通道，与渲染并行。"""
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(self._wake_cpu.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._wake_cpu.clear()
            try:
                progressed = await self.process_cpu_once()
            except Exception as e:  # the fallback lane must never kill the loop
                self.store.log_event(None, None, f"cpu_scheduler_error: {e}", ok=False)
                progressed = False
            await self.sup.stop_if_idle(lane="cpu")
            if not progressed:
                await asyncio.sleep(0)

    def _cpu_target(self) -> dict | None:
        """路由条件：GPU 通道**当下接不了**该家族任务——被别的家族占用，或该
        profile 正准入退避——且队列里有它；GPU 空闲或已亲自跑着 cosyvoice 时
        保持安静（快通道优先，也避免双 worker 双倍吃 RAM）。"""
        now_ts = time.monotonic()
        active = self.sup.active_profile
        for src, cpu_profile in CPU_FALLBACK_ROUTES.items():
            if self._backoff.get(cpu_profile, 0) > now_ts:
                continue
            if active is None and self._backoff.get(src, 0) <= now_ts:
                continue  # GPU 空转：主循环会立刻认领队首，轮不到兜底通道
            if active == src:
                continue  # GPU 正在跑同一家族：worker 已在，直接排队最快
            queued = [j for j in self._queued_sorted() if j["runtime_profile"] == src]
            if queued:
                return queued[0]
        return None

    async def process_cpu_once(self) -> bool:
        job = self._cpu_target()
        if job is None:
            return False
        claimed = self.store.claim_next(job["runtime_profile"])
        if claimed is None or claimed["id"] != job["id"]:
            if claimed:
                self.store.set_status(claimed["id"], "queued")
                self.wake()
            return bool(claimed)
        await self.run_job(claimed, lane="cpu")
        return True

    # ---------- selection ----------
    def _queued_sorted(self) -> list[dict]:
        return sorted(self.store.oldest_queued(),
                      key=lambda j: (-j["priority"], j["created_at"]))

    def choose(self) -> dict | None:
        queued = self._queued_sorted()
        if not queued:
            return None
        now_ts = time.monotonic()
        queued = [j for j in queued if self._backoff.get(j["runtime_profile"], 0) <= now_ts]
        if not queued:
            return None
        active = self.sup.active_profile
        if not active:
            return queued[0]
        mine = [j for j in queued if j["runtime_profile"] == active]
        others = [j for j in queued if j["runtime_profile"] != active]
        if not others:
            return mine[0] if mine else None
        wait_other = max(_age_s(j["created_at"]) for j in others)
        if mine and self.streak < self.max_same and wait_other < self.max_other_wait:
            return mine[0]
        return others[0]

    # ---------- execution ----------
    async def process_once(self) -> bool:
        job = self.choose()
        if job is None:
            return False
        claimed = self.store.claim_next(job["runtime_profile"])
        # priority ordering could have been beaten by another claim; loop is
        # single-flight, so claim must match unless job got cancelled.
        if claimed is None or claimed["id"] != job["id"]:
            if claimed:
                self.store.set_status(claimed["id"], "queued")
                self.wake()
            return bool(claimed)
        await self.run_job(claimed)
        return True

    async def run_job(self, job: dict, lane: str = "gpu"):
        payload = job["payload"]
        payload["job_id"] = job["id"]
        profile_name = job["runtime_profile"]
        # CPU 兜底通道换用执行 profile（任务记录里的 runtime_profile 不变，
        # API 契约稳定；实际设备在输出 meta.device 里如实回执）
        exec_profile = (CPU_FALLBACK_ROUTES.get(profile_name, profile_name)
                        if lane == "cpu" else profile_name)
        if self.store.cancel_requested(job["id"]):
            self._finish(job["id"], "cancelled", error="cancelled by user")
            return
        try:
            slots = providers.asset_slots(payload)
            paths = {}
            for slot, asset_id in slots.items():
                asset = self.store.get_asset(asset_id)
                if not asset or not Path(asset["path"]).exists():
                    raise BackendCrash(f"asset missing for {slot}: {asset_id}")
                paths[slot] = asset["path"]
            pending = config.DATA_DIR / "pending"
            pending.mkdir(exist_ok=True)
            task = providers.compile(payload, paths, pending)
        except Exception as e:
            self._finish(job["id"], "failed", error=str(e)[:500], error_type="compile_error")
            return

        try:
            backend = await self.sup.ensure(exec_profile)
        except AdmissionDenied as e:
            self._backoff[exec_profile] = time.monotonic() + self._next_backoff(exec_profile)
            self.store.set_status(job["id"], "queued")
            self.store.log_event(exec_profile, exec_profile,
                                 f"admission_denied: {e}", ok=False)
            return
        except BackendCrash as e:
            await self._fail_or_retry(job, f"runtime_start_failed: {e}", "start_error")
            return

        if lane == "gpu":
            if self.last_profile != profile_name:
                self.streak = 0
                self.last_profile = profile_name
            self.streak += 1
        self._backoff.pop(exec_profile, None)
        self._backoff_attempts.pop(exec_profile, None)

        if self.store.cancel_requested(job["id"]):
            self._finish(job["id"], "cancelled", error="cancelled by user")
            return

        self.store.set_status(job["id"], "running")
        self.bus.publish(job["id"], {"status": "running"})
        timeout = self.sup.profiles[exec_profile].job_timeout_s
        try:
            result = await asyncio.wait_for(backend.submit(job["id"], task), timeout=timeout)
        except asyncio.TimeoutError:
            await backend.cancel(job["id"])
            # never reuse a hung worker (lane-targeted: a CPU worker must not
            # tear down the GPU runtime and vice versa)
            await self.sup.shutdown("job_timeout", lane=lane)
            await self._fail_or_retry(job, f"job timeout after {timeout}s", "timeout")
            self.sup.job_finished(lane=lane)
            return
        except BackendCrash as e:
            await self._fail_or_retry(job, str(e), "runtime_crash")
            self.sup.job_finished(lane=lane)
            return
        except Exception as e:  # defensive: unknown backend failure
            await self._fail_or_retry(job, f"unexpected: {e}", "internal_error")
            self.sup.job_finished(lane=lane)
            return

        self.sup.job_finished(lane=lane)
        if result.get("cancel_requested") or self.store.cancel_requested(job["id"]):
            for p in result.get("paths", []):
                if self._contained(p):
                    Path(p).unlink(missing_ok=True)
            self._finish(job["id"], "cancelled", error="cancelled by user")
            return
        if result.get("status") == "success":
            try:
                await asyncio.to_thread(self._ingest_outputs, job["id"],
                                        result.get("paths", []), result.get("meta"))
            except Exception as e:
                await self._fail_or_retry(job, f"output ingest failed: {e}", "ingest_error")
                return
            capabilities.mark_validated(job["provider"], job["workflow"])
            self._finish(job["id"], "completed")
        else:
            if result.get("error_type") == "oom_error":
                await self.sup.shutdown("oom", lane=lane)
            await self._fail_or_retry(job, result.get("error", "generation failed"),
                                      result.get("error_type", "generation_error"))

    def _next_backoff(self, profile: str) -> int:
        seq = config.ADMISSION_BACKOFF_S
        attempts = self._backoff_attempts.get(profile, 0) + 1
        self._backoff_attempts[profile] = attempts
        return seq[min(attempts - 1, len(seq) - 1)]

    async def _fail_or_retry(self, job: dict, error: str, error_type: str):
        if self.store.cancel_requested(job["id"]):
            self._finish(job["id"], "cancelled", error="cancelled by user")
            return
        if job["retry_count"] < 1 and error_type in RETRYABLE_ERRORS:
            self.store.update_job(job["id"], retry_count=job["retry_count"] + 1)
            self.store.set_status(job["id"], "queued")
            self.store.log_event(job["runtime_profile"], None,
                                 f"retry: {error}"[:300], ok=False)
            self.wake()
            return
        self._finish(job["id"], "failed", error=error[:500], error_type=error_type)

    def _managed_roots(self) -> list[Path]:
        # Managed output roots: ONLY paths under these are ever read, moved or
        # deleted from backend/callback-reported data. This is the containment
        # line against forged /v1/internal/task_complete payloads.
        return [(config.DATA_DIR / "pending").resolve(),
                config.OUTPUTS_DIR.resolve(),
                (config.COMFYUI_DIR / "output").resolve()]

    def _contained(self, p: str) -> bool:
        rp = Path(p).resolve()
        return any(rp == r or rp.is_relative_to(r) for r in self._managed_roots())

    def _ingest_outputs(self, job_id: str, paths: list[str], meta: dict | None = None):
        roots = self._managed_roots()
        for p in paths:
            rp = Path(p).resolve()
            if not any(rp == r or rp.is_relative_to(r) for r in roots):
                raise ValueError(f"output path outside managed dirs: {p!r}")
            if not rp.exists():
                continue
            with open(rp, "rb") as fh:
                sha = hashlib.file_digest(fh, "sha256").hexdigest()
            size = os.path.getsize(rp)
            ext = rp.suffix or ".bin"
            kind = EXT_KIND.get(ext.lower(), "file")
            dest_dir = config.OUTPUTS_DIR / sha[:2]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{sha[:24]}{ext}"
            if not dest.exists():
                shutil.move(str(rp), dest)
            else:
                rp.unlink()
            # worker-reported meta (duration_s etc.) rides on the asset, not
            # the job: content-addressed, so every reuse of this exact bytes
            # keeps the same honest duration. Only the first writer sets it.
            asset = self.store.put_asset(sha256=sha, kind=kind, path=str(dest), size=size,
                                         meta=meta)
            self.store.add_output(job_id, kind, asset["id"], str(dest), role="main")

    def _finish(self, job_id: str, status: str, **extra):
        # Final chokepoint: if a cancel was accepted (API already answered
        # "cancelling") the job may never land as completed, even if it raced
        # through the last checkpoint. Ingested content-addressed assets stay
        # (dedup may share them); only the job status is honored as cancelled.
        if status == "completed" and self.store.cancel_requested(job_id):
            status, extra = "cancelled", {"error": "cancelled by user"}
        self.store.set_status(job_id, status, **extra)
        self.bus.publish(job_id, {"status": status, **extra})

    # ---------- observation ----------
    def snapshot(self) -> dict:
        return {
            "active_profile": self.sup.active_profile,
            "state": self.sup.state,
            "cpu_active_profile": self.sup.cpu_active_profile,
            "cpu_state": self.sup.cpu_state,
            "streak": self.streak,
            "queued_by_profile": dict(self.store.queued_profiles()),
            "queued_total": self.store.queued_count(),
            "admission_backoff": {k: round(v - time.monotonic(), 1)
                                  for k, v in self._backoff.items() if v > time.monotonic()},
        }
