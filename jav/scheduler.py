"""Scheduler: one persistent SQLite queue + runtime_profile affinity with
anti-starvation, RAM admission backoff, idle unload, crash retry (JAV-DESIGN 4)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from datetime import datetime
from datetime import timezone as datetime_timezone
from pathlib import Path

from . import capabilities, config, providers
from .models import PROVIDER_WORKFLOWS
from .runtime.base import BackendCrash
from .runtime.supervisor import AdmissionDenied
from .store import cache_hash, now


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
                    "internal_error", "ingest_error", "runtime_start_failed",
                    "oom_error"}


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
        self._task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

    def wake(self):
        self._wake.set()

    async def start(self):
        self._task = asyncio.create_task(self._loop(), name="jav-scheduler")

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
        if self.sup.backend:
            await self.sup.shutdown("exit")

    async def stop(self):
        self.stopped.set()
        self.wake()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=90)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

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
        wait_other = min(_age_s(j["created_at"]) for j in others)
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

    async def run_job(self, job: dict):
        payload = job["payload"]
        payload["job_id"] = job["id"]
        profile_name = job["runtime_profile"]
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
            backend = await self.sup.ensure(profile_name)
        except AdmissionDenied as e:
            self._backoff[profile_name] = time.monotonic() + self._next_backoff(profile_name)
            self.store.set_status(job["id"], "queued")
            self.store.log_event(self.sup.active_profile, profile_name,
                                 f"admission_denied: {e}", ok=False)
            return
        except BackendCrash as e:
            await self._fail_or_retry(job, f"runtime_start_failed: {e}", "start_error")
            return

        if self.last_profile != profile_name:
            self.streak = 0
            self.last_profile = profile_name
        self.streak += 1
        self._backoff.pop(profile_name, None)
        self._backoff_attempts.pop(profile_name, None)

        self.store.set_status(job["id"], "running")
        self.bus.publish(job["id"], {"status": "running"})
        timeout = self.sup.profiles[profile_name].job_timeout_s
        try:
            result = await asyncio.wait_for(backend.submit(job["id"], task), timeout=timeout)
        except asyncio.TimeoutError:
            await backend.cancel(job["id"])
            await self.sup.shutdown("job_timeout")  # never reuse a hung worker
            await self._fail_or_retry(job, f"job timeout after {timeout}s", "timeout")
            self.sup.job_finished()
            return
        except BackendCrash as e:
            await self._fail_or_retry(job, str(e), "runtime_crash")
            self.sup.job_finished()
            return
        except Exception as e:  # defensive: unknown backend failure
            await self._fail_or_retry(job, f"unexpected: {e}", "internal_error")
            self.sup.job_finished()
            return

        self.sup.job_finished()
        if result.get("cancel_requested"):
            for p in result.get("paths", []):
                Path(p).unlink(missing_ok=True)
            self._finish(job["id"], "cancelled", error="cancelled by user")
            return
        if result.get("status") == "success":
            try:
                self._ingest_outputs(job["id"], result.get("paths", []))
            except Exception as e:
                await self._fail_or_retry(job, f"output ingest failed: {e}", "ingest_error")
                return
            capabilities.mark_validated(job["provider"], job["workflow"])
            self._finish(job["id"], "completed")
        else:
            if result.get("error_type") == "oom_error":
                await self.sup.shutdown("oom")
            await self._fail_or_retry(job, result.get("error", "generation failed"),
                                      result.get("error_type", "generation_error"))

    def _next_backoff(self, profile: str) -> int:
        seq = config.ADMISSION_BACKOFF_S
        attempts = self._backoff_attempts.get(profile, 0) + 1
        self._backoff_attempts[profile] = attempts
        return seq[min(attempts - 1, len(seq) - 1)]

    async def _fail_or_retry(self, job: dict, error: str, error_type: str):
        if job["retry_count"] < 1 and error_type in RETRYABLE_ERRORS:
            self.store.update_job(job["id"], retry_count=job["retry_count"] + 1)
            self.store.set_status(job["id"], "queued")
            self.store.log_event(job["runtime_profile"], None,
                                 f"retry: {error}"[:300], ok=False)
            self.wake()
            return
        self._finish(job["id"], "failed", error=error[:500], error_type=error_type)

    def _ingest_outputs(self, job_id: str, paths: list[str]):
        for p in paths:
            src = Path(p)
            if not src.exists():
                continue
            data = src.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            ext = src.suffix or ".bin"
            kind = {".png": "image", ".jpg": "image", ".jpeg": "image",
                    ".webp": "image", ".mp4": "video", ".mov": "video",
                    ".webm": "video", ".mp3": "audio", ".wav": "audio",
                    ".flac": "audio"}.get(ext.lower(), "file")
            dest_dir = config.OUTPUTS_DIR / sha[:2]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{sha[:24]}{ext}"
            if not dest.exists():
                shutil.move(str(src), dest)
            else:
                src.unlink()
            asset = self.store.put_asset(sha256=sha, kind=kind, path=str(dest), size=len(data))
            self.store.add_output(job_id, kind, asset["id"], str(dest), role="main")

    def _finish(self, job_id: str, status: str, **extra):
        self.store.set_status(job_id, status, **extra)
        self.bus.publish(job_id, {"status": status, **extra})

    # ---------- observation ----------
    def snapshot(self) -> dict:
        return {
            "active_profile": self.sup.active_profile,
            "state": self.sup.state,
            "streak": self.streak,
            "queued_by_profile": dict(self.store.queued_profiles()),
            "queued_total": self.store.queued_count(),
            "admission_backoff": {k: round(v - time.monotonic(), 1)
                                  for k, v in self._backoff.items() if v > time.monotonic()},
        }
