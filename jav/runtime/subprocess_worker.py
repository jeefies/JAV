"""SubprocessWorkerBackend: generic stdin-JSON + HTTP-callback worker protocol
shared by the ZIT image worker and the CosyVoice TTS worker.

Backend subclasses only declare identity bits:
  label       -> log file prefix + error text
  preflight() -> weight/repo existence checks (raise BackendCrash)
  env_vars()  -> worker-specific env on top of the callback triple
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from .. import config
from .base import BaseBackend, BackendCrash, StartTimeout, _await_sync


class SubprocessWorkerBackend(BaseBackend):
    kind = "subprocess_worker"
    label = "worker"

    def __init__(self, profile):
        super().__init__(profile)
        self.proc = None
        self.log_handle = None
        self._loaded = asyncio.Event()
        self._load_error: str | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._cancel_requested: set[str] = set()
        self._unowned: list[dict] = []  # callbacks before future registered

    # ---------- subclass hooks ----------
    def preflight(self) -> None:
        return None

    def env_vars(self) -> dict[str, str]:
        return {}

    # ---------- lifecycle ----------
    async def start(self):
        if not Path(self.profile.script).exists():
            raise BackendCrash(f"worker script missing: {self.profile.script}")
        self.preflight()
        log_dir = config.LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_dir / f"{self.label}.{os.getpid()}.log", "a+")
        env = os.environ.copy()
        # The worker only ever needs the callback secret (set explicitly
        # below); never expose the service's own API bearer token to it.
        env.pop("JAV_API_TOKEN", None)
        env.update({
            "JAV_CALLBACK": f"http://127.0.0.1:{config.SERVICE_PORT}/v1/internal/task_complete",
            "JAV_PIPELINE_STATUS": f"http://127.0.0.1:{config.SERVICE_PORT}/v1/internal/pipeline_status",
            "JAV_CALLBACK_SECRET": config.CALLBACK_SECRET,
            "CUDA_VISIBLE_DEVICES": os.getenv("JAV_GPU_INDEX", "0"),
        })
        env.update(self.env_vars())
        self.proc = await _await_sync(
            lambda: subprocess.Popen(
                [self.profile.python_bin, "-u", self.profile.script],
                stdin=subprocess.PIPE,
                stdout=self.log_handle, stderr=self.log_handle, env=env,
                cwd=str(config.BASE_DIR)))
        self.pid = self.proc.pid
        if self.on_spawn:
            self.on_spawn(self.pid)
        try:
            await asyncio.wait_for(self._loaded.wait(), timeout=self.profile.start_timeout_s)
        except asyncio.TimeoutError:
            await self.stop("start_timeout")
            raise StartTimeout(f"{self.label} worker did not report loaded in time")
        if self._load_error:
            raise BackendCrash(f"worker load error: {self._load_error}")

    async def stop(self, reason: str = "stop"):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                await _await_sync(self.proc.wait, 15)
            except Exception:
                self.proc.kill()
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        self.proc = None
        self.pid = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------- job protocol ----------
    async def submit(self, job_id: str, task: dict) -> dict:
        if not self.alive():
            raise BackendCrash(f"{self.label} worker not alive")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[job_id] = fut
        for late in [c for c in self._unowned if c.get("job_id") == job_id]:
            self._resolve(late)
            self._unowned.remove(late)
        line = json.dumps(task) + "\n"

        def _write():
            self.proc.stdin.write(line.encode())
            self.proc.stdin.flush()
        await _await_sync(_write)
        try:
            return await fut
        finally:
            self._pending.pop(job_id, None)

    async def cancel(self, job_id: str):
        self._cancel_requested.add(job_id)

    def _resolve(self, payload: dict):
        job_id = payload.get("job_id")
        fut = self._pending.get(job_id)
        if fut is None or fut.done():
            return
        status = payload.get("status")
        if status == "processing":
            return
        result = {
            "status": "success" if status == "success" else "failed",
            "paths": [payload["path"]] if payload.get("path") else [],
            "error": payload.get("error"),
            "error_type": payload.get("error_type"),
            "meta": payload.get("meta") or None,
            "cancel_requested": job_id in self._cancel_requested,
        }
        self._cancel_requested.discard(job_id)
        fut.set_result(result)

    def deliver(self, payload: dict) -> bool:
        job_id = payload.get("job_id")
        if job_id in self._pending:
            self._resolve(payload)
            return True
        self._unowned.append(payload)
        if len(self._unowned) > 100:  # bounded: forged/late callbacks cannot grow it
            self._unowned.pop(0)
        return True

    def pipeline_status(self, payload: dict) -> bool:
        s = payload.get("status")
        if s == "loaded":
            self._loaded.set()
        elif s == "error":
            self._load_error = payload.get("error", "unknown")
            self._loaded.set()
        return True
