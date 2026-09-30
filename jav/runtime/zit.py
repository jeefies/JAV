"""ZitBackend: manages the diffusers worker subprocess (stdin JSON + HTTP
callback), same validated protocol as legacy ZIT-service."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

from .. import config
from .base import BaseBackend, BackendCrash, StartTimeout, _await_sync

MARKER = "jav.runtime.zit_worker"


class ZitBackend(BaseBackend):
    kind = "zit_subprocess"

    def __init__(self, profile):
        super().__init__(profile)
        self.proc = None
        self.log_handle = None
        self._loaded = asyncio.Event()
        self._load_error: str | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._cancel_requested: set[str] = set()
        self._unowned: list[dict] = []  # callbacks before future registered

    async def start(self):
        if not Path(self.profile.script).exists():
            raise BackendCrash(f"worker script missing: {self.profile.script}")
        model_dir = config.ZIT_WEIGHTS_DIR
        if not model_dir.is_dir():
            raise BackendCrash(f"ZIT weights missing at {model_dir}")
        log_dir = config.LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(log_dir / f"zit.{os.getpid()}.log", "a+")
        env = os.environ.copy()
        env.update({
            "ZIT_MODEL_DIR": str(model_dir),
            "JAV_CALLBACK": f"http://127.0.0.1:{config.SERVICE_PORT}/v1/internal/task_complete",
            "JAV_PIPELINE_STATUS": f"http://127.0.0.1:{config.SERVICE_PORT}/v1/internal/pipeline_status",
            "ZIT_WORKER_IDLE_TIMEOUT": str(max(self.profile.idle_unload_s + 300, 1800)),
            "CUDA_VISIBLE_DEVICES": os.getenv("JAV_GPU_INDEX", "0"),
        })
        self.proc = await _await_sync(
            lambda: subprocess.Popen(
                [self.profile.python_bin, "-u", self.profile.script],
                stdin=subprocess.PIPE,
                stdout=self.log_handle, stderr=self.log_handle, env=env,
                cwd=str(config.BASE_DIR)))
        self.pid = self.proc.pid
        try:
            await asyncio.wait_for(self._loaded.wait(), timeout=self.profile.start_timeout_s)
        except asyncio.TimeoutError:
            await self.stop("start_timeout")
            raise StartTimeout("zit worker did not report loaded in time")
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

    async def submit(self, job_id: str, task: dict) -> dict:
        if not self.alive():
            raise BackendCrash("zit worker not alive")
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
        return True

    def pipeline_status(self, payload: dict) -> bool:
        s = payload.get("status")
        if s == "loaded":
            self._loaded.set()
        elif s == "error":
            self._load_error = payload.get("error", "unknown")
            self._loaded.set()
        return True
