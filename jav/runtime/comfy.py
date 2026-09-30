"""ComfyBackend: ComfyUI-as-RPC engine (JAV-DESIGN 1/9).

Headless ComfyUI subprocess on 127.0.0.1:8188; jobs compile to API-format
graphs, submit via /prompt, track via /history polling, collect files from
ComfyUI's output dir. Cross-family switch kills the whole process
(struct.md 12/13)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

from .. import config
from .base import BaseBackend, BackendCrash, StartTimeout, _await_sync

POLL_INTERVAL_S = 2.0


class ComfyBackend(BaseBackend):
    kind = "comfyui"

    def __init__(self, profile, port: int = config.COMFY_PORT):
        super().__init__(profile)
        self.port = port
        self.proc = None
        self.log_handle = None
        self.base = f"http://127.0.0.1:{port}"
        self.input_dir = config.COMFYUI_DIR / "input"
        self.output_dir = config.COMFYUI_DIR / "output"

    async def start(self):
        if not (config.COMFYUI_DIR / "main.py").exists():
            raise BackendCrash(f"ComfyUI not found at {config.COMFYUI_DIR}")
        if not Path(self.profile.python_bin).exists():
            raise BackendCrash(f"python env missing: {self.profile.python_bin}")
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(config.LOG_DIR / f"comfyui.{self.profile.name}.log", "a+")
        env = os.environ.copy()
        env.update({
            "HF_HOME": str(config.HF_HOME_DIR),
            "CUDA_VISIBLE_DEVICES": os.getenv("JAV_GPU_INDEX", "0"),
        })
        cmd = [self.profile.python_bin, "-u", str(config.COMFYUI_DIR / "main.py"),
               "--listen", "127.0.0.1", "--port", str(self.port),
               "--disable-auto-launch"] + list(self.profile.extra_args)
        self.proc = await _await_sync(
            lambda: subprocess.Popen(cmd, cwd=str(config.COMFYUI_DIR),
                                     stdout=self.log_handle, stderr=self.log_handle, env=env))
        self.pid = self.proc.pid
        if self.on_spawn:
            self.on_spawn(self.pid)
        deadline = time.monotonic() + self.profile.start_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BackendCrash(f"ComfyUI exited during start (code {self.proc.returncode})")
            try:
                r = await _await_sync(lambda: requests.get(f"{self.base}/system_stats", timeout=3))
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            await self.stop("start_timeout")
            raise StartTimeout("ComfyUI did not become ready")
        await self._check_nodes()

    async def _check_nodes(self):
        if not self.profile.required_nodes:
            return
        try:
            r = await _await_sync(
                lambda: requests.get(f"{self.base}/object_info", timeout=30))
            nodes = set(r.json().keys())
        except Exception as e:
            raise BackendCrash(f"object_info failed: {e}")
        missing = [n for n in self.profile.required_nodes if n not in nodes]
        if missing:
            raise BackendCrash(f"ComfyUI missing required nodes: {missing}")

    async def stop(self, reason: str = "stop"):
        if self.proc and self.proc.poll() is None:
            try:  # soft cleanup; process exit is the real free (JAV-DESIGN 4.3)
                await _await_sync(lambda: requests.post(
                    f"{self.base}/free",
                    json={"unload_models": True, "free_memory": True}, timeout=10))
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

    async def _upload_asset(self, path: str) -> str:
        src = Path(path)
        name = f"jav_{src.name}"

        def _copy():
            self.input_dir.mkdir(parents=True, exist_ok=True)
            dest = self.input_dir / name
            if not dest.exists():
                shutil.copyfile(src, dest)
            return name
        return await _await_sync(_copy)

    async def submit(self, job_id: str, task: dict) -> dict:
        if not self.alive():
            raise BackendCrash("ComfyUI not alive")
        graph = task["graph"]
        asset_names = {}
        for key, local_path in (task.get("asset_paths") or {}).items():
            asset_names[key] = await self._upload_asset(local_path)
        for node in graph.values():
            inputs = node.get("inputs", {})
            for k, v in list(inputs.items()):
                if isinstance(v, str) and v.startswith("asset:"):
                    inputs[k] = asset_names[v.split(":", 1)[1]]

        def _post():
            r = requests.post(f"{self.base}/prompt",
                              json={"prompt": graph, "client_id": job_id}, timeout=30)
            if r.status_code != 200:
                raise BackendCrash(f"/prompt rejected: {r.text[:400]}")
            return r.json()["prompt_id"]
        prompt_id = await _await_sync(_post)

        started = time.monotonic()
        timeout = self.profile.job_timeout_s
        poll = _await_sync
        while time.monotonic() - started < timeout:
            if not self.alive():
                raise BackendCrash("ComfyUI died during job")
            await asyncio.sleep(POLL_INTERVAL_S)

            def _hist():
                r = requests.get(f"{self.base}/history/{prompt_id}", timeout=10)
                return r.json() if r.status_code == 200 else {}
            hist = await poll(_hist)
            entry = hist.get(prompt_id)
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = entry.get("messages") or []
                parts = []
                for m in msgs:
                    if isinstance(m, (list, tuple)) and len(m) > 1 and isinstance(m[1], dict):
                        d = m[1]
                        parts.append(str(d.get("exec_info", {}).get("exception_message")
                                         or d.get("node_id", "")) + " " + str(d.get("exception_type", "")))
                    else:
                        parts.append(json.dumps(m, default=str)[:200])
                err = " | ".join(p for p in parts if p.strip()) or json.dumps(entry, default=str)[:400]
                return {"status": "failed", "paths": [], "error": f"comfyui error: {err[:500]}"}
            if entry.get("outputs"):
                paths = self._collect_outputs(entry["outputs"])
                if paths:
                    return {"status": "success", "paths": paths, "error": None}
        return {"status": "failed", "paths": [],
                "error": f"job timed out after {timeout}s waiting on ComfyUI history"}

    def _collect_outputs(self, outputs: dict) -> list[str]:
        paths = []
        for node_out in outputs.values():
            for kind in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(kind, []) or []:
                    sub = item.get("subfolder", "")
                    fn = item.get("filename", "")
                    p = self.output_dir / sub / fn
                    if p.exists():
                        paths.append(str(p))
        return paths

    async def cancel(self, job_id: str):
        try:
            await _await_sync(lambda: requests.post(f"{self.base}/interrupt",
                                                     json={}, timeout=5))
        except Exception:
            pass
