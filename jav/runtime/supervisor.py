"""RuntimeSupervisor: process mutex + state machine + admission (JAV-DESIGN 2/4.3).

At most one ACTIVE runtime process box-wide. Cross-family switch = kill old
process, verify VRAM release, spawn new. Admission control protects
co-located workloads (unichess training): a profile may only start when
live MemAvailable+SwapFree covers its RAM budget PLUS a hard floor.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

from .. import config
from .base import BaseBackend, BackendCrash
from .comfy import ComfyBackend
from .zit import ZitBackend

BACKEND_CLASSES: dict[str, type[BaseBackend]] = {
    "zit_subprocess": ZitBackend,
    "comfyui": ComfyBackend,
}

STATES = ("STOPPED", "STARTING", "READY", "BUSY", "DRAINING", "STOPPING", "FAILED")


def register_backend(kind: str, cls: type[BaseBackend]):
    BACKEND_CLASSES[kind] = cls


class AdmissionDenied(RuntimeError):
    def __init__(self, profile: str, need_mb: int, have_mb: int, resource: str = "RAM+swap"):
        super().__init__(
            f"admission denied for {profile}: need {need_mb}MB free {resource}, have {have_mb}MB")
        self.profile, self.need_mb, self.have_mb = profile, need_mb, have_mb


def mem_available_mb() -> int:
    kb = {"MemAvailable": 0, "SwapFree": 0}
    with open("/proc/meminfo") as fh:
        for line in fh:
            m = re.match(r"(\w+):\s+(\d+)", line)
            if m and m.group(1) in kb:
                kb[m.group(1)] = int(m.group(2))
    return (kb["MemAvailable"] + kb["SwapFree"]) // 1024


def vram_pids() -> dict[int, int]:
    """pid -> used MB via nvidia-smi (empty dict if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        res = {}
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                res[int(parts[0].strip())] = int(parts[1].strip())
        return res
    except Exception:
        return {}


def vram_free_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        lines = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
        return int(lines[0]) if lines else 0
    except Exception:
        return 0


class Supervisor:
    def __init__(self, store, profiles: dict | None = None,
                 mem_probe=mem_available_mb, vram_probe=vram_pids,
                 vram_free_probe=vram_free_mb):
        self.store = store
        self.profiles = profiles or config.load_profiles()
        self.mem_probe = mem_probe
        self.vram_probe = vram_probe
        self.vram_free_probe = vram_free_probe
        self.backend: BaseBackend | None = None
        self.active_profile: str | None = None
        self.state = "STOPPED"
        self.last_switch: dict | None = None
        self.last_job_done: float | None = None
        self.keepalive_ttl_s: int | None = None
        self._lock = asyncio.Lock()

    # ---------- lifecycle ----------
    def check_admission(self, profile_name: str):
        p = self.profiles[profile_name]
        have = self.mem_probe()
        need = p.ram_budget_mb + config.MEM_FLOOR_MB
        if have < need:
            raise AdmissionDenied(profile_name, need, have)
        vfree = self.vram_free_probe()
        vneed = p.vram_budget_mb + config.VRAM_FLOOR_MB
        if vfree and vfree < vneed:
            raise AdmissionDenied(profile_name, vneed, vfree, resource="VRAM")

    async def ensure(self, profile_name: str) -> BaseBackend:
        async with self._lock:
            if (self.backend is not None and self.active_profile == profile_name
                    and self.backend.alive()):
                return self.backend
            t0 = time.monotonic()
            prev = self.active_profile
            if self.backend is not None:
                await self._teardown("switch")
            p = self.profiles.get(profile_name)
            if p is None or not p.enabled:
                raise BackendCrash(f"profile {profile_name} disabled/unknown")
            self.check_admission(profile_name)
            cls = BACKEND_CLASSES.get(p.backend)
            if cls is None:
                raise BackendCrash(f"no backend class for {p.backend}")
            backend = cls(p)
            self.state = "STARTING"
            # route worker callbacks to the starting backend instance too
            self.backend = backend
            self.active_profile = profile_name
            try:
                await backend.start()
            except Exception as e:
                await self._teardown("start_failed")
                self.store.log_event(prev, profile_name, f"start_failed: {e}", ok=False)
                raise BackendCrash(f"start failed for {profile_name}: {e}") from e
            self.state = "READY"
            dur = int((time.monotonic() - t0) * 1000)
            self.store.log_event(prev, profile_name, "switch", duration_ms=dur)
            self.last_switch = {"from": prev, "to": profile_name,
                                "reason": "switch", "duration_ms": dur}
            self.store.save_runtime_state(backend.pid or -1, profile_name)
            return backend

    async def _teardown(self, reason: str):
        if self.backend is None:
            return
        self.state = "DRAINING" if reason == "idle" else "STOPPING"
        pid = self.backend.pid
        try:
            await self.backend.stop(reason)
        except Exception:
            pass
        if pid and pid > 0:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if pid not in self.vram_probe():
                    break
                await asyncio.sleep(1)
        self.backend = None
        self.active_profile = None
        self.state = "STOPPED"
        self.store.clear_runtime_state()

    async def shutdown(self, reason: str = "shutdown"):
        async with self._lock:
            await self._teardown(reason)

    def job_finished(self):
        self.last_job_done = time.time()
        self.keepalive_ttl_s = None

    async def stop_if_idle(self):
        async with self._lock:
            if self.backend and self.active_profile and self.last_job_done:
                p = self.profiles[self.active_profile]
                window = self.keepalive_ttl_s or p.idle_unload_s
                if time.time() - self.last_job_done > window:
                    self.keepalive_ttl_s = None
                    await self._teardown("idle")
                    self.store.log_event(p.name, None, "idle_unload")

    # ---------- callback routing ----------
    def deliver(self, payload: dict) -> bool:
        return self.backend.deliver(payload) if self.backend else False

    def pipeline_status(self, payload: dict) -> bool:
        return self.backend.pipeline_status(payload) if self.backend else False

    # ---------- startup sweep ----------
    def sweep_orphans(self):
        st = self.store.load_runtime_state()
        if not st:
            return None
        pid = st.get("pid", -1)
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        except OSError:
            self.store.clear_runtime_state()
            return None
        owned = ("zit_worker.py" in cmdline) or ("ComfyUI" in str(config.COMFYUI_DIR) and "main.py" in cmdline)
        if pid > 0 and owned:
            try:
                os.kill(pid, 15)
                time.sleep(2)
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, 9)
            except ProcessLookupError:
                pass
            self.store.log_event(st.get("profile"), None, "startup_sweep_killed", ok=True,
                                 detail=f"pid={pid}")
        self.store.clear_runtime_state()
        return pid
