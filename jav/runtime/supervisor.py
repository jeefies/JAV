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
from .base import BaseBackend, BackendCrash, _await_sync
from .comfy import ComfyBackend
from .zit import ZitBackend

BACKEND_CLASSES: dict[str, type[BaseBackend]] = {
    "zit_subprocess": ZitBackend,
    "comfyui": ComfyBackend,
}


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


def vram_pids() -> dict[int, int] | None:
    """pid -> used MB via nvidia-smi. None means PROBE FAILED (distinct from
    an empty dict = genuinely no GPU processes) — callers must fail closed."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        res = {}
        for line in r.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                res[int(parts[0].strip())] = int(parts[1].strip())
        return res
    except Exception:
        return None


def vram_free_mb() -> int | None:
    """Free VRAM MB; None on probe failure (must not be treated as 0)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip().isdigit()]
        return int(lines[0]) if lines else None
    except Exception:
        return None


def proc_starttime(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat: creation time in clock ticks (PID-reuse
    discriminator)."""
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes().decode(errors="ignore")
    except OSError:
        return None
    rest = data[data.rfind(")") + 1:].split()
    return int(rest[19]) if len(rest) > 19 else None


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
        vneed = p.vram_budget_mb + config.VRAM_FLOOR_MB
        if os.getenv("JAV_VRAM_GATE", "on") == "off":
            return
        vfree = self.vram_free_probe()
        if vfree is None:
            # fail CLOSED: an unverifiable GPU state must not green-light a
            # heavyweight runtime (design contract: RAM+VRAM double check)
            raise AdmissionDenied(profile_name, vneed, 0, resource="VRAM(probe failed)")
        if vfree < vneed:
            raise AdmissionDenied(profile_name, vneed, vfree, resource="VRAM")

    def _spawn_hook(self, profile_name: str):
        """Record pid+starttime the instant the child spawns (not only at
        READY): a crash during multi-minute weight loading would otherwise
        leave the orphan invisible to sweep_orphans."""
        def hook(pid: int):
            self.store.save_runtime_state(pid, profile_name, proc_starttime(pid))
        return hook

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
            await _await_sync(self.check_admission, profile_name)
            cls = BACKEND_CLASSES.get(p.backend)
            if cls is None:
                raise BackendCrash(f"no backend class for {p.backend}")
            backend = cls(p)
            backend.on_spawn = self._spawn_hook(profile_name)
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
            self.store.save_runtime_state(backend.pid or -1, profile_name,
                                          proc_starttime(backend.pid) if backend.pid else None)
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
            verified = False
            while time.monotonic() < deadline:
                used = await _await_sync(self.vram_probe)
                if used is None:
                    await asyncio.sleep(1)
                    continue  # probe failure never confirms release
                if pid not in used:
                    verified = True
                    break
                await asyncio.sleep(1)
            if not verified:
                self.store.log_event(self.active_profile, None,
                                     f"vram_release_unverified pid={pid}", ok=False)
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
            if pid <= 0:
                return None
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                errors="ignore").replace("\0", " ")
            # Strict identity: the exact script the recorded profile spawns.
            profile = st.get("profile")
            p = self.profiles.get(profile) if profile else None
            if p is not None and p.backend == "zit_subprocess" and p.script:
                expected = str(p.script)
            elif p is not None and p.backend == "comfyui":
                expected = str(config.COMFYUI_DIR / "main.py")
            else:
                self.store.log_event(profile, None, "startup_sweep_skipped_unknown_profile",
                                     ok=False, detail=f"pid={pid}")
                return None
            owned = expected in cmdline
            recorded_st = st.get("start_time")
            if recorded_st is not None and proc_starttime(pid) != recorded_st:
                owned = False  # PID reused since recording
            if owned:
                try:
                    os.kill(pid, 15)
                    time.sleep(2)
                    if Path(f"/proc/{pid}").exists():
                        live_st = proc_starttime(pid)
                        if recorded_st is None or live_st == recorded_st:
                            os.kill(pid, 9)
                except OSError as e:
                    self.store.log_event(profile, None, f"startup_sweep_kill_failed: {e}",
                                         ok=False, detail=f"pid={pid}")
                else:
                    self.store.log_event(profile, None, "startup_sweep_killed", ok=True,
                                         detail=f"pid={pid}")
        except OSError:
            pass  # /proc gone: process already exited
        finally:
            self.store.clear_runtime_state()
        return pid
