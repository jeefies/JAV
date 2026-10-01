"""JAV configuration: paths, runtime profiles, scheduler knobs.

Single source of truth for absolute paths (all under /mnt/data/AV per user
redline 2026-09-29). profiles.yaml is optional; built-in defaults match
JAV-DESIGN.md section 2.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(os.getenv("JAV_BASE_DIR", "/mnt/data/AV/JAV"))
DATA_DIR = Path(os.getenv("JAV_DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "jav.db"
ASSETS_DIR = DATA_DIR / "assets"
OUTPUTS_DIR = DATA_DIR / "outputs"
HF_HOME_DIR = DATA_DIR / "hf_home"
LOG_DIR = Path(os.getenv("JAV_LOG_DIR", BASE_DIR / "logs"))

ZIT_WEIGHTS_DIR = Path(os.getenv("ZIT_WEIGHTS_DIR", "/mnt/data/AV/models/Z-Image-Turbo"))
COMFYUI_DIR = Path(os.getenv("JAV_COMFYUI_DIR", "/mnt/data/AV/ComfyUI"))
# Unified env (2026-09-30): JAV server + ZIT worker + ComfyUI backend all run
# on /home/jeefy/miniconda3/envs/comfyui (py3.11, torch 2.11+cu130,
# diffusers git@50e7158, fastapi stack). Legacy openclaw-home image env retired.
CONDA_IMAGE_PY = os.getenv("JAV_PYTHON_BIN", "/home/jeefy/miniconda3/envs/comfyui/bin/python3")
CONDA_COMFYUI_PY = CONDA_IMAGE_PY

SERVICE_PORT = int(os.getenv("JAV_PORT", "8765"))
COMFY_PORT = int(os.getenv("JAV_COMFY_PORT", "8188"))

# Scheduler knobs (JAV-DESIGN.md section 4.2)
MAX_SAME_RUNTIME_JOBS = int(os.getenv("JAV_MAX_SAME_RUNTIME_JOBS", "3"))
MAX_OTHER_WAIT_S = int(os.getenv("JAV_MAX_OTHER_WAIT_S", "600"))
QUEUE_DEPTH_LIMIT = int(os.getenv("JAV_QUEUE_DEPTH_LIMIT", "500"))
BATCH_MAX_JOBS = 64

# Optional API bearer token (set JAV_API_TOKEN in a systemd drop-in before
# exposing 8765 through a tunnel; empty = auth disabled, local-only posture).
API_TOKEN = os.getenv("JAV_API_TOKEN", "")
# Worker-callback shared secret: per-process random unless pinned via env.
# /v1/internal/* only accepts requests presenting X-JAV-Callback.
CALLBACK_SECRET = os.getenv("JAV_CALLBACK_SECRET") or secrets.token_hex(16)

# OOM protection for co-located workloads (unichess training etc.):
# admission uses live MemAvailable+SwapFree; keep a hard floor so JAV never
# pushes the box into OOM-killer territory.
MEM_FLOOR_MB = int(os.getenv("JAV_MEM_FLOOR_MB", "4096"))
# VRAM floor must cover the EXTERNAL (unmanaged) usage that races admission:
# unichess app.py (~916MiB persistent) + Kit selfplay (~1.16->1.4GiB, GROWS
# after the spawn-time snapshot; measured live 2026-10-01). 256 was too thin
# (real OOM); budgets must equal the honest measured worker peak, floor = the
# external growth allowance on top. With expandable_segments active (pre-torch
# in zit_worker) the ZIT true peak drops ~0.6G; 13312+768 keeps today's
# knife-edge cases queued instead of burning a doomed cold-start + retry.
VRAM_FLOOR_MB = int(os.getenv("JAV_VRAM_FLOOR_MB", "768"))
ADMISSION_BACKOFF_S = (15, 30, 60, 120, 240)


@dataclass
class Profile:
    name: str
    backend: str                      # "zit_subprocess" | "comfyui"
    ram_budget_mb: int
    vram_budget_mb: int
    start_timeout_s: int = 300
    job_timeout_s: int = 1800
    idle_unload_s: int = 1800
    python_bin: str = ""
    script: str = ""
    required_nodes: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    enabled: bool = True              # operator kill-switch

    @staticmethod
    def from_dict(d: dict, base: "Profile | None" = None) -> "Profile":
        # Field-level merge: an override in profiles.yaml only replaces the
        # keys it actually names. Without a base, the class defaults apply.
        # (A whole-profile replace would silently blank script/python_bin
        #  and timeouts — a service-killing trap documented 2026-10-01.)
        import dataclasses
        if base is not None:
            vals = dataclasses.asdict(base)
        else:
            vals = dataclasses.asdict(
                Profile(name=d.get("name") or "", backend=d.get("backend") or "",
                        ram_budget_mb=12288, vram_budget_mb=8192))
        for k, v in d.items():
            if k not in vals:
                raise ValueError(f"profiles.yaml: unknown key {k!r} for profile "
                                 f"{d.get('name')!r}")
            vals[k] = v
        for k in ("ram_budget_mb", "vram_budget_mb", "start_timeout_s",
                  "job_timeout_s", "idle_unload_s"):
            vals[k] = int(vals[k])
        for k in ("required_nodes", "extra_args"):
            vals[k] = tuple(vals[k])
        vals["enabled"] = bool(vals["enabled"])
        if not vals["name"] or not vals["backend"]:
            raise ValueError("profile requires name and backend")
        return Profile(**vals)


def default_profiles() -> dict[str, Profile]:
    return {
        "zit": Profile(
            name="zit", backend="zit_subprocess",
            # observed peak 29.2G RSS during i2i/inpaint fp32->bf16 derived
            # cast (resident bf16 ~20G + temporary fp32 checkpoint copy);
            # 20480 under-budgeted the peak and thrashed against the cgroup cap
            ram_budget_mb=30720, vram_budget_mb=13312,
            start_timeout_s=600, job_timeout_s=900,
            idle_unload_s=int(os.getenv("JAV_ZIT_IDLE_UNLOAD_S", "300")),
            python_bin=os.getenv("ZIT_PYTHON_BIN", CONDA_IMAGE_PY),
            script=str(BASE_DIR / "jav" / "runtime" / "zit_worker.py"),
        ),
        "ltx25": Profile(
            name="ltx25", backend="comfyui",
            ram_budget_mb=24576, vram_budget_mb=12288,
            start_timeout_s=420, job_timeout_s=2400, idle_unload_s=300,
            python_bin=os.getenv("LTX_PYTHON_BIN", CONDA_COMFYUI_PY),
            required_nodes=("LTXVConditioning", "LTXVScheduler", "EmptyLTXVLatentVideo"),
        ),
        "mh3.fl2va": Profile(
            name="mh3.fl2va", backend="comfyui",
            ram_budget_mb=28672, vram_budget_mb=12288,
            start_timeout_s=600, job_timeout_s=3600, idle_unload_s=300,
            python_bin=os.getenv("MH3_PYTHON_BIN", CONDA_COMFYUI_PY),
            required_nodes=("MiniMaxH3ImageToVideo",),
        ),
        "mh3.ref2va": Profile(
            name="mh3.ref2va", backend="comfyui",
            ram_budget_mb=28672, vram_budget_mb=12288,
            start_timeout_s=600, job_timeout_s=3600, idle_unload_s=300,
            python_bin=os.getenv("MH3_PYTHON_BIN", CONDA_COMFYUI_PY),
            required_nodes=("MiniMaxH3ReferenceToVideo",),
        ),
    }


def load_profiles() -> dict[str, Profile]:
    # NOT memoized: callers may rebase config.BASE_DIR at runtime (tests), and
    # default_profiles() embeds BASE_DIR paths. The per-submit YAML cost is now
    # absorbed by the capabilities() result cache instead.
    profiles = default_profiles()
    cfg = BASE_DIR / "config" / "profiles.yaml"
    if cfg.exists():
        try:
            import yaml  # not guaranteed installed; yaml section is optional
            for d in yaml.safe_load(cfg.read_text()) or []:
                p = Profile.from_dict(d, base=profiles.get(d.get("name")))
                profiles[p.name] = p
        except ImportError:
            pass
    return profiles


def ensure_dirs() -> None:
    for d in (DATA_DIR, ASSETS_DIR, OUTPUTS_DIR, HF_HOME_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
