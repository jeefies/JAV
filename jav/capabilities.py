"""Dynamic capability discovery (JAV-DESIGN 5): weights present + manifest
exists + operator flag. Interface existing != hardware validated."""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import PROVIDER_WORKFLOWS

# Workflows whose provider templates are implemented in code.
IMPLEMENTED = {"zit.t2i", "zit.i2i", "zit.inpaint",
               "ltx25.t2v", "ltx25.i2v", "ltx25.flf2v", "ltx25.a2v", "ltx25.bbox_control",
               "mh3.t2v", "mh3.i2v", "mh3.fl2v", "mh3.ref2v", "mh3.fun_control",
               "mh3.multiframe"}

# Workflows that passed a real-hardware smoke test (recorded, not inferred).
FLAGS_FILE = Path(config.DATA_DIR) / "capability_flags.json"


def _load_flags() -> dict:
    if FLAGS_FILE.exists():
        try:
            return json.loads(FLAGS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _weights_for(profile: str) -> tuple[bool, str]:
    """Return (present, detail) for the profile's big weights on disk."""
    if profile == "zit":
        ok = config.ZIT_WEIGHTS_DIR.is_dir() and any(
            (config.ZIT_WEIGHTS_DIR / p).exists()
            for p in ("model_index.json", "transformer"))
        return ok, "Z-Image-Turbo diffusers snapshot"
    if profile == "ltx25":
        needles = ("ltx-2.5", "ltx2.5", "ltx_2.5")
        hits = []
        for d in ("checkpoints", "diffusion_models"):
            root = config.COMFYUI_DIR / "models" / d
            if root.is_dir():
                hits += [f.name for f in root.iterdir()
                         if any(n in f.name.lower() for n in needles)]
        return bool(hits), f"LTX 2.5 weights {hits or ''}"
    if profile.startswith("mh3"):
        kind = "ref2va" if profile.endswith("ref2va") else "fl2va"
        m = config.COMFYUI_DIR / "models"
        need = [m / "diffusion_models" / f"minimax_h3_{kind}_pruned_int8_convrot.safetensors",
                m / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                m / "vae" / "minimax_h3_video_vae_int8_convrot.safetensors",
                m / "vae" / "minimax_h3_audio_vae_fp32.safetensors"]
        missing = [p.name for p in need if not p.exists()]
        return not missing, f"MiniMax H3 {kind} weights missing {missing}"
    return False, "unknown profile"


def capabilities() -> dict:
    flags = _load_flags()
    profiles = config.load_profiles()
    out: dict = {}
    for provider, wfmap in PROVIDER_WORKFLOWS.items():
        out[provider] = {"workflows": {}}
        for wf, profile in sorted(wfmap.items()):
            key = f"{provider}.{wf}"
            entry: dict = {"available": False, "runtime": profile}
            if key not in IMPLEMENTED:
                entry["reason"] = "template not implemented yet"
            elif not profiles[profile].enabled:
                entry["reason"] = "runtime disabled by operator"
            else:
                present, detail = _weights_for(profile)
                if not present:
                    entry["reason"] = f"weights missing: {detail}"
                elif not flags.get(key, provider == "zit"):
                    entry["reason"] = "not validated on this hardware"
                else:
                    entry["available"] = True
            out[provider]["workflows"][wf] = entry
    return out


def is_available(provider: str, workflow: str) -> tuple[bool, str]:
    caps = capabilities()
    entry = caps.get(provider, {}).get("workflows", {}).get(workflow)
    if entry is None:
        return False, f"unsupported workflow {provider}.{workflow}"
    return entry["available"], entry.get("reason", "")


def mark_validated(provider: str, workflow: str):
    """Called after a successful real smoke test of a workflow."""
    flags = _load_flags()
    flags[f"{provider}.{workflow}"] = True
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FLAGS_FILE.write_text(json.dumps(flags, indent=1))
