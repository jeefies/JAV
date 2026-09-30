"""Dynamic capability discovery (JAV-DESIGN 5): weights present + manifest
exists + operator flag. Interface existing != hardware validated."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from . import config
from .models import PROVIDER_WORKFLOWS

# Workflows whose provider templates are implemented in code.
IMPLEMENTED = {"zit.t2i", "zit.i2i", "zit.inpaint",
               "ltx25.t2v", "ltx25.i2v", "ltx25.flf2v", "ltx25.a2v", "ltx25.bbox_control",
               "ltx25.union_control", "ltx25.motion_control", "ltx25.inpaint",
               "ltx25.outpaint", "ltx25.ic_lora",
               "mh3.t2v", "mh3.i2v", "mh3.fl2v", "mh3.ref2v", "mh3.fun_control",
               "mh3.multiframe"}

# Workflows that passed a real-hardware smoke test (recorded, not inferred).
FLAGS_FILE = Path(config.DATA_DIR) / "capability_flags.json"
_LAST_GOOD_FLAGS: dict | None = None


def _load_flags() -> dict:
    global _LAST_GOOD_FLAGS
    if FLAGS_FILE.exists():
        try:
            flags = json.loads(FLAGS_FILE.read_text())
            _LAST_GOOD_FLAGS = flags
            return flags
        except Exception:
            if _LAST_GOOD_FLAGS is not None:
                return _LAST_GOOD_FLAGS  # torn/corrupt write: keep last good
            return {}
    return {}


def _template_files(provider: str, workflows: list[str]) -> set[str]:
    """Every .safetensors referenced by the given workflow templates
    (vae_name/unet_name/clip_name/lora_name), so the weight gate can never
    green-light a workflow whose loader target was purged."""
    keys = ("vae_name", "unet_name", "clip_name", "lora_name")
    names: set[str] = set()
    root = config.BASE_DIR / "jav" / "workflows" / provider
    for wf in workflows:
        tpl = root / f"{wf}.api.json"
        if not tpl.exists():
            continue
        try:
            graph = json.loads(tpl.read_text())
        except Exception:
            continue
        stack = [graph]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, str) and k in keys and v.endswith(".safetensors"):
                        names.add(v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(n for n in node if isinstance(n, (dict, list)))
    return names


_MH3_MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints")


def _weights_for(profile: str) -> tuple[bool, str]:
    """Return (present, detail) for the profile's big weights on disk."""
    if profile == "zit":
        ok = config.ZIT_WEIGHTS_DIR.is_dir() and any(
            (config.ZIT_WEIGHTS_DIR / p).exists()
            for p in ("model_index.json", "transformer"))
        return ok, "Z-Image-Turbo diffusers snapshot"
    if profile == "ltx25":
        m = config.COMFYUI_DIR / "models"
        need = _template_files("ltx25", sorted(PROVIDER_WORKFLOWS["ltx25"]))
        # upscale variants share names; the distilled template sweep above
        # already contains every loader target of shipped graphs
        missing = [n for n in sorted(need)
                   if not any((m / sub / n).exists() for sub in _MH3_MODEL_SUBDIRS)]
        return not missing, f"LTX 2.5 weights missing {missing}"
    if profile.startswith("mh3"):
        kind = "ref2va" if profile.endswith("ref2va") else "fl2va"
        m = config.COMFYUI_DIR / "models"
        wfs = [w for w, p in PROVIDER_WORKFLOWS["mh3"].items() if p == profile]
        need = _template_files("mh3", wfs)
        need |= {f"minimax_h3_{kind}_pruned_int8_convrot.safetensors",
                 "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                 "minimax_h3_audio_vae_fp32.safetensors"}
        from .providers.mh3 import TURBO_LORA  # code-injected loras, not in templates
        need |= set(TURBO_LORA.values())
        missing = [n for n in sorted(need)
                   if not any((m / sub / n).exists() for sub in _MH3_MODEL_SUBDIRS)]
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


_FLAGS_LOCK = threading.Lock()


def mark_validated(provider: str, workflow: str):
    """Called after a successful real smoke test of a workflow."""
    global _LAST_GOOD_FLAGS
    with _FLAGS_LOCK:  # serialize read-modify-write (no lost updates in-proc)
        flags = _load_flags()
        flags[f"{provider}.{workflow}"] = True
        FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FLAGS_FILE.with_name(f".flags.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(flags, indent=1))
        os.replace(tmp, FLAGS_FILE)  # atomic: readers never see a torn file
        _LAST_GOOD_FLAGS = flags
