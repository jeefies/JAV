"""Dynamic capability discovery (JAV-DESIGN 5): weights present + manifest
exists + operator flag. Interface existing != hardware validated."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import config
from .models import PROVIDER_WORKFLOWS

# Workflows whose provider templates are implemented in code.
IMPLEMENTED = {"zit.t2i", "zit.i2i", "zit.inpaint",
               "ltx25.t2v", "ltx25.i2v", "ltx25.flf2v", "ltx25.a2v", "ltx25.ia2v", "ltx25.bbox_control",
               "ltx25.union_control", "ltx25.motion_control", "ltx25.inpaint",
               "ltx25.outpaint", "ltx25.ic_lora",
               "mh3.t2v", "mh3.i2v", "mh3.fl2v", "mh3.ref2v", "mh3.fun_control",
               "mh3.multiframe", "cosyvoice.t2a"}

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


def _template_files(provider: str, workflows: list[str] | None = None) -> set[str]:
    """Every .safetensors referenced by the provider's workflow templates
    (any *_name loader key of any node: unet/vae/clip/lora/model_name/name),
    so the weight gate can never green-light a workflow whose loader target
    was purged. workflows=None sweeps ALL *.api.json in the provider dir —
    variant templates (upscale/v2v) gate exactly like base ones."""
    keys = ("vae_name", "unet_name", "clip_name", "lora_name", "model_name", "name")
    names: set[str] = set()
    root = config.BASE_DIR / "jav" / "workflows" / provider
    tpls = (sorted(root.glob("*.api.json")) if workflows is None
            else [root / f"{wf}.api.json" for wf in workflows])
    for tpl in tpls:
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


_MH3_MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras",
                      "checkpoints", "model_patches", "latent_upscale_models")


def _weights_for(profile: str) -> tuple[bool, str]:
    """Return (present, detail) for the profile's big weights on disk."""
    if profile == "zit":
        ok = config.ZIT_WEIGHTS_DIR.is_dir() and any(
            (config.ZIT_WEIGHTS_DIR / p).exists()
            for p in ("model_index.json", "transformer"))
        return ok, "Z-Image-Turbo diffusers snapshot"
    if profile == "ltx25":
        m = config.COMFYUI_DIR / "models"
        need = _template_files("ltx25")  # full sweep incl. .upscale/.v2v variants:
        # ic_lora_v2v (cinemagraph lora) and the upscale models are loader targets
        # of shipped graphs and must gate the provider too (fail-closed over-gating
        # is the safe direction for a per-profile gate).
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
    if profile == "cosyvoice":
        m = config.COSYVOICE_WEIGHTS_DIR
        r = config.COSYVOICE_REPO_DIR
        need = ["cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt",
                "campplus.onnx", "speech_tokenizer_v3.onnx"]
        missing = [n for n in need if not (m / n).exists()]
        if not (m / "CosyVoice-BlankEN").is_dir():
            missing.append("CosyVoice-BlankEN/")
        if not (r / "cosyvoice").is_dir():
            missing.append("repo:cosyvoice/")
        if not (r / "third_party" / "Matcha-TTS").is_dir():
            missing.append("repo:third_party/Matcha-TTS/")
        return not missing, f"CosyVoice3 files missing {missing}"
    return False, "unknown profile"


_CAPS_TTL_S = 5.0
_CAPS_CACHE: tuple[object, float, dict] | None = None


def capabilities() -> dict:
    # Cached (flags-mtime + short TTL): POST /v1/jobs calls is_available per
    # job (batch = 64x) and a full disk re-scan on the event loop is waste.
    # Invalidation on FLAGS_FILE mtime keeps operator/test writes immediate.
    global _CAPS_CACHE
    now = time.monotonic()
    fkey = FLAGS_FILE.stat().st_mtime_ns if FLAGS_FILE.exists() else None
    if (_CAPS_CACHE is not None and _CAPS_CACHE[0] == fkey
            and now - _CAPS_CACHE[1] < _CAPS_TTL_S):
        return _CAPS_CACHE[2]
    out = _capabilities_compute()
    _CAPS_CACHE = (fkey, now, out)
    return out


def _capabilities_compute() -> dict:
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
    if "cosyvoice" in out:
        out["cosyvoice"]["model"] = _cosyvoice_model_info()
    return out


def _cosyvoice_weights_fingerprint() -> str | None:
    """llm.pt sha from the registered weights manifest (audit_weights output)."""
    try:
        data = json.loads((Path(config.DATA_DIR) / "weights.json").read_text())
        key = str(config.COSYVOICE_WEIGHTS_DIR / "llm.pt")
        return (data.get(key) or {}).get("sha256", "")[:12] or None
    except Exception:
        return None


def _cosyvoice_model_info() -> dict:
    import subprocess
    try:
        code_rev = subprocess.run(
            ["git", "-C", str(config.COSYVOICE_REPO_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        code_rev = None
    return {
        "name": config.COSYVOICE_MODEL_NAME,
        "weights": config.COSYVOICE_MODEL_REPO,
        "weights_fingerprint": _cosyvoice_weights_fingerprint(),
        "code_revision": code_rev,
        "native_sample_rate": config.COSYVOICE_NATIVE_SR,
        "voice_cloning": "zero-shot (10-15 s reference take + verbatim transcript)",
        "features": {
            "instruction": True,       # inference_instruct2, per-sentence delivery control
            "pinyin_hotfix": True,     # [j][ǐ] inline polyphone marks pass through verbatim
            "text_frontend": True,     # wetext normalizes numbers/abbreviations
            "mono_lossless_wav": True,
        },
        "voice_kinds": ["local", "community", "cloud"],
    }


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
    key = f"{provider}.{workflow}"
    with _FLAGS_LOCK:  # serialize read-modify-write (no lost updates in-proc)
        flags = _load_flags()
        if flags.get(key):
            return  # already validated: skip the per-completion no-op rewrite
        flags[key] = True
        FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FLAGS_FILE.with_name(f".flags.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(flags, indent=1))
        os.replace(tmp, FLAGS_FILE)  # atomic: readers never see a torn file
        _LAST_GOOD_FLAGS = flags
