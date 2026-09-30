"""Z-Image-Turbo provider: t2i / i2i / inpaint on the diffusers worker.

Semantics preserved 1:1 from the validated ZIT-service implementation
(t2i base pipeline + derived img2img/inpaint, full-white mask rule,
strength in the i2i cache key).
"""
from __future__ import annotations

from .. import config
from ..store import cache_hash

DEFAULTS = {
    "width": int(1024),
    "height": int(1024),
    "steps": 9,
    "guidance": 0.0,
    "strength": 0.8,
    "negative_prompt": "",
}
IMAGE_EXT = {"image/png", "image/jpeg", "image/webp", "image/bmp"}


def normalize(workflow: str, inputs: dict, generation: dict) -> dict:
    if workflow not in ("t2i", "i2i", "inpaint"):
        raise ValueError(f"zit: unknown workflow {workflow}")
    gen = {**DEFAULTS, **{k: v for k, v in generation.items()
                          if k in DEFAULTS or k == "seed"}}
    gen["seed"] = int(generation.get("seed", -1))
    for key in ("width", "height", "steps"):
        gen[key] = int(gen[key])
        if gen[key] <= 0:
            raise ValueError(f"zit: {key} must be positive")
    for key in ("guidance", "strength"):
        gen[key] = float(gen[key])
    if workflow == "i2i" and not 0 < gen["strength"] <= 1:
        raise ValueError("zit.i2i: strength must be in (0, 1]")
    prompt = str(inputs.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("zit: prompt is required")

    assets: dict[str, str] = {}
    if workflow in ("i2i", "inpaint"):
        if not inputs.get("image"):
            raise ValueError(f"zit.{workflow}: 'image' asset_id is required")
        assets["image"] = str(inputs["image"])
    if workflow == "inpaint":
        if not inputs.get("mask"):
            raise ValueError("zit.inpaint: 'mask' asset_id is required")
        assets["mask"] = str(inputs["mask"])
    return {
        "provider": "zit", "workflow": workflow, "prompt": prompt,
        "negative_prompt": str(inputs.get("negative_prompt", gen["negative_prompt"])),
        "generation": gen, "assets": assets,
    }


def asset_ids(payload: dict) -> list[str]:
    return list(payload["assets"].values())


def cache_key(payload: dict) -> str | None:
    """Deterministic requests only (seed >= 0), mirroring old ZIT behavior
    but excluding the seed=-1 false-hit quirk."""
    gen = payload["generation"]
    if int(gen.get("seed", -1)) < 0:
        return None
    return cache_hash({
        "provider": "zit", "workflow": payload["workflow"],
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "width": gen["width"], "height": gen["height"],
        "steps": gen["steps"], "guidance": gen["guidance"],
        "strength": gen["strength"], "seed": gen["seed"],
        "assets": payload["assets"],
    })


def asset_slots(payload: dict) -> dict[str, str]:
    """slot name -> asset id"""
    return dict(payload["assets"])


def compile(payload: dict, asset_paths: dict[str, str], output_dir, base_dir) -> dict:
    """asset_paths: slot name -> local file path"""
    gen = payload["generation"]
    task = {
        "job_id": payload["job_id"],
        "mode": payload["workflow"],
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "width": gen["width"], "height": gen["height"],
        "steps": gen["steps"], "guidance": gen["guidance"],
        "strength": gen["strength"], "seed": int(gen["seed"]),
        "output_dir": str(output_dir),
    }
    if "image" in asset_paths:
        task["input_image_path"] = asset_paths["image"]
    if "mask" in asset_paths:
        task["mask_path"] = asset_paths["mask"]
    return task
