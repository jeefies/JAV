"""LTX 2.5 Fast provider (ComfyUI int8 convrot distilled chain).

Graphs are derived from the official LTX-2.5 workflows
(Lightricks/ComfyUI-LTXVideo example_workflows/2.5 + shipped blueprints),
with bf16 loaders swapped for the comfy-int8-convrot files that fit
16GB VRAM + 30GB RAM.
"""
from __future__ import annotations

import json
import math

from ..models import ProviderError
from . import comfy_provider

WORKFLOWS = ("t2v", "i2v", "flf2v", "a2v", "bbox_control")


def normalize(workflow: str, inputs: dict, generation: dict) -> dict:
    if workflow not in WORKFLOWS:
        raise ProviderError(f"ltx25: unknown workflow {workflow}")
    prompt = str(inputs.get("prompt", "")).strip()
    if not prompt and workflow != "bbox_control":
        raise ProviderError("ltx25: prompt is required")
    bbox_project = None
    if workflow == "bbox_control":
        bbox_project = inputs.get("bbox_project")
        if not isinstance(bbox_project, dict) or not bbox_project.get("objects"):
            raise ProviderError("ltx25.bbox_control: bbox_project with objects[] required")
    mode = str(generation.get("mode", "fast")).lower()
    if mode not in ("fast", "high"):
        raise ProviderError("ltx25: mode must be 'fast' or 'high'")
    assets = {}
    if workflow in ("i2v", "flf2v"):
        if not inputs.get("first_image"):
            raise ProviderError(f"ltx25.{workflow}: first_image asset required")
        assets["first_image"] = inputs["first_image"]
    if workflow == "flf2v":
        if not inputs.get("last_image"):
            raise ProviderError("ltx25.flf2v: last_image asset required")
        assets["last_image"] = inputs["last_image"]
    if workflow == "a2v":
        if not inputs.get("audio"):
            raise ProviderError("ltx25.a2v: audio asset required")
        assets["audio"] = inputs["audio"]
    gen = {"width": 960, "height": 544, "duration": 5, "fps": 24, "cfg": 1.0,
           "strength": 0.7, "seed": int(generation.get("seed", -1))}
    for k in ("width", "height", "duration", "fps", "cfg", "strength"):
        if k in generation:
            gen[k] = generation[k]
    for k in ("width", "height"):
        gen[k] = int(gen[k])
        if gen[k] <= 0 or gen[k] % 32:
            raise ProviderError(f"ltx25: {k} must be a positive multiple of 32")
    dur, fps = float(gen["duration"]), int(gen["fps"])
    if not (1 <= dur <= 30):
        raise ProviderError("ltx25: duration must be 1..30 s")
    if fps <= 0 or fps > 120:
        raise ProviderError("ltx25: fps must be 1..120")
    gen["num_frames"] = 1 + math.floor(dur * fps / 8) * 8
    if gen["num_frames"] < 9:
        raise ProviderError("ltx25: duration*fps too small for >=1 sampled frame")
    return {"provider": "ltx25", "workflow": workflow, "mode": mode, "prompt": prompt,
            "bbox_project": bbox_project,
            "negative_prompt": str(inputs.get("negative_prompt",
                                              "blurry, out of focus, low contrast")),
            "generation": gen, "assets": assets}


def asset_slots(payload: dict) -> dict[str, str]:
    return dict(payload["assets"])


def cache_key(payload: dict) -> str | None:
    return None


def compile(payload: dict, asset_paths: dict[str, str], output_dir, base_dir) -> dict:
    gen = payload["generation"]
    if payload["workflow"] == "bbox_control":
        high = payload.get("mode") == "high"
        params = {
            "bbox_json": json.dumps(payload["bbox_project"], ensure_ascii=False),
            "width": gen["width"], "height": gen["height"],
            "total_frames": gen["num_frames"], "num_frames": gen["num_frames"],
            "width_lat": gen["width"], "height_lat": gen["height"],
            "audio_frames": gen["num_frames"],
            "width_up": gen["width"] * 2, "height_up": gen["height"] * 2,
            "fps": gen["fps"], "animator_fps": gen["fps"], "video_fps": gen["fps"],
            "seed": gen["seed"], "cfg": gen["cfg"],
            "negative_prompt": payload["negative_prompt"],
            "regional_weight": gen.get("regional_weight", 0.85),
            "global_weight": gen.get("global_weight", 0.15),
        }
        tpl = "bbox_control.upscale" if high else "bbox_control"
        graph = comfy_provider.compile_graph("ltx25", tpl, params)
        return {"graph": graph, "asset_paths": {}, "output_kinds": ["video"]}
    params = {
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "width": gen["width"], "height": gen["height"],
        "num_frames": gen["num_frames"], "audio_frames": gen["num_frames"],
        "fps": gen["fps"], "video_fps": 30, "seed": gen["seed"], "cfg": gen["cfg"],
        "video_cfg": gen["cfg"], "audio_cfg": gen["cfg"],
        "strength": gen["strength"], "strength2": gen["strength"],
        "width_img": gen["width"], "height_img": gen["height"],
        "width_img2": gen["width"], "height_img2": gen["height"],
        "seed2": gen["seed"] + 1, "cfg2": gen["cfg"],
        "trim_duration": gen["duration"],
    }
    graph = comfy_provider.compile_graph(
        "ltx25", _template_for(payload), params,
        asset_names={slot: f"asset:{slot}" for slot in asset_paths})
    return {"graph": graph, "asset_paths": asset_paths,
            "output_kinds": comfy_provider.output_kinds("ltx25", _template_for(payload))}


def _template_for(payload: dict) -> str:
    # two-stage high-res path (official two-stage for t2v/i2v; flf2v stage2
    # re-anchors both keyframes via AddGuide after the latent x2 upsample)
    if payload.get("mode") == "high" and payload["workflow"] in ("t2v", "i2v", "flf2v"):
        return payload["workflow"] + ".upscale"
    return payload["workflow"]
