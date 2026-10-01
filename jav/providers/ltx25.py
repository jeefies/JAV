"""LTX 2.5 Fast provider (ComfyUI int8 convrot distilled chain).

Graphs are derived from the official LTX-2.5 workflows
(Lightricks/ComfyUI-LTXVideo example_workflows/2.5 + shipped blueprints),
with bf16 loaders swapped for the comfy-int8-convrot files that fit
16GB VRAM + 30GB RAM.
"""
from __future__ import annotations

import json
import math

from ..models import ProviderError, eff_seed
from . import comfy_provider

WORKFLOWS = ("t2v", "i2v", "flf2v", "a2v", "bbox_control",
             "union_control", "motion_control", "inpaint", "outpaint", "ic_lora")

# IC-LoRA control workflows: media slots + allowed loras (+guide grid factor
# from the lora's reference_downscale_factor metadata * 32).
CONTROL_SLOTS = {"union_control": ("control_video",),
                 "motion_control": ("source_video",),
                 "inpaint": ("source_video", "mask_image"),
                 "outpaint": ("source_video",)}
LORA_FILES = {
    "union": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "slow_motion": "ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors",
    "ingredients": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
    "cinemagraph": "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors",
    "clean_plate": "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors",
}
LORA_MULTIPLE = {"union": 64}  # factor 2 loras need 64-px guide alignment
ALLOWED_LORAS = {"union_control": ("union",),
                 "motion_control": ("slow_motion",),
                 "inpaint": ("clean_plate",),
                 "outpaint": ("clean_plate",),
                 "ic_lora": {"reference": ("ingredients",),
                             "v2v": ("cinemagraph", "clean_plate", "ingredients",
                                     "slow_motion")}}


def _normalize_control(workflow: str, inputs: dict, generation: dict, prompt: str) -> dict:
    assets = {}
    if workflow in CONTROL_SLOTS:
        for slot in CONTROL_SLOTS[workflow]:
            if not inputs.get(slot):
                raise ProviderError(f"ltx25.{workflow}: {slot} asset required")
            assets[slot] = str(inputs[slot])
    mode = "reference"
    if workflow == "ic_lora":
        mode = str(inputs.get("mode", "reference")).lower()
        if mode not in ("reference", "v2v"):
            raise ProviderError("ltx25.ic_lora: mode must be 'reference' or 'v2v'")
        req = "reference_sheet" if mode == "reference" else "source_video"
        if not inputs.get(req):
            raise ProviderError(f"ltx25.ic_lora[{mode}]: {req} asset required")
        assets[req] = str(inputs[req])
    allowed = (ALLOWED_LORAS[workflow][mode] if workflow == "ic_lora"
               else ALLOWED_LORAS[workflow])
    lora = str(inputs.get("lora") or allowed[0])
    if lora not in allowed:
        raise ProviderError(f"ltx25.{workflow}: lora '{lora}' not in {list(allowed)}")
    gen = {"seed": int(generation.get("seed", -1)),
           "cfg": float(generation.get("cfg", 1.0)),
           "strength": float(generation.get("strength", 1.0)),
           "shorter_size": int(inputs.get("shorter_size", 512))}
    if gen["shorter_size"] % 32 or not 128 <= gen["shorter_size"] <= 768:
        raise ProviderError("ltx25 control: shorter_size must be a 32-multiple in 128..768")
    if not 0 < gen["strength"] <= 1:
        raise ProviderError("ltx25 control: strength must be in (0, 1]")
    if not 0.01 <= gen["cfg"] <= 4:
        raise ProviderError("ltx25 control: cfg must be in 0.01..4")
    out = {"provider": "ltx25", "workflow": workflow, "mode": mode, "control": True,
           "prompt": prompt, "lora": lora,
           "negative_prompt": str(inputs.get("negative_prompt",
                                             "blurry, out of focus, low contrast")),
           "generation": gen, "assets": assets}
    if workflow == "ic_lora" and mode == "reference":
        # no source video to follow: duration/fps are user params (t2v math)
        gen["fps"] = int(generation.get("fps", 24))
        dur = float(generation.get("duration", 5))
        if not (1 <= dur <= 30):
            raise ProviderError("ltx25.ic_lora[reference]: duration must be 1..30 s")
        if not (0 < gen["fps"] <= 120):
            raise ProviderError("ltx25.ic_lora[reference]: fps must be 1..120")
        gen["num_frames"] = 1 + math.floor(dur * gen["fps"] / 8) * 8
        if gen["num_frames"] < 9:
            raise ProviderError("ltx25.ic_lora[reference]: duration*fps too small")
    if workflow == "union_control":
        out["canny"] = [float(inputs.get("canny_low", 0.4)),
                        float(inputs.get("canny_high", 0.8))]
        if not (0.01 <= out["canny"][0] < out["canny"][1] <= 0.99):
            raise ProviderError("ltx25.union_control: need canny_low < canny_high in 0.01..0.99")
    if workflow == "inpaint":
        r = int(inputs.get("dilate_radius", 5))
        if not 0 <= r <= 32:
            raise ProviderError("ltx25.inpaint: dilate_radius must be 0..32")
        out["spatial_radius"] = r
    if workflow == "outpaint":
        for k in ("canvas_width", "canvas_height"):
            v = int(inputs.get(k, 0))
            if v % 32 or not 256 <= v <= 2048:
                raise ProviderError(f"ltx25.outpaint: {k} must be a 32-multiple in 256..2048")
        out["canvas"] = [int(inputs["canvas_width"]), int(inputs["canvas_height"])]
    return out


def normalize(workflow: str, inputs: dict, generation: dict) -> dict:
    if workflow not in WORKFLOWS:
        raise ProviderError(f"ltx25: unknown workflow {workflow}")
    prompt = str(inputs.get("prompt", "")).strip()
    if workflow in ALLOWED_LORAS:
        # prompt-guided control families; inpaint/outpaint/ic_lora may run
        # prompt-free (source/mask driven) per the published contract
        if not prompt and workflow in ("union_control", "motion_control"):
            raise ProviderError("ltx25: prompt is required")
        return _normalize_control(workflow, inputs, generation, prompt)
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
        fi = (inputs.get("first_image") or inputs.get("first_frame")
              or inputs.get("image"))  # docs aliases
        if not fi:
            raise ProviderError(f"ltx25.{workflow}: first_image asset required")
        assets["first_image"] = fi
    if workflow == "flf2v":
        li = inputs.get("last_image") or inputs.get("last_frame")  # docs alias
        if not li:
            raise ProviderError("ltx25.flf2v: last_image (or last_frame) asset required")
        assets["last_image"] = li
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


def _control_params(payload: dict) -> dict:
    gen = payload["generation"]
    lora = payload["lora"]
    params = {"prompt": payload["prompt"],
              "negative_prompt": payload["negative_prompt"],
              "seed": eff_seed(gen), "cfg": gen["cfg"], "strength": gen["strength"],
              "lora_name": LORA_FILES[lora],
              "guide_shorter": {"resize_type": "scale shorter dimension",
                                "shorter_size": gen["shorter_size"]},
              "guide_multiple": {"resize_type": "scale to multiple",
                                 "multiple": LORA_MULTIPLE.get(lora, 32)}}
    if payload["workflow"] == "ic_lora" and payload["mode"] == "reference":
        params.update({"length": gen["num_frames"], "audio_frames": gen["num_frames"],
                       "repeat_amount": gen["num_frames"], "fps_video": gen["fps"],
                        "cond_fps": gen["fps"], "audio_fps": gen["fps"]})
    if "canny" in payload:
        params["canny_low"], params["canny_high"] = payload["canny"]
    if "spatial_radius" in payload:
        params["spatial_radius"] = payload["spatial_radius"]
    if "canvas" in payload:
        params["canvas_width"], params["canvas_height"] = payload["canvas"]
    return params


def compile(payload: dict, asset_paths: dict[str, str], output_dir, base_dir) -> dict:
    if payload.get("control"):
        if payload["workflow"] == "ic_lora":
            tpl = "ic_lora" if payload["mode"] == "reference" else "ic_lora_v2v"
        else:
            tpl = payload["workflow"]
        graph = comfy_provider.compile_graph(
            "ltx25", tpl, _control_params(payload),
            asset_names={slot: f"asset:{slot}" for slot in asset_paths})
        return {"graph": graph, "asset_paths": asset_paths}
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
            "audio_fps": gen["fps"],
            "seed": eff_seed(gen), "cfg": gen["cfg"],
            "negative_prompt": payload["negative_prompt"],
            "regional_weight": gen.get("regional_weight", 0.85),
            "global_weight": gen.get("global_weight", 0.15),
        }
        tpl = "bbox_control.upscale" if high else "bbox_control"
        graph = comfy_provider.compile_graph("ltx25", tpl, params)
        return {"graph": graph, "asset_paths": {}}
    params = {
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "width": gen["width"], "height": gen["height"],
        "num_frames": gen["num_frames"], "audio_frames": gen["num_frames"],
        "fps": gen["fps"], "video_fps": gen["fps"], "audio_fps": gen["fps"],
        "seed": eff_seed(gen), "cfg": gen["cfg"],
        "video_cfg": gen["cfg"], "audio_cfg": gen["cfg"],
        "strength": gen["strength"], "strength2": gen["strength"],
        "width_img": gen["width"], "height_img": gen["height"],
        "width_img2": gen["width"], "height_img2": gen["height"],
        "seed2": eff_seed(gen) + 1, "cfg2": gen["cfg"],
        "trim_duration": gen["duration"],
    }
    graph = comfy_provider.compile_graph(
        "ltx25", _template_for(payload), params,
        asset_names={slot: f"asset:{slot}" for slot in asset_paths})
    return {"graph": graph, "asset_paths": asset_paths}


def _template_for(payload: dict) -> str:
    # two-stage high-res path (official two-stage for t2v/i2v; flf2v stage2
    # re-anchors both keyframes via AddGuide after the latent x2 upsample)
    if payload.get("mode") == "high" and payload["workflow"] in ("t2v", "i2v", "flf2v"):
        return payload["workflow"] + ".upscale"
    return payload["workflow"]
