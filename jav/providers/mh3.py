"""MiniMax-H3 fl2va provider (ComfyUI int8 convrot + nvfp4 Qwen3-VL TE).

Chain replicates the official "Image to Video (MiniMax H3)" blueprint.
t2v / i2v / fl2v share the single MiniMaxH3ImageToVideo node
(first_frame/last_frame optional); ref2v template lands with mh3.ref2va
weights (not downloaded yet — gated by capabilities).
"""
from __future__ import annotations

from ..models import ProviderError
from . import comfy_provider

WORKFLOWS = ("t2v", "i2v", "fl2v", "ref2v", "fun_control", "multiframe")
TURBO_LORA = {
    "fl2va": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "ref2va": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}
TURBO_STEPS = {"t2v": 8, "i2v": 8, "fl2v": 8, "ref2v": 4, "multiframe": 4}
UNET_NODE = {"t2v": "135", "i2v": "135", "fl2v": "135", "ref2v": "127",
             "fun_control": "127", "multiframe": "127"}


def _align_length(frames: int) -> int:
    # H3 latent stride: length ≡ 5 (mod 17)
    return frames + (5 - frames % 17) % 17


def normalize(workflow: str, inputs: dict, generation: dict) -> dict:
    if workflow not in WORKFLOWS:
        raise ProviderError(f"mh3: unknown workflow {workflow}")
    prompt = str(inputs.get("prompt", "")).strip()
    if not prompt:
        raise ProviderError("mh3: prompt is required")
    assets = {}
    if workflow in ("i2v", "fl2v"):
        if not inputs.get("first_frame"):
            raise ProviderError(f"mh3.{workflow}: first_frame asset required")
        assets["first_frame"] = inputs["first_frame"]
    if workflow == "fl2v":
        if not inputs.get("last_frame"):
            raise ProviderError("mh3.fl2v: last_frame asset required")
        assets["last_frame"] = inputs["last_frame"]
    refs = inputs.get("reference_images") or []
    if workflow in ("ref2v", "multiframe"):
        if not refs:
            raise ProviderError(f"mh3.{workflow}: reference_images (1-9 asset ids) required")
        if len(refs) > 9:
            raise ProviderError(f"mh3.{workflow}: max 9 reference images")
        for i, aid in enumerate(refs):
            assets[f"ref_image.{i}"] = aid
    keyframes = inputs.get("keyframes") or []
    if workflow == "multiframe":
        if not keyframes:
            raise ProviderError("mh3.multiframe: keyframes [{image|video, time}] required")
        if len(keyframes) > 8:
            raise ProviderError("mh3.multiframe: max 8 keyframes")
        for i, kf in enumerate(keyframes):
            src = kf.get("image") or kf.get("video")
            if not src:
                raise ProviderError(f"mh3.multiframe: keyframes[{i}] needs image or video")
            assets[f"keyframe.{i}"] = src
    if workflow == "fun_control":
        if not inputs.get("control_video"):
            raise ProviderError("mh3.fun_control: control_video asset required")
        assets["control_video"] = inputs["control_video"]
    gen = {"width": 768, "height": 416, "duration": 5, "fps": 24,
           "steps": 20 if workflow in ("fun_control", "multiframe") else 4,
           "strength": 1.0 if workflow == "fun_control" else 0.7,
           "seed": int(generation.get("seed", -1))}
    if workflow == "fun_control" and inputs.get("control_strength") is not None:
        # documented in API.md: inputs.control_strength (generation.strength
        # below still wins when both are given)
        try:
            cs = float(inputs["control_strength"])
        except (TypeError, ValueError):
            raise ProviderError("mh3.fun_control: control_strength must be numeric")
        if not (0.0 <= cs <= 2.0):
            raise ProviderError("mh3.fun_control: control_strength must be 0..2")
        gen["strength"] = cs
    for k in ("width", "height", "duration", "fps", "steps", "strength"):
        if k in generation:
            gen[k] = generation[k]
    for k in ("width", "height"):
        gen[k] = int(gen[k])
        if gen[k] <= 0 or gen[k] % 32:
            raise ProviderError(f"mh3: {k} must be a positive multiple of 32")
    dur = float(gen["duration"])
    if not (1 <= dur <= 15):
        raise ProviderError("mh3: duration must be 1..15 s")
    if int(gen["fps"]) <= 0 or int(gen["fps"]) > 60:
        raise ProviderError("mh3: fps must be 1..60")
    turbo = bool(inputs.get("turbo"))
    if turbo:
        if workflow not in TURBO_STEPS:
            raise ProviderError(f"mh3.{workflow}: turbo path not available")
        gen["steps"] = TURBO_STEPS[workflow]
    gen["length"] = _align_length(max(5, round(dur * int(gen["fps"]))))
    payload = {"provider": "mh3", "workflow": workflow, "prompt": prompt,
               "generation": gen, "assets": assets, "turbo": turbo}
    if workflow == "multiframe":
        payload["keyframe_times"] = [float(kf.get("time", 0)) for kf in keyframes]
    return payload


def asset_slots(payload: dict) -> dict[str, str]:
    return dict(payload["assets"])


def cache_key(payload: dict) -> str | None:
    return None


def _inject_turbo_lora(graph: dict, workflow: str):
    """Insert LoraLoaderModelOnly between the UNET and the sampler for the
    official turbo path (fl2v 8-step / ref2v 4-step)."""
    unet = UNET_NODE[workflow]
    lora_file = TURBO_LORA["ref2va" if workflow == "ref2v" else "fl2va"]
    graph["900"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": [unet, 0], "lora_name": lora_file,
                               "strength_model": 1.0}}
    for node in graph.values():
        if node is graph["900"]:
            continue
        for k, v in list(node["inputs"].items()):
            if isinstance(v, list) and v == [unet, 0]:
                node["inputs"][k] = ["900", 0]


def compile(payload: dict, asset_paths: dict[str, str], output_dir, base_dir) -> dict:
    gen = payload["generation"]
    params = {"prompt": payload["prompt"], "width": gen["width"],
              "height": gen["height"], "length": gen["length"],
              "steps": gen["steps"], "seed": gen["seed"], "fps": gen["fps"],
              "width_img": gen["width"], "height_img": gen["height"],
              "strength": gen.get("strength", 0.7)}
    if payload["workflow"] == "fl2v":
        params["width_img2"] = gen["width"]
        params["height_img2"] = gen["height"]
    wf_key = "ref2v" if payload["workflow"] == "multiframe" else payload["workflow"]
    graph = comfy_provider.compile_graph(
        "mh3", wf_key, params,
        asset_names={slot: f"asset:{slot}" for slot in asset_paths})
    if payload.get("turbo"):
        _inject_turbo_lora(graph, payload["workflow"])
    if payload["workflow"] in ("ref2v", "multiframe"):
        refs = sorted((s for s in asset_paths if s.startswith("ref_image.")),
                      key=lambda s: int(s.split(".")[1]))
        autogrow = {}
        for i, slot in enumerate(refs):
            nid = f"3{i}"
            graph[nid] = {"class_type": "LoadImage",
                          "inputs": {"image": f"asset:{slot}"}}
            autogrow[f"ref_image_{i}"] = [nid, 0]
        graph["136"]["inputs"]["ref_images"] = autogrow
    if payload["workflow"] == "multiframe":
        cond_src = ["136", 0]
        for i, t in enumerate(payload.get("keyframe_times", [])):
            frame = max(0, min(gen["length"] - 1, round(t * int(gen["fps"]))))
            img, guide = f"4{i}", f"5{i}"
            src = f"asset:keyframe.{i}"
            if asset_paths.get(f"keyframe.{i}", "").lower().endswith(
                    (".mp4", ".mov", ".webm", ".mkv")):
                vid, comp = f"6{i}", f"7{i}"
                graph[vid] = {"class_type": "LoadVideo", "inputs": {"file": src}}
                graph[comp] = {"class_type": "GetVideoComponents",
                               "inputs": {"video": [vid, 0]}}
                img_node = [comp, 0]
            else:
                graph[img] = {"class_type": "LoadImage", "inputs": {"image": src}}
                img_node = [img, 0]
            graph[guide] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
                "positive": cond_src, "latent": ["136", 1],
                "vae": ["119", 0], "audio_vae": ["120", 0],
                "image": img_node, "frame_idx": frame}}
            cond_src = [guide, 0]
        graph["126"]["inputs"]["conditioning"] = cond_src
    return {"graph": graph, "asset_paths": asset_paths,
            "output_kinds": ["video"]}
