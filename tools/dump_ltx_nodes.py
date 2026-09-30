"""Dump full object_info for LTX-related nodes to tools/ltx_nodes_info.json."""
import asyncio
import json
import sys

import requests

sys.path.insert(0, "/mnt/data/AV/JAV")
from jav import config
from jav.config import Profile
from jav.runtime.comfy import ComfyBackend

WANT = ["LTXVGemmaCLIPModelLoader", "LTXVBaseSampler", "LTXVConditioning",
        "LTXVScheduler", "CLIPTextEncode", "VAELoader", "UNETLoader",
        "LTXVAudioVAELoader", "LTXVEmptyLatentAudio", "LTXVImgToVideo",
        "LTXVImgToVideoConditionOnly", "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
        "KSamplerSelect", "RandomNoise", "CFGGuider", "VAEDecode", "SaveVideo",
        "VHS_VideoCombine", "LTXVSpatioTemporalTiledVAEDecode", "LtxvApiTextToVideo",
        "LtxvApiImageToVideo", "EmptyLTXVLatentVideo", "LTXVLatentUpsampler",
        "LTXVSetAudioVideoMaskByTime", "LTXVCropGuides"]


async def main():
    profile = Profile(name="probe", backend="comfyui", ram_budget_mb=1024,
                      vram_budget_mb=1024, start_timeout_s=240,
                      python_bin=config.CONDA_COMFYUI_PY)
    be = ComfyBackend(profile)
    await be.start()
    try:
        objs = requests.get(f"{be.base}/object_info", timeout=120).json()
        out = {}
        for n in WANT:
            if n in objs:
                info = objs[n]
                out[n] = {
                    "required": info["input"].get("required"),
                    "optional": info["input"].get("optional"),
                    "output": info.get("output"),
                    "output_name": info.get("output_name"),
                }
            else:
                out[n] = "MISSING"
        with open("/mnt/data/AV/JAV/tools/ltx_nodes_info.json", "w") as f:
            json.dump(out, f, indent=1)
        print("DUMPED", len(out), "nodes", flush=True)
        print("MISSING:", [k for k, v in out.items() if v == "MISSING"], flush=True)
    finally:
        await be.stop("dump_done")


if __name__ == "__main__":
    asyncio.run(main())
