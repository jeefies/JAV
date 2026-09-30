"""ComfyBackend E2E smoke: headless ComfyUI launch -> upload -> /prompt ->
history -> collect outputs -> clean stop. Model-free (LoadImage/SaveImage)
so it exercises the full RPC path without any large weights."""
import asyncio
import json
import subprocess
import sys

sys.path.insert(0, "/mnt/data/AV/JAV")

from jav import config
from jav.config import Profile
from jav.runtime.comfy import ComfyBackend


def vram_apps():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return out


async def main():
    profile = Profile(
        name="smoke", backend="comfyui",
        ram_budget_mb=6144, vram_budget_mb=4096,
        start_timeout_s=240, job_timeout_s=180, idle_unload_s=60,
        python_bin=config.CONDA_COMFYUI_PY,
        script="", required_nodes=("LoadImage", "SaveImage"))
    be = ComfyBackend(profile)
    print("starting comfyui...", flush=True)
    await be.start()
    try:
        print(f"comfyui up: pid={be.pid}", flush=True)

        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "asset:src"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0],
                                                        "filename_prefix": "javsmoke"}},
        }
        task = {"graph": graph,
                "asset_paths": {"src": "/tmp/kilo/i2i_input.png"},
                "output_kinds": ["image"]}
        result = await be.submit("smoke_job_1", task)
        print("submit result:", json.dumps(result), flush=True)
        assert result["status"] == "success" and result["paths"], "smoke job failed"
        import os
        for p in result["paths"]:
            print("output file:", p, os.path.getsize(p), "bytes", flush=True)
        import requests
        stats = requests.get(f"{be.base}/system_stats", timeout=5).json()
        print("system_stats ok, devices:", list(stats.get("devices", {}))[:2], flush=True)
    finally:
        await be.stop("smoke_done")
        print("stopped; alive:", be.alive(), flush=True)
        await asyncio.sleep(3)
        print("vram apps after stop:\n" + vram_apps(), flush=True)
        print("COMFY_SMOKE_OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
