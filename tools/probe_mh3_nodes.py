"""Probe: list MiniMax-H3 / Qwen3VL related nodes in current ComfyUI build."""
import asyncio
import json
import sys

import requests

sys.path.insert(0, "/mnt/data/AV/JAV")
from jav import config
from jav.config import Profile
from jav.runtime.comfy import ComfyBackend


async def main():
    be = ComfyBackend(Profile(name="probe", backend="comfyui", ram_budget_mb=1024,
                              vram_budget_mb=1024, start_timeout_s=300,
                              python_bin=config.CONDA_COMFYUI_PY))
    await be.start()
    try:
        objs = requests.get(f"{be.base}/object_info", timeout=120).json()
        hits = sorted(n for n in objs if any(k in n.lower() for k in ("minimax", "h3", "qwen3vl")))
        print(json.dumps({n: {
            "required": list(objs[n]["input"].get("required", {}).keys()),
            "optional": list(objs[n]["input"].get("optional", {}).keys()),
            "output": objs[n].get("output")} for n in hits}, indent=1)[:6000])
        print("PROBE2_DONE", len(hits), "nodes", flush=True)
    finally:
        await be.stop("probe2")


if __name__ == "__main__":
    asyncio.run(main())
