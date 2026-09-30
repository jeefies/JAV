"""Probe: start headless ComfyUI, list LTX/Gemma-related node classes, stop.
Read-only discovery for building workflow templates."""
import asyncio
import sys
import time

import requests

sys.path.insert(0, "/mnt/data/AV/JAV")
from jav import config
from jav.config import Profile
from jav.runtime.comfy import ComfyBackend


async def main():
    profile = Profile(
        name="probe", backend="comfyui", ram_budget_mb=1024, vram_budget_mb=1024,
        start_timeout_s=240, python_bin=config.CONDA_COMFYUI_PY)
    be = ComfyBackend(profile)
    await be.start()
    try:
        objs = requests.get(f"{be.base}/object_info", timeout=60).json()
        names = sorted(n for n in objs
                       if any(k in n.lower() for k in ("ltx", "gemma", "emptylatent")))
        for n in names:
            inputs = list(objs[n]["input"]["required"].keys())[:6]
            print(n, "::", inputs)
        print("PROBE_DONE", len(objs), "nodes total", flush=True)
    finally:
        await be.stop("probe_done")


if __name__ == "__main__":
    asyncio.run(main())
