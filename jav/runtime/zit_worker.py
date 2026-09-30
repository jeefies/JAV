#!/usr/bin/env python3
"""ZIT diffusers worker subprocess (migrated from ZIT-service
image_service_pipeline.py — validated t2i/i2i/inpaint logic preserved 1:1).

Protocol (unchanged in spirit):
  stdin: one JSON task per line
  HTTP:  POST $JAV_PIPELINE_STATUS  {"status": "loaded"|"unloaded"|"error", ...}
         POST $JAV_CALLBACK         {"job_id": ..., "status": "processing"|"success"|"failed", ...}
"""
import os
import sys
import json
import gc
import time
import errno
import fcntl
import logging
import requests
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from diffusers import ZImagePipeline, ZImageInpaintPipeline, ZImageImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

MODEL_DIR = os.getenv("ZIT_MODEL_DIR", "/mnt/data/AV/models/Z-Image-Turbo")
CALLBACK_URL = os.getenv("JAV_CALLBACK", "http://127.0.0.1:8765/v1/internal/task_complete")
PIPELINE_STATUS_URL = os.getenv(
    "JAV_PIPELINE_STATUS", "http://127.0.0.1:8765/v1/internal/pipeline_status")
CALLBACK_SECRET = os.getenv("JAV_CALLBACK_SECRET", "")
CALLBACK_HEADERS = {"X-JAV-Callback": CALLBACK_SECRET} if CALLBACK_SECRET else {}
IDLE_TIMEOUT = int(os.getenv("ZIT_WORKER_IDLE_TIMEOUT", "1800"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] PID:%(process)d %(funcName)s:%(lineno)d - %(message)s")
logger = logging.getLogger("zit_worker")


def send_callback(result):
    try:
        requests.post(CALLBACK_URL, json=result, headers=CALLBACK_HEADERS, timeout=10)
    except Exception as e:
        print(f"callback failed: {e}", file=sys.stderr, flush=True)


def send_status(status, **extra):
    try:
        requests.post(PIPELINE_STATUS_URL,
                      json={"status": status, "timestamp": datetime.now().isoformat(), **extra},
                      headers=CALLBACK_HEADERS, timeout=5)
    except Exception as e:
        logger.warning(f"status notify failed: {e}")


def _cleanup_after_generation():
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("vram cleaned", file=sys.stderr, flush=True)


def load_pipeline():
    """Base = ZImagePipeline for t2i; i2i/inpaint derived lazily via from_pipe.
    Derived pipelines MUST be cast back to bf16 (fp32 Qwen3 TE ~15GB OOMs on
    16GB cards); offload hooks belong to one pipeline at a time."""
    pipe = ZImagePipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=False)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("auto")
    derived = {}

    def _derive(cls):
        p = derived.get(cls.__name__)
        if p is None:
            p = cls.from_pipe(pipe).to(torch.bfloat16)
            derived[cls.__name__] = p
        p.enable_model_cpu_offload()
        return p

    return pipe, _derive


def run_task(pipe, derive, task):
    mode = task.get("mode", "t2i")
    seed = int(task.get("seed", -1))
    generator = torch.Generator("cuda").manual_seed(
        seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item())
    width, height = int(task["width"]), int(task["height"])
    steps = int(task["steps"])
    guidance = float(task["guidance"])

    if mode == "t2i":
        pipe.enable_model_cpu_offload()
        image = pipe(prompt=task["prompt"],
                     negative_prompt=task.get("negative_prompt") or None,
                     width=width, height=height,
                     num_inference_steps=steps, guidance_scale=guidance,
                     generator=generator).images[0]
    elif mode == "i2i":
        input_path = task.get("input_image_path")
        if not input_path:
            raise ValueError("i2i task missing input image path")
        image = load_image(input_path).convert("RGB").resize((width, height), Image.LANCZOS)
        mask = None
        use_img2img = True
        mask_path = task.get("mask_path")
        if mask_path and os.path.exists(mask_path):
            mask = load_image(mask_path).convert("L").resize((width, height), Image.LANCZOS)
            # full-white mask -> img2img (strength); partial -> true inpaint
            use_img2img = bool(np.asarray(mask).min() >= 250)
        if use_img2img:
            p = derive(ZImageImg2ImgPipeline)
            image = p(image=image, prompt=task["prompt"],
                      negative_prompt=task.get("negative_prompt") or None,
                      width=width, height=height,
                      strength=float(task.get("strength", 0.8)),
                      num_inference_steps=steps, guidance_scale=guidance,
                      generator=generator).images[0]
        else:
            p = derive(ZImageInpaintPipeline)
            image = p(image=image, mask_image=mask, prompt=task["prompt"],
                      negative_prompt=task.get("negative_prompt") or None,
                      width=width, height=height,
                      num_inference_steps=steps, guidance_scale=guidance,
                      generator=generator).images[0]
    elif mode == "inpaint":
        input_path = task.get("input_image_path")
        mask_path = task.get("mask_path")
        if not input_path:
            raise ValueError("inpaint task missing input image path")
        if not mask_path or not os.path.exists(mask_path):
            raise ValueError("inpaint task missing mask path")
        image = load_image(input_path).convert("RGB").resize((width, height), Image.LANCZOS)
        mask = load_image(mask_path).convert("L").resize((width, height), Image.LANCZOS)
        p = derive(ZImageInpaintPipeline)
        image = p(image=image, mask_image=mask, prompt=task["prompt"],
                  negative_prompt=task.get("negative_prompt") or None,
                  width=width, height=height,
                  num_inference_steps=steps, guidance_scale=guidance,
                  generator=generator).images[0]
    else:
        raise ValueError(f"unknown mode {mode}")

    out_dir = Path(task.get("output_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task['job_id']}.png"
    image.save(out_path, "PNG")
    return str(out_path)


def main():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except Exception:
        pass

    logger.info(f"loading pipeline from {MODEL_DIR}")
    try:
        pipe, derive = load_pipeline()
        send_status("loaded")
        logger.info("pipeline loaded")
    except Exception as e:
        logger.fatal(f"pipeline load failed: {e}")
        send_status("error", error=str(e)[:500])
        return

    last_active = datetime.now()
    while True:
        if (datetime.now() - last_active).total_seconds() > IDLE_TIMEOUT:
            logger.info("idle timeout, exiting")
            break
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.2)
                continue
            task = json.loads(line.strip())
        except (IOError, OSError) as e:
            if getattr(e, "errno", None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                time.sleep(0.2)
                continue
            logger.error(f"stdin error: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"bad json: {e}")
            continue
        except Exception as e:
            logger.error(f"loop error: {e}")
            time.sleep(0.2)
            continue

        job_id = task.get("job_id")
        last_active = datetime.now()
        send_callback({"job_id": job_id, "status": "processing",
                       "mode": task.get("mode", "t2i")})
        try:
            path = run_task(pipe, derive, task)
            send_callback({"job_id": job_id, "status": "success", "path": path,
                           "completed_at": datetime.now().isoformat()})
            _cleanup_after_generation()
        except torch.cuda.OutOfMemoryError as e:
            _cleanup_after_generation()
            send_callback({"job_id": job_id, "status": "failed",
                           "error": f"vram OOM: {e}", "error_type": "oom_error",
                           "retryable": False})
        except Exception as e:
            logger.error(f"generation failed: {e}")
            send_callback({"job_id": job_id, "status": "failed",
                           "error": str(e)[:500], "error_type": "generation_error",
                           "retryable": False})
        last_active = datetime.now()

    send_status("unloaded")


if __name__ == "__main__":
    main()
