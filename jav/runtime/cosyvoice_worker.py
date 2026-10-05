#!/usr/bin/env python3
"""CosyVoice3 TTS worker subprocess (配音/对白干声).

Protocol identical to zit_worker:
  stdin: one JSON task per line
  HTTP:  POST $JAV_PIPELINE_STATUS  {"status": "loaded"|"unloaded"|"error", ...}
         POST $JAV_CALLBACK         {"job_id":..., "status":..., "path":..., "meta":{...}}

Task (from providers/cosyvoice.compile):
  mode="t2a", text, instruct_text|None, voice_id, prompt_text, prompt_wav,
  speed, seed, sample_rate, output_dir

Behavior notes (verified against upstream 074ca6d):
  - 有 instruction 时走 inference_instruct2 且**必须**逐句传 prompt_wav：
    zero_shot_spk_id 路径会把 instruct 文本丢弃（frontend_instruct2 复用
    frontend_zero_shot 的 spk2info 分支），所以"固定音色 + 表演指令"不能
    用注册 spk 捷径。
  - 无 instruction 时用 add_zero_shot_spk 注册过的快速路径，参考音频特征
    只现算一次。
"""
import os

# MUST be set before torch import (same expandable_segments lesson as zit)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import gc
import time
import errno
import fcntl
import logging
from datetime import datetime
from pathlib import Path

import requests
import torch

REPO_DIR = os.getenv("CV_REPO_DIR", "/mnt/data/AV/CosyVoice")
MODEL_DIR = os.getenv("CV_MODEL_DIR", "/mnt/data/AV/models/Fun-CosyVoice3-0.5B")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "Matcha-TTS"))

import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

CALLBACK_URL = os.getenv("JAV_CALLBACK", "http://127.0.0.1:8765/v1/internal/task_complete")
PIPELINE_STATUS_URL = os.getenv(
    "JAV_PIPELINE_STATUS", "http://127.0.0.1:8765/v1/internal/pipeline_status")
CALLBACK_SECRET = os.getenv("JAV_CALLBACK_SECRET", "")
CALLBACK_HEADERS = {"X-JAV-Callback": CALLBACK_SECRET} if CALLBACK_SECRET else {}
IDLE_TIMEOUT = int(os.getenv("CV_WORKER_IDLE_TIMEOUT", "1800"))

INSTRUCT_SYSTEM = "You are a helpful assistant.<|endofprompt|>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] PID:%(process)d %(funcName)s:%(lineno)d - %(message)s")
logger = logging.getLogger("cosyvoice_worker")


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


class VoiceCache:
    """add_zero_shot_spk 结果只在进程内注册（不写 spk2info.pt：模型目录
    是只读快照，音色真相在 voices.yaml，重启后按需重建）。

    缓存键含参考文件的 (mtime,size) 与逐字稿：注册表里同名音色被 replace
    后，签名变化 → 重新提特征，不再吃进程内的陈旧 embedding。"""

    def __init__(self, cosy):
        self.cosy = cosy
        self.registered: dict[str, tuple] = {}

    @staticmethod
    def _sig(prompt_wav: str, prompt_text: str) -> tuple:
        try:
            st = os.stat(prompt_wav)
            return (st.st_mtime_ns, st.st_size, prompt_text)
        except OSError:
            return (0, 0, prompt_text)

    def ensure(self, voice_id: str, prompt_text: str, prompt_wav: str):
        sig = self._sig(prompt_wav, prompt_text)
        if self.registered.get(voice_id) == sig:
            return
        full_text = f"{INSTRUCT_SYSTEM}{prompt_text}"
        if not self.cosy.add_zero_shot_spk(full_text, prompt_wav, voice_id):
            raise RuntimeError(f"add_zero_shot_spk failed for {voice_id}")
        self.registered[voice_id] = sig
        logger.info(f"voice registered: {voice_id}")


def _soft_limit(wav: torch.Tensor, knee: float = 0.9, ceiling: float = 0.995):
    """Broadcast-grade safety net (wants.md §7 避免削波): above the knee the
    signal is tanh-compressed toward `ceiling` so PCM16 never hard-clips.
    Only touches samples already over the knee — inaudible for clean takes."""
    a = wav.abs()
    if float(a.max()) <= knee:
        return wav, 0
    over = a > knee
    span = ceiling - knee
    limited = torch.sign(wav) * (knee + span * torch.tanh((a - knee) / span))
    return torch.where(over, limited, wav), int(over.sum())


def _audio_metrics(wav: torch.Tensor, sr: int) -> dict:
    """技术验收指标（wants.md §7/§8）：峰值/响度/削波样本数。"""
    import math
    a = wav.abs()
    peak = float(a.max()) if a.numel() else 0.0
    rms = float(wav.pow(2).mean().sqrt()) if wav.numel() else 0.0
    return {
        "peak_dbfs": round(20 * math.log10(peak), 2) if peak > 0 else -120.0,
        "rms_dbfs": round(20 * math.log10(rms), 2) if rms > 0 else -120.0,
        "clipped_samples": int((a >= 0.999).sum()),
    }


def run_task(cosy: AutoModel, voices: VoiceCache, task: dict) -> tuple[str, dict]:
    text = task["text"]
    speed = float(task.get("speed", 1.0))
    seed = int(task.get("seed", 42))
    out_sr = int(task.get("sample_rate", 48000))
    dur_limit = task.get("duration_limit_s")
    prompt_wav = task["prompt_wav"]
    if not prompt_wav or not os.path.exists(prompt_wav):
        raise ValueError(f"reference audio missing: {prompt_wav!r}")
    if seed >= 0:
        set_all_random_seed(seed)
    else:
        set_all_random_seed(torch.randint(0, 2**31, (1,)).item())

    instruct = task.get("instruct_text")
    voice_id = task["voice_id"]
    tfe = bool(task.get("text_frontend", True))
    chunks = []
    if instruct:
        gen = cosy.inference_instruct2(text, instruct, prompt_wav, stream=False,
                                       speed=speed, text_frontend=tfe)
    else:
        voices.ensure(voice_id, task["prompt_text"], prompt_wav)
        gen = cosy.inference_zero_shot(text, "", "", zero_shot_spk_id=voice_id,
                                       stream=False, speed=speed, text_frontend=tfe)
    for out in gen:
        chunks.append(out["tts_speech"])
    if not chunks:
        raise RuntimeError("empty synthesis result")
    wav = torch.cat(chunks, dim=1).float()
    # 干声必须是单声道（可直接进剪辑轨道）；模型偶发多声道时坍缩到均值
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav, limited_n = _soft_limit(wav)
    model_sr = cosy.sample_rate
    if out_sr != model_sr:
        wav = torchaudio.functional.resample(wav, model_sr, out_sr)

    out_dir = Path(task.get("output_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task['job_id']}.wav"
    torchaudio.save(str(out_path), wav, out_sr, encoding="PCM_S", bits_per_sample=16)
    duration_s = round(wav.shape[1] / out_sr, 3)
    # 溢出只做报告，绝不截断/自动加速（wants.md §7：保完整句尾）
    overflow = bool(dur_limit) and duration_s > float(dur_limit)
    meta = {
        "duration_s": duration_s,
        "sample_rate": out_sr,
        "model_sample_rate": model_sr,
        "channels": int(wav.shape[0]),
        "voice_id": voice_id,
        "mode": "instruct2" if instruct else "zero_shot",
        "seed_used": seed,
        "instruction": task.get("instruction") or None,
        "speed": speed,
        "text_frontend": tfe,
        **_audio_metrics(wav, out_sr),
    }
    if limited_n:
        meta["soft_limited_samples"] = limited_n
    if dur_limit:
        meta["duration_limit_s"] = float(dur_limit)
        meta["overflow"] = overflow
    return str(out_path), meta


def main():
    try:
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except Exception:
        pass

    logger.info(f"loading CosyVoice3 from {MODEL_DIR} (repo {REPO_DIR})")
    try:
        cosy = AutoModel(model_dir=MODEL_DIR)
        voices = VoiceCache(cosy)
        send_status("loaded", sample_rate=cosy.sample_rate)
        logger.info(f"model loaded, sr={cosy.sample_rate}")
    except Exception as e:
        logger.fatal(f"model load failed: {e}")
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
                       "mode": task.get("mode", "t2a")})
        try:
            path, meta = run_task(cosy, voices, task)
            send_callback({"job_id": job_id, "status": "success", "path": path,
                           "meta": meta, "completed_at": datetime.now().isoformat()})
            _cleanup_after_generation()
        except torch.cuda.OutOfMemoryError as e:
            _cleanup_after_generation()
            send_callback({"job_id": job_id, "status": "failed",
                           "error": f"vram OOM: {e}", "error_type": "oom_error",
                           "retryable": False})
        except Exception as e:
            logger.error(f"synthesis failed: {e}")
            send_callback({"job_id": job_id, "status": "failed",
                           "error": str(e)[:500], "error_type": "generation_error",
                           "retryable": False})
        last_active = datetime.now()

    send_status("unloaded")


if __name__ == "__main__":
    main()
