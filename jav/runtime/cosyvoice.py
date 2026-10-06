"""CosyVoiceBackend: manages the CosyVoice3 TTS worker subprocess
(same stdin/callback protocol as ZitBackend, separate interpreter + repo)."""
from __future__ import annotations

from .. import config
from .base import BackendCrash
from .subprocess_worker import SubprocessWorkerBackend


class CosyVoiceBackend(SubprocessWorkerBackend):
    kind = "cosyvoice_subprocess"
    label = "cosyvoice"

    def preflight(self):
        if not (config.COSYVOICE_REPO_DIR / "cosyvoice").is_dir():
            raise BackendCrash(f"CosyVoice repo missing at {config.COSYVOICE_REPO_DIR}")
        if not (config.COSYVOICE_WEIGHTS_DIR / "cosyvoice3.yaml").is_file():
            raise BackendCrash(f"CosyVoice3 weights missing at {config.COSYVOICE_WEIGHTS_DIR}")
        if not config.COSYVOICE_PYTHON_BIN or not __import__("pathlib").Path(
                config.COSYVOICE_PYTHON_BIN).exists():
            raise BackendCrash(f"cosyvoice interpreter missing: {config.COSYVOICE_PYTHON_BIN}")

    def env_vars(self) -> dict[str, str]:
        env = {
            "CV_REPO_DIR": str(config.COSYVOICE_REPO_DIR),
            "CV_MODEL_DIR": str(config.COSYVOICE_WEIGHTS_DIR),
            "CV_WORKER_IDLE_TIMEOUT": str(max(self.profile.idle_unload_s + 300, 1800)),
        }
        if not self.profile.gpu:
            # CPU 兜底通道：torch 必须看不到 CUDA（worker 在 import torch 前生效），
            # 线程数显式限死，避免和 GPU runtime 的宿主侧负载抢全部核心。
            env.update({
                "CV_LANE": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": str(config.COSYVOICE_CPU_THREADS),
                "MKL_NUM_THREADS": str(config.COSYVOICE_CPU_THREADS),
            })
        return env
