"""ZitBackend: manages the diffusers Z-Image-Turbo worker subprocess
(stdin JSON + HTTP callback), same validated protocol as legacy ZIT-service."""
from __future__ import annotations

from .. import config
from .base import BackendCrash
from .subprocess_worker import SubprocessWorkerBackend


class ZitBackend(SubprocessWorkerBackend):
    kind = "zit_subprocess"
    label = "zit"

    def preflight(self):
        if not config.ZIT_WEIGHTS_DIR.is_dir():
            raise BackendCrash(f"ZIT weights missing at {config.ZIT_WEIGHTS_DIR}")

    def env_vars(self) -> dict[str, str]:
        return {
            "ZIT_MODEL_DIR": str(config.ZIT_WEIGHTS_DIR),
            "ZIT_WORKER_IDLE_TIMEOUT": str(max(self.profile.idle_unload_s + 300, 1800)),
        }
