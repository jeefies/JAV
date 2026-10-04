import os

os.environ["JAV_BASE_DIR"] = "/tmp/kilo/jav-test"
os.environ["JAV_DATA_DIR"] = "/tmp/kilo/jav-test/data"
os.environ["JAV_LOG_DIR"] = "/tmp/kilo/jav-test/logs"

import shutil
import sys

sys.path.insert(0, "/mnt/data/AV/JAV")

import pytest

shutil.rmtree("/tmp/kilo/jav-test", ignore_errors=True)

from jav import config

config.DATA_DIR = __import__("pathlib").Path("/tmp/kilo/jav-test/data")
config.ASSETS_DIR = config.DATA_DIR / "assets"
config.OUTPUTS_DIR = config.DATA_DIR / "outputs"
config.HF_HOME_DIR = config.DATA_DIR / "hf_home"
config.LOG_DIR = __import__("pathlib").Path("/tmp/kilo/jav-test/logs")
config.DB_PATH = config.DATA_DIR / "jav.db"
config.ZIT_WEIGHTS_DIR = config.DATA_DIR / "fake_zit_weights"
config.ZIT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
(config.ZIT_WEIGHTS_DIR / "model_index.json").write_text("{}")

# Fake CosyVoice3 install so the t2a capability gate passes in tests
config.COSYVOICE_REPO_DIR = config.DATA_DIR / "fake_cv_repo"
(config.COSYVOICE_REPO_DIR / "cosyvoice").mkdir(parents=True, exist_ok=True)
(config.COSYVOICE_REPO_DIR / "third_party" / "Matcha-TTS").mkdir(parents=True, exist_ok=True)
config.COSYVOICE_WEIGHTS_DIR = config.DATA_DIR / "fake_cv_weights"
config.COSYVOICE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
for _f in ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt",
           "campplus.onnx", "speech_tokenizer_v3.onnx"):
    (config.COSYVOICE_WEIGHTS_DIR / _f).write_text("{}")
(config.COSYVOICE_WEIGHTS_DIR / "CosyVoice-BlankEN").mkdir(exist_ok=True)

config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class FakeBackend:
    """Scriptable backend: mode = success|fail|crash|wait"""
    kind = "fake"
    default_mode = "success"
    instances: list = []
    release = False

    def __init__(self, profile):
        self.profile = profile
        self.pid = 999999
        self.mode = FakeBackend.default_mode
        self.started = False
        self.submitted = []
        self.cancelled = []
        self.stopped_with = None
        FakeBackend.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self, reason="stop"):
        self.stopped_with = reason

    def alive(self):
        return True

    async def submit(self, job_id, task):
        self.submitted.append((job_id, task))
        if self.mode == "fail":
            return {"status": "failed", "paths": [], "error": "fake fail",
                    "error_type": "generation_error"}
        if self.mode == "oom":
            return {"status": "failed", "paths": [], "error": "fake vram OOM",
                    "error_type": "oom_error"}
        if self.mode == "crash":
            from jav.runtime.base import BackendCrash
            raise BackendCrash("fake crash")
        if self.mode == "wait":
            import time as _t
            deadline = _t.monotonic() + 300
            while not FakeBackend.release and _t.monotonic() < deadline:
                import asyncio
                await asyncio.sleep(0.05)
            if not FakeBackend.release:
                return {"status": "failed", "paths": [], "error": "wait deadline",
                        "error_type": "timeout"}
            # mirror ZitBackend._resolve: the flag reflects an ACTUAL cancel
            # call, never a blanket True (hard-coded True made the
            # unload-guard test unresolvable: 2026-10-04)
            return {"status": "success", "paths": [], "error": None,
                    "cancel_requested": job_id in self.cancelled}
        import hashlib
        task = self.submitted[-1][1]
        is_tts = task.get("mode") == "t2a"
        ext = ".wav" if is_tts else ".png"
        label = "wav-bytes" if is_tts else "png-bytes"
        data = f"{label}-{job_id}".encode()
        sha = hashlib.sha256(data).hexdigest()
        d = config.OUTPUTS_DIR / sha[:2]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{job_id}{ext}"
        p.write_bytes(data)
        res = {"status": "success", "paths": [str(p)], "error": None}
        if is_tts:
            res["meta"] = {"duration_s": 2.5, "sample_rate": 48000,
                           "mode": "zero_shot", "voice_id": task.get("voice_id")}
        return res

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    def deliver(self, payload):
        return False

    def pipeline_status(self, payload):
        return False


@pytest.fixture
def fresh_store():
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    for sub in ("assets", "outputs", "pending"):
        shutil.rmtree(config.DATA_DIR / sub, ignore_errors=True)
    flags = config.DATA_DIR / "capability_flags.json"
    if flags.exists():
        flags.unlink()
    from jav.store import Store
    s = Store()
    yield s
    s.close()


@pytest.fixture
def app_client(fresh_store):
    """Full app with auto-running scheduler; fakes replace all backends."""
    from jav import server
    from jav.runtime import supervisor as sup_mod
    from jav.scheduler import EventBus, Scheduler
    from jav.store import Store

    sup_mod.register_backend("fake", FakeBackend)
    FakeBackend.instances.clear()
    FakeBackend.default_mode = "success"
    FakeBackend.release = False

    profiles = sup_mod.config.load_profiles()
    for name in ("zit", "ltx25", "mh3.fl2va", "mh3.ref2va", "cosyvoice"):
        profiles[name].backend = "fake"

    store = Store()
    sup = sup_mod.Supervisor(store, profiles,
                             mem_probe=lambda: 60000, vram_probe=lambda: {},
                             vram_free_probe=lambda: 60000)
    bus = EventBus()
    sched = Scheduler(store, sup, bus)
    app = server.create_app(profiles=profiles, store=store, sup=sup, bus=bus, sched=sched)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client
    FakeBackend.release = True
