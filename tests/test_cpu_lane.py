"""CPU 兜底通道（cosyvoice-cpu lane）路由与隔离测试。"""
import asyncio
import time

import pytest
import yaml

from jav import config, voices
from jav.providers import cosyvoice as cv
from jav.runtime import supervisor as sup_mod
from jav.scheduler import Scheduler

from conftest import FakeBackend


def _sched(store, mem=60000):
    sup_mod.register_backend("fake", FakeBackend)
    profiles = sup_mod.config.load_profiles()
    for p in profiles.values():
        p.backend = "fake"
    sup = sup_mod.Supervisor(store, profiles, mem_probe=lambda: mem,
                             vram_probe=lambda: {}, vram_free_probe=lambda: 60000)
    return Scheduler(store, sup)


@pytest.fixture
def registry_b():
    f = config.BASE_DIR / "config" / "voices.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(
        [{"id": "actor_b", "name": "角色乙", "prompt_text": "逐字稿。",
          "path": str(config.DATA_DIR / "refs/actor_b.wav")}], allow_unicode=True))
    voices.load_voices(force=True)
    yield
    f.unlink(missing_ok=True)
    voices.load_voices(force=True)


@pytest.fixture
def store(fresh_store):
    return fresh_store


def _mk_cv(store):
    payload = cv.normalize("t2a", {"text": "测试台词。", "voice_id": "actor_b"}, {})
    j = store.create_job(provider="cosyvoice", workflow="t2a",
                         runtime_profile="cosyvoice", payload=payload, assets=[])
    return store.get_job(j["id"])


def test_cpu_profile_declared(store):
    p = sup_mod.config.load_profiles()["cosyvoice-cpu"]
    assert p.gpu is False and p.backend == "cosyvoice_subprocess"
    assert p.vram_budget_mb == 0


def test_admission_skips_vram_for_cpu_profile(store):
    s = _sched(store)
    # VRAM 探测为 0（GPU 满）：GPU profile 拒绝，CPU profile 仍放行
    s.sup.vram_free_probe = lambda: 0
    with pytest.raises(sup_mod.AdmissionDenied):
        s.sup.check_admission("cosyvoice")
    s.sup.check_admission("cosyvoice-cpu")  # no raise


def test_routes_to_cpu_lane_when_gpu_busy(store, registry_b):
    s = _sched(store)
    FakeBackend.instances.clear()
    s.sup.active_profile = "mh3.fl2va"
    s.sup.backend = FakeBackend(s.sup.profiles["mh3.fl2va"])
    j = _mk_cv(store)
    assert asyncio.run(s.process_cpu_once()) is True
    done = store.get_job(j["id"])
    assert done["status"] == "completed"
    # 任务记录的 runtime_profile 不变（API 契约稳定）
    assert done["runtime_profile"] == "cosyvoice"
    spawned = {b.profile.name for b in FakeBackend.instances}
    assert "cosyvoice-cpu" in spawned
    assert "cosyvoice" not in spawned
    # GPU 通道分毫未动
    assert s.sup.active_profile == "mh3.fl2va"
    assert s.sup.backend.alive()


def test_idle_gpu_keeps_fast_lane(store, registry_b):
    s = _sched(store)
    j = _mk_cv(store)
    # GPU 空闲：兜底通道必须保持安静，让主循环走 GPU 快通道
    assert s._cpu_target() is None
    assert store.get_job(j["id"])["status"] == "queued"


def test_gpu_running_same_family_no_double_worker(store, registry_b):
    s = _sched(store)
    s.sup.active_profile = "cosyvoice"
    s.sup.backend = FakeBackend(s.sup.profiles["cosyvoice"])
    _mk_cv(store)
    assert s._cpu_target() is None


def test_gpu_backoff_for_source_enables_route(store, registry_b):
    s = _sched(store)
    j = _mk_cv(store)
    # GPU 空闲但 cosyvoice 正准入退避 → 兜底通道接管
    s._backoff["cosyvoice"] = time.monotonic() + 30
    t = s._cpu_target()
    assert t is not None and t["id"] == j["id"]


def test_cpu_admission_denied_requeues(store, registry_b):
    s = _sched(store, mem=1000)  # RAM 不足（< 10240+floor）
    FakeBackend.instances.clear()
    s.sup.active_profile = "mh3.fl2va"
    s.sup.backend = FakeBackend(s.sup.profiles["mh3.fl2va"])
    j = _mk_cv(store)
    asyncio.run(s.process_cpu_once())
    assert store.get_job(j["id"])["status"] == "queued"
    assert "cosyvoice-cpu" in s._backoff  # 退避已登记，循环不会热转


def test_runtime_state_files_are_lane_scoped(store):
    store.save_runtime_state(1111, "cosyvoice-cpu", 42, lane="cpu")
    store.save_runtime_state(2222, "zit", 43)
    assert store.load_runtime_state(lane="cpu")["pid"] == 1111
    assert store.load_runtime_state()["pid"] == 2222
    store.clear_runtime_state(lane="cpu")
    assert store.load_runtime_state(lane="cpu") is None
    assert store.load_runtime_state() is not None  # gpu lane untouched


def test_callback_routing_by_lane_tag(store):
    s = _sched(store)
    gpu = FakeBackend(s.sup.profiles["cosyvoice"])
    cpu = FakeBackend(s.sup.profiles["cosyvoice-cpu"])
    s.sup.backend = gpu
    s.sup.cpu_backend = cpu
    assert s.sup._route({"lane": "cpu", "job_id": "x"}) is cpu
    assert s.sup._route({"lane": "gpu", "job_id": "x"}) is gpu
    assert s.sup._route({"job_id": "x"}) is gpu          # untagged → historical GPU
    s.sup.backend = None
    cpu._pending = {"job_1": None}
    assert s.sup._route({"job_id": "job_1"}) is cpu      # untagged late callback fallback
    s.sup.cpu_backend = None
    assert s.sup._route({"lane": "cpu", "job_id": "x"}) is None


def test_snapshot_exposes_cpu_lane(store):
    s = _sched(store)
    s.sup.cpu_active_profile = "cosyvoice-cpu"
    s.sup.cpu_state = "READY"
    snap = s.snapshot()
    assert snap["cpu_active_profile"] == "cosyvoice-cpu"
    assert snap["cpu_state"] == "READY"


def test_shutdown_both_lanes(store):
    s = _sched(store)
    gpu = FakeBackend(s.sup.profiles["cosyvoice"])
    cpu = FakeBackend(s.sup.profiles["cosyvoice-cpu"])
    s.sup.backend = gpu
    s.sup.cpu_backend = cpu
    s.sup.active_profile = "cosyvoice"
    s.sup.cpu_active_profile = "cosyvoice-cpu"
    asyncio.run(s.sup.shutdown("test", lane="both"))
    assert gpu.stopped_with == "test" and cpu.stopped_with == "test"
    assert s.sup.backend is None and s.sup.cpu_backend is None
