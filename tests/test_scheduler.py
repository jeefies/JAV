import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from jav.runtime import supervisor as sup_mod
from jav.scheduler import Scheduler

from conftest import FakeBackend


def mk_sched(store, max_same=2, max_other_wait=3600, mem=60000, vram=60000):
    profiles = sup_mod.config.load_profiles()
    for p in profiles.values():
        p.backend = "fake"
    sup = sup_mod.Supervisor(store, profiles, mem_probe=lambda: mem,
                             vram_probe=lambda: {}, vram_free_probe=lambda: vram)
    return Scheduler(store, sup, max_same=max_same, max_other_wait=max_other_wait)


def mk(store, profile, age_s=0, priority=0):
    payload = {"provider": "zit", "workflow": "t2i", "prompt": "x",
               "negative_prompt": "", "generation": {"width": 64, "height": 64,
                                                     "steps": 4, "guidance": 0.0,
                                                     "strength": 0.8, "seed": -1},
               "assets": {}}
    j = store.create_job(provider="zit", workflow="t2i", runtime_profile=profile,
                         payload=payload, assets=[])
    if age_s:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
        store.update_job(j["id"], created_at=ts)
    return store.get_job(j["id"])


@pytest.fixture
def store(fresh_store):
    return fresh_store


def test_choose_affinity_then_switch(store):
    s = mk_sched(store)
    a, b = mk(store, "zit"), mk(store, "zit")
    d = mk(store, "ltx25")
    # no active → head of queue
    assert s.choose()["id"] == a["id"]
    store.update_job(a["id"], status="running")  # simulate a in flight
    s.sup.active_profile = "zit"
    s.streak = 1
    assert s.choose()["id"] == b["id"]          # affinity while streak < max
    store.update_job(b["id"], status="running")
    s.streak = 2
    assert s.choose()["id"] == d["id"]          # anti-starvation switch


def test_choose_other_wait_overrides_affinity(store):
    s = mk_sched(store, max_same=5, max_other_wait=60)
    mk(store, "zit"),
    other = mk(store, "ltx25", age_s=120)
    s.sup.active_profile = "zit"
    s.streak = 0
    assert s.choose()["id"] == other["id"]


def test_admission_denied_requeues_with_backoff(store):
    s = mk_sched(store, mem=100)  # below any budget
    j = mk(store, "zit")
    s.sup.active_profile = None
    claimed = store.claim_next("zit")
    asyncio.run(s.run_job(claimed))
    assert store.get_job(j["id"])["status"] == "queued"
    snap = s.snapshot()
    assert "zit" in snap["admission_backoff"]
    evs = store.recent_events(5)
    assert any("admission_denied" in (e["reason"] or "") for e in evs)


def test_vram_admission_denied(store):
    """ghost CUDA context holding VRAM must keep jobs queued, not OOM-kill."""
    s = mk_sched(store, vram=3000)  # far below zit vram budget
    j = mk(store, "zit")
    claimed = store.claim_next("zit")
    asyncio.run(s.run_job(claimed))
    got = store.get_job(j["id"])
    assert got["status"] == "queued"
    evs = store.recent_events(5)
    assert any("admission_denied" in (e["reason"] or "") and "VRAM" in (e["reason"] or "")
               for e in evs)
    assert s.snapshot()["admission_backoff"].get("zit", 0) > 0


def test_oom_shuts_down_runtime_and_retries_once(store):
    class OomBackend(FakeBackend):
        async def submit(self, job_id, task):
            return {"status": "failed", "paths": [], "error": "vram OOM",
                    "error_type": "oom_error"}

    profiles = sup_mod.config.load_profiles()
    for p in profiles.values():
        p.backend = "oom"
    sup_mod.register_backend("oom", OomBackend)
    sup = sup_mod.Supervisor(store, profiles, mem_probe=lambda: 60000,
                             vram_probe=lambda: {}, vram_free_probe=lambda: 60000)
    s = Scheduler(store, sup)
    j = mk(store, "zit")
    asyncio.run(s.process_once())
    first = store.get_job(j["id"])
    assert first["status"] == "queued" and first["retry_count"] == 1
    assert sup.backend is None          # runtime torn down after OOM
    asyncio.run(s.process_once())
    second = store.get_job(j["id"])
    assert second["status"] == "failed"


def test_full_run_success_lifecycle(store):
    s = mk_sched(store)
    j = mk(store, "zit")
    asyncio.run(s.process_once())
    got = store.get_job(j["id"])
    assert got["status"] == "completed", got["error"]
    assert store.get_outputs(j["id"])


def test_crash_retries_once_then_fails(store):
    FakeBackend.default_mode = "crash"
    s = mk_sched(store)
    j = mk(store, "zit")
    asyncio.run(s.process_once())
    first = store.get_job(j["id"])
    assert first["status"] == "queued" and first["retry_count"] == 1
    asyncio.run(s.process_once())
    second = store.get_job(j["id"])
    assert second["status"] == "failed"
    FakeBackend.default_mode = "success"


def test_generation_error_no_retry(store):
    FakeBackend.default_mode = "fail"
    s = mk_sched(store)
    j = mk(store, "zit")
    asyncio.run(s.process_once())
    got = store.get_job(j["id"])
    assert got["status"] == "failed" and got["retry_count"] == 0
    FakeBackend.default_mode = "success"
