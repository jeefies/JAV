"""Regression tests for the 2026-10-01 whole-branch audit fixes.

Every test pins one reviewed finding: scheduler starvation/chokepoint/
containment, profile merge, timeout policy, capability gates, docs-vs-code
contract aliases, and auth details."""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "/mnt/data/AV/JAV")

import pytest

from jav import config
from jav.models import BatchSpec, ProviderError
from jav.providers import ltx25, mh3
from jav.scheduler import RETRYABLE_ERRORS


# ------------------------------------------------------- contract: graphs --
class TestGraphContract:
    @pytest.fixture(autouse=True)
    def real_workflows(self, monkeypatch):
        monkeypatch.setattr(config, "BASE_DIR", Path("/mnt/data/AV/JAV"))

    def _compile(self, provider, workflow, inputs, generation=None):
        p = provider.normalize(workflow, inputs, generation or {})
        t = provider.compile(p, {k: f"/tmp/{k}.bin" for k in p["assets"]},
                             "/tmp", "/base")
        return p, t

    def test_ltx25_image_alias_maps_first_image(self):
        p, _ = self._compile(ltx25, "i2v", {"prompt": "x", "image": "asset_1"})
        assert p["assets"]["first_image"] == "asset_1"

    def test_ltx25_flf2v_frame_aliases(self):
        p, _ = self._compile(ltx25, "flf2v", {"prompt": "x", "first_frame": "asset_a",
                                              "last_frame": "asset_b"})
        assert p["assets"] == {"first_image": "asset_a", "last_image": "asset_b"}
        p, _ = self._compile(ltx25, "flf2v", {"prompt": "x", "image": "asset_a",
                                              "last_image": "asset_b"})
        assert p["assets"] == {"first_image": "asset_a", "last_image": "asset_b"}

    def test_mh3_i2v_image_alias(self):
        p, _ = self._compile(mh3, "i2v", {"prompt": "x", "image": "asset_9"})
        assert p["assets"]["first_frame"] == "asset_9"

    def test_audio_frame_rate_follows_requested_fps(self):
        _, t = self._compile(ltx25, "t2v", {"prompt": "x"}, {"fps": 30})
        assert t["graph"]["9"]["inputs"]["frame_rate"] == 30
        _, t = self._compile(ltx25, "flf2v", {"prompt": "x", "image": "a",
                                              "last_image": "b"}, {"fps": 12})
        assert t["graph"]["266"]["inputs"]["frame_rate"] == 12

    def test_control_prompt_optional_but_union_motion_required(self):
        ltx25.normalize("inpaint", {"source_video": "s", "mask_image": "m"}, {})
        ltx25.normalize("ic_lora", {"source_video": "s", "lora": "cinemagraph",
                                    "mode": "v2v"}, {})
        with pytest.raises(ProviderError):
            ltx25.normalize("union_control", {"control_video": "c"}, {})
        with pytest.raises(ProviderError):
            ltx25.normalize("motion_control", {"source_video": "s"}, {})

    def test_multiframe_turbo_uses_ref2v_lora(self):
        _, t = self._compile(mh3, "multiframe",
                             {"prompt": "x", "reference_images": ["r1"],
                              "keyframes": [{"image": "r1", "time": 0.0}],
                              "turbo": True})
        assert t["graph"]["900"]["inputs"]["lora_name"].startswith(
            "minimax_h3_ref2v_turbo_4step")

    def test_mh3_default_steps_is_base_quality(self):
        # 2026-10-09 regression: base int8 must never default to turbo-step counts
        p = mh3.normalize("t2v", {"prompt": "x"}, {})
        assert p["generation"]["steps"] == 20
        p = mh3.normalize("ref2v", {"prompt": "x", "reference_images": ["a"]}, {})
        assert p["generation"]["steps"] == 20
        p = mh3.normalize("ref2v", {"prompt": "x", "reference_images": ["a"], "turbo": True}, {})
        assert p["generation"]["steps"] == 4          # turbo forces official schedule
        p = mh3.normalize("fl2v", {"prompt": "x", "first_frame": "a", "last_frame": "b",
                                   "turbo": True}, {})
        assert p["generation"]["steps"] == 8
        p = mh3.normalize("t2v", {"prompt": "x"}, {"steps": 30})
        assert p["generation"]["steps"] == 30          # explicit wins

    def test_mh3_steps_validation(self):
        with pytest.raises(ProviderError):
            mh3.normalize("t2v", {"prompt": "x"}, {"steps": 0})
        with pytest.raises(ProviderError):
            mh3.normalize("t2v", {"prompt": "x"}, {"steps": -3})


# ---------------------------------------------------- scheduler semantics --
class FakeSup:
    active_profile = "zit"
    profiles = {}

    async def ensure(self, name):
        raise AssertionError("not used by choose()")


class TestSchedulerCore:
    def test_starvation_switch_uses_oldest_other(self, fresh_store):
        from jav.scheduler import EventBus, Scheduler
        sched = Scheduler(fresh_store, FakeSup(), EventBus())
        j_zit = fresh_store.create_job(provider="zit", workflow="t2i",
                                       runtime_profile="zit", payload={}, assets=[])
        old = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
        j_starved = fresh_store.create_job(provider="ltx25", workflow="t2v",
                                           runtime_profile="ltx25", payload={},
                                           assets=[])
        fresh_store.update_job(j_starved["id"], created_at=old)
        fresh_store.create_job(provider="ltx25", workflow="t2v",
                               runtime_profile="ltx25", payload={}, assets=[])
        nxt = sched.choose()
        # a >10min starved job must trigger the switch even when a young
        # same-other-profile job exists (min->max regression)
        assert nxt["runtime_profile"] == "ltx25"

    def test_finish_chokepoint_cancel_beats_completed(self, fresh_store):
        from jav.scheduler import EventBus, Scheduler
        sched = Scheduler(fresh_store, FakeSup(), EventBus())
        j = fresh_store.create_job(provider="zit", workflow="t2i",
                                   runtime_profile="zit", payload={}, assets=[])
        fresh_store.set_status(j["id"], "running")
        fresh_store.request_cancel(j["id"])
        sched._finish(j["id"], "completed")
        assert fresh_store.get_job(j["id"])["status"] == "cancelled"

    def test_retryable_set_contract(self):
        assert "runtime_start_failed" not in RETRYABLE_ERRORS
        assert {"timeout", "oom_error", "runtime_crash"} <= RETRYABLE_ERRORS


# --------------------------------------------------------------- config ----
class TestProfileOverrides:
    def test_yaml_override_merges_fields(self, tmp_path, monkeypatch):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "profiles.yaml").write_text(
            "- name: zit\n  ram_budget_mb: 24576\n")
        monkeypatch.setattr(config, "BASE_DIR", tmp_path)
        p = config.load_profiles()["zit"]
        assert p.ram_budget_mb == 24576
        # the trap: pre-fix these came back blanked by Profile defaults
        assert p.script == str(tmp_path / "jav" / "runtime" / "zit_worker.py")
        assert p.python_bin  # not blanked
        assert p.start_timeout_s == 600
        assert p.job_timeout_s == 900
        assert p.idle_unload_s == 300


# --------------------------------------------------------------- API -------
class TestAPIFixes:
    def _wait_status(self, client, jid, wanted, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = client.get(f"/v1/jobs/{jid}").json()["status"]
            if st in wanted:
                return st
            time.sleep(0.05)
        raise AssertionError(f"job {jid} never reached {wanted}")

    def test_unload_guard_refuses_running_job(self, app_client):
        from conftest import FakeBackend
        FakeBackend.default_mode = "wait"
        j = app_client.post("/v1/jobs", json={"provider": "zit", "workflow": "t2i",
                                              "inputs": {"prompt": "x"}}).json()
        assert self._wait_status(app_client, j["id"], {"running"}) == "running"
        r = app_client.post("/v1/runtime/unload")
        assert r.status_code == 409
        FakeBackend.release = True
        assert self._wait_status(app_client, j["id"], {"completed"}) == "completed"
        assert app_client.post("/v1/runtime/unload").status_code == 200

    def test_keepalive_ttl_clamped_both_ways(self, app_client):
        j = app_client.post("/v1/jobs", json={"provider": "zit", "workflow": "t2i",
                                              "inputs": {"prompt": "x"}}).json()
        assert self._wait_status(app_client, j["id"], {"completed"}) == "completed"
        r = app_client.post("/v1/runtime/keepalive?ttl_s=-1")
        assert r.json()["idle_unload_in_s"] == 1
        assert app_client.post("/v1/runtime/keepalive?ttl_s=99999") \
            .json()["idle_unload_in_s"] == 7200

    def test_queued_cancel_still_200_after_terminal_fix(self, fresh_store):
        # unit-level pin of _cancel_one's (status, already_terminal) contract:
        # a fresh queued cancel must NOT be classified as a terminal race
        import asyncio
        from types import SimpleNamespace
        from jav.api.jobs import _cancel_one
        from jav.scheduler import EventBus
        ctx = SimpleNamespace(store=fresh_store, bus=EventBus(),
                              sup=SimpleNamespace(backend=None))
        j = fresh_store.create_job(provider="zit", workflow="t2i",
                                   runtime_profile="zit", payload={}, assets=[])
        status, raced = asyncio.run(_cancel_one(ctx, j["id"]))
        assert (status, raced) == ("cancelled", False)
        done = fresh_store.create_job(provider="zit", workflow="t2i",
                                      runtime_profile="zit", payload={}, assets=[],
                                      status="completed")
        status, raced = asyncio.run(_cancel_one(ctx, done["id"]))
        assert (status, raced) == ("completed", True)

    def test_bearer_constant_time_path_enforced(self, fresh_store, monkeypatch):
        # middleware is installed at create_app time from config.API_TOKEN,
        # so build a dedicated token-enabled app
        from fastapi.testclient import TestClient
        from jav import server
        from jav.runtime import supervisor as sup_mod
        from jav.scheduler import EventBus, Scheduler
        from conftest import FakeBackend
        monkeypatch.setattr(config, "API_TOKEN", "sekret")
        sup_mod.register_backend("fake", FakeBackend)
        FakeBackend.instances.clear()
        FakeBackend.default_mode = "success"
        profiles = config.load_profiles()
        for name in profiles:
            profiles[name].backend = "fake"
        sup = sup_mod.Supervisor(fresh_store, profiles, mem_probe=lambda: 60000,
                                 vram_probe=lambda: {}, vram_free_probe=lambda: 60000)
        bus = EventBus()
        app = server.create_app(profiles=profiles, store=fresh_store, sup=sup,
                                bus=bus, sched=Scheduler(fresh_store, sup, bus))
        body = {"provider": "zit", "workflow": "t2i", "inputs": {"prompt": "x"}}
        with TestClient(app) as client:
            assert client.post("/v1/jobs", json=body).status_code == 401
            assert client.post("/v1/jobs", json=body,
                               headers={"Authorization": "Bearer nope"}).status_code == 401
            assert client.post("/v1/jobs", json=body,
                               headers={"Authorization": "Bearer sekret"}).status_code == 201
            assert client.get("/v1/jobs").status_code == 200  # GET stays open

    def test_upload_dedup_uses_streamed_digest(self, app_client):
        body = b"jpeg-bytes" * 1000
        import hashlib
        want = "asset_" + hashlib.sha256(body).hexdigest()[:16]
        a1 = app_client.post("/v1/assets?kind=image", content=body).json()
        a2 = app_client.post("/v1/assets?kind=image", content=body).json()
        assert a1["id"] == a2["id"] == want


# --------------------------------------------------------------- batch ----
class TestBatchContract:
    def test_client_ref_precedence(self):
        spec = BatchSpec(shared={"provider": "zit", "workflow": "t2i",
                                 "client_ref": "from-shared"},
                        jobs=[{"inputs": {"prompt": "a"}},
                              {"inputs": {"prompt": "b"}, "client_ref": "job-tag"}],
                        client_ref="batch-tag")
        m0 = spec.merged(spec.jobs[0])
        m1 = spec.merged(spec.jobs[1])
        assert m0["client_ref"] == "from-shared"   # shared > batch-level
        assert m1["client_ref"] == "job-tag"       # job wins


# ------------------------------------------------------------- dead code ---
class TestDeadChainRemoved:
    def test_output_kinds_gone(self, monkeypatch):
        monkeypatch.setattr(config, "BASE_DIR", Path("/mnt/data/AV/JAV"))
        from jav.providers import comfy_provider
        assert not hasattr(comfy_provider, "output_kinds")
        p = ltx25.normalize("t2v", {"prompt": "x"}, {})
        t = ltx25.compile(p, {}, "/tmp", "/base")
        assert "output_kinds" not in t
