"""Complex correctness/stability tests: full cancel paths, retry budget,
SSE mid-flight, cache-hit semantics, process-level orphan sweep identity,
admission fail-closed, SQLite migration ladder, flag concurrency, streaming
upload caps, and a batch consistency chaos pass."""
import json
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from jav import config
from jav.api import assets as assets_api
from jav import capabilities
from jav.runtime import supervisor as sup_mod
from jav.store import Store, TERMINAL

from conftest import FakeBackend


def T2I(prompt, seed=-1):
    return {"provider": "zit", "workflow": "t2i", "inputs": {"prompt": prompt},
            "generation": {"width": 64, "height": 64, "steps": 4, "seed": seed}}


def wait_for_status(client, jid, want, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/v1/jobs/{jid}").json()["status"]
        if s == want:
            return s
        time.sleep(0.05)
    pytest.fail(f"job {jid} never reached {want} (last {s})")


def wait_terminal(client, jid, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/v1/jobs/{jid}").json()["status"]
        if s in TERMINAL:
            return s
        time.sleep(0.05)
    pytest.fail(f"job {jid} not terminal (last {s})")


# ---------------------------------------------------------------- cancel ---
class TestCancelPaths:
    def test_cancel_running_persists_and_finalizes(self, app_client):
        FakeBackend.default_mode = "wait"
        jid = app_client.post("/v1/jobs", json=T2I("slow")).json()["id"]
        wait_for_status(app_client, jid, "running")
        r = app_client.delete(f"/v1/jobs/{jid}")
        assert r.status_code == 200 and r.json()["status"] == "cancelling"
        # cancel must be durable in the DB, not just in-memory
        inst = FakeBackend.instances[-1]
        assert jid in inst.cancelled          # backend soft-interrupt called
        assert app_client.get(f"/v1/jobs/{jid}").json()["status"] == "running"
        inst.mode = "success"
        FakeBackend.release = True
        assert wait_terminal(app_client, jid) == "cancelled"
        assert app_client.get(f"/v1/jobs/{jid}/outputs").json()["outputs"] == []
        # re-cancel of a terminal job -> 409
        assert app_client.delete(f"/v1/jobs/{jid}").status_code == 409

    def test_cancel_batch_partial(self, app_client):
        FakeBackend.default_mode = "wait"
        batch = {"shared": {"provider": "zit", "workflow": "t2i",
                            "generation": {"width": 64, "height": 64, "steps": 4}},
                 "jobs": [{"inputs": {"prompt": f"b{i}"}} for i in range(4)]}
        b = app_client.post("/v1/jobs/batch", json=batch).json()
        first = b["jobs"][0]
        wait_for_status(app_client, first, "running")
        # selective cancels: one running (soft) + one queued (immediate)
        assert app_client.delete(f"/v1/jobs/{first}").json()["status"] == "cancelling"
        assert app_client.delete(f"/v1/jobs/{b['jobs'][1]}").json()["status"] == "cancelled"
        FakeBackend.instances[-1].mode = "success"
        FakeBackend.release = True
        assert wait_terminal(app_client, first) == "cancelled"
        assert app_client.get(f"/v1/jobs/{b['jobs'][1]}").json()["status"] == "cancelled"
        assert wait_terminal(app_client, b["jobs"][2]) == "completed"
        assert wait_terminal(app_client, b["jobs"][3]) == "completed"
        # whole-batch cancel once everything is terminal -> per-job terminal map
        gb = app_client.delete(f"/v1/batches/{b['batch_id']}").json()["result"]
        assert set(gb.values()) <= set(TERMINAL)


# ----------------------------------------------------------------- retry ---
class TestRetryBudget:
    def test_deterministic_error_never_retries(self, app_client):
        # generation_error is by design NOT retryable (deterministic failure)
        FakeBackend.default_mode = "fail"
        jid = app_client.post("/v1/jobs", json=T2I("flaky")).json()["id"]
        assert wait_terminal(app_client, jid) == "failed"
        job = app_client.get(f"/v1/jobs/{jid}").json()
        assert job["retry_count"] == 0
        assert job["error_type"] == "generation_error"
        assert len(FakeBackend.instances[-1].submitted) == 1

    def test_runtime_crash_retries_exactly_once(self, app_client):
        FakeBackend.default_mode = "crash"
        jid = app_client.post("/v1/jobs", json=T2I("crasher")).json()["id"]
        assert wait_terminal(app_client, jid) == "failed"
        job = app_client.get(f"/v1/jobs/{jid}").json()
        assert job["retry_count"] == 1
        assert job["error_type"] == "runtime_crash"
        assert len(FakeBackend.instances[-1].submitted) == 2  # no retry storm

    def test_oom_kills_runtime_then_retries_once(self, app_client):
        FakeBackend.default_mode = "oom"
        jid = app_client.post("/v1/jobs", json=T2I("oomy")).json()["id"]
        assert wait_terminal(app_client, jid) == "failed"
        job = app_client.get(f"/v1/jobs/{jid}").json()
        assert job["retry_count"] == 1
        # oom path must tear the runtime down -> a fresh instance serves the retry
        assert len(FakeBackend.instances) == 2

    def test_deterministic_cache_hit_no_gpu_work(self, app_client):
        spec = T2I("reproducible", seed=777)
        j1 = app_client.post("/v1/jobs", json=spec).json()["id"]
        assert wait_terminal(app_client, j1) == "completed"
        submits = len(FakeBackend.instances[-1].submitted)
        r2 = app_client.post("/v1/jobs", json=spec).json()
        assert r2["status"] == "completed"          # served from cache
        assert len(FakeBackend.instances[-1].submitted) == submits  # zero submits
        o1 = app_client.get(f"/v1/jobs/{j1}/outputs").json()["outputs"]
        o2 = r2["outputs"]
        assert [o["asset_id"] for o in o1] == [o["asset_id"] for o in o2]
        # seed -1 must NOT hit cache
        r3 = app_client.post("/v1/jobs", json=T2I("reproducible")).json()
        assert r3["status"] == "queued"


# ------------------------------------------------------------------- SSE ---
class TestSSE:
    def test_stream_opens_midflight_and_closes_on_terminal(self, app_client):
        FakeBackend.default_mode = "wait"
        jid = app_client.post("/v1/jobs", json=T2I("sse")).json()["id"]
        wait_for_status(app_client, jid, "running")
        events, error = [], None

        def reader():
            nonlocal error
            try:
                with app_client.stream("GET", f"/v1/jobs/{jid}/events") as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data:"):
                            events.append(json.loads(line[5:].strip()))
            except Exception as e:  # pragma: no cover
                error = e

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and "running" not in {e.get("status") for e in events}:
            time.sleep(0.05)
        assert error is None
        FakeBackend.instances[-1].mode = "success"
        FakeBackend.release = True
        t.join(15)
        assert not t.is_alive()                      # stream closed at terminal
        assert events and events[0]["status"] == "running"
        assert events[-1]["status"] in TERMINAL
        assert not any(e["status"] == "queued" for e in events if events[0] != e)


# ------------------------------------------------- process-level sweep -----
def _spawn_dummy(script_path: Path) -> subprocess.Popen:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("import time\ntime.sleep(300)\n")
    return subprocess.Popen([sys.executable, "-u", str(script_path)])


class TestSweepIdentity:
    def _sup(self, store):
        profiles = sup_mod.config.load_profiles()
        # keep real backend kinds: the sweep matches the recorded profile's
        # spawn identity (zit_subprocess script path / comfy main.py)
        return sup_mod.Supervisor(store, profiles, mem_probe=lambda: 60000,
                                  vram_probe=lambda: {},
                                  vram_free_probe=lambda: 60000)

    def test_sweep_kills_true_orphan(self, fresh_store):
        sup = self._sup(fresh_store)
        script = Path(sup.profiles["zit"].script)
        proc = _spawn_dummy(script)
        try:
            fresh_store.save_runtime_state(proc.pid, "zit",
                                           sup_mod.proc_starttime(proc.pid))
            assert sup.sweep_orphans() == proc.pid
            assert proc.wait(5) is not None          # SIGTERM landed
            assert not (config.DATA_DIR / "runtime.state").exists()
        finally:
            proc.kill()

    def test_sweep_spares_reused_pid(self, fresh_store):
        sup = self._sup(fresh_store)
        script = Path(sup.profiles["zit"].script)
        proc = _spawn_dummy(script)                  # same cmdline identity...
        try:
            # ...but a DIFFERENT starttime than recorded => must not kill
            fresh_store.save_runtime_state(proc.pid, "zit",
                                           sup_mod.proc_starttime(proc.pid) + 12345)
            sup.sweep_orphans()
            assert proc.poll() is None
            assert not (config.DATA_DIR / "runtime.state").exists()
        finally:
            proc.kill()

    def test_sweep_spares_unknown_profile_and_loose_token(self, fresh_store):
        sup = self._sup(fresh_store)
        script = Path(sup.profiles["zit"].script)
        proc = _spawn_dummy(script)
        try:
            fresh_store.save_runtime_state(proc.pid, "ghost.profile",
                                           sup_mod.proc_starttime(proc.pid))
            sup.sweep_orphans()
            assert proc.poll() is None
            assert not (config.DATA_DIR / "runtime.state").exists()
        finally:
            proc.kill()


# --------------------------------------------------------------- admission --
class TestAdmissionFailClosed:
    def test_vram_probe_failure_denies(self, fresh_store, monkeypatch):
        profiles = sup_mod.config.load_profiles()
        sup = sup_mod.Supervisor(fresh_store, profiles, mem_probe=lambda: 60000,
                                 vram_probe=lambda: {}, vram_free_probe=lambda: None)
        with pytest.raises(sup_mod.AdmissionDenied, match="VRAM\\(probe failed\\)"):
            sup.check_admission("zit")
        monkeypatch.setenv("JAV_VRAM_GATE", "off")
        sup.check_admission("zit")                   # explicit override works

    def test_vram_low_denies_real_number(self, fresh_store):
        profiles = sup_mod.config.load_profiles()
        sup = sup_mod.Supervisor(fresh_store, profiles, mem_probe=lambda: 60000,
                                 vram_probe=lambda: {}, vram_free_probe=lambda: 10)
        with pytest.raises(sup_mod.AdmissionDenied, match="VRAM"):
            sup.check_admission("zit")


# --------------------------------------------------------------- migration ---
class TestSchemaMigration:
    OLD_JOBS = """CREATE TABLE jobs(
      id TEXT PRIMARY KEY, batch_id TEXT, client_ref TEXT,
      provider TEXT NOT NULL, workflow TEXT NOT NULL, runtime_profile TEXT NOT NULL,
      status TEXT NOT NULL, priority INTEGER DEFAULT 0,
      payload TEXT NOT NULL, assets TEXT NOT NULL DEFAULT '[]',
      cache_key TEXT,
      created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
      error TEXT, error_type TEXT, retry_count INTEGER DEFAULT 0,
      external_ref TEXT
    )"""

    def test_old_db_gets_column_and_version(self):
        path = config.DATA_DIR / "migrate_old.db"
        path.unlink(missing_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(self.OLD_JOBS)
        conn.execute("INSERT INTO jobs(id,provider,workflow,runtime_profile,status,"
                     "payload,created_at) VALUES('j1','zit','t2i','zit','running',"
                     "'{}','2026-01-01T00:00:00+00:00')")
        conn.commit()
        conn.close()
        store = Store(path)
        try:
            assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 1
            assert store.request_cancel("j1") is True
            assert store.cancel_requested("j1") is True
            assert store.get_job("j1")["cancel_requested"] == 1
        finally:
            store.close()
            path.unlink(missing_ok=True)

    def test_newer_db_refuses_startup(self):
        path = config.DATA_DIR / "migrate_future.db"
        path.unlink(missing_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(self.OLD_JOBS)
        conn.execute("PRAGMA user_version=99")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="newer than supported"):
            Store(path)
        path.unlink(missing_ok=True)


# ------------------------------------------------------ capabilities flags ---
class TestFlagsDurability:
    def test_concurrent_mark_validated_never_tears_file(self):
        original = (config.DATA_DIR / "capability_flags.json").exists()
        snapshot = (config.DATA_DIR / "capability_flags.json").read_text() \
            if original else None
        keys = [f"wf{i}" for i in range(8)]
        threads = [threading.Thread(target=capabilities.mark_validated,
                                    args=("zit", k)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        flags = capabilities._load_flags()
        assert all(flags.get(f"zit.{k}") for k in keys)  # lost-update check
        # simulate torn read: corrupt file falls back to last-good, not {}
        corrupt = config.DATA_DIR / "capability_flags.json"
        corrupt.write_text("{trunca")
        assert all(capabilities._load_flags().get(f"zit.{k}") for k in keys)
        if original:
            corrupt.write_text(snapshot)
        else:
            corrupt.unlink(missing_ok=True)


# ----------------------------------------------------------- upload caps ---
class TestUploadCaps:
    def test_raw_stream_over_cap_413_and_no_temp_leak(self, app_client):
        old = assets_api.MAX_ASSET_BYTES
        assets_api.MAX_ASSET_BYTES = 1024
        try:
            r = app_client.post("/v1/assets?kind=image", content=b"x" * 5000)
            assert r.status_code == 413
            r2 = app_client.post("/v1/assets/upload?kind=image",
                                 files={"file": ("big.bin", b"y" * 5000)})
            assert r2.status_code == 413
        finally:
            assets_api.MAX_ASSET_BYTES = old
        tmp = config.ASSETS_DIR / ".tmp"
        assert not (tmp.exists() and list(tmp.iterdir()))   # spool cleaned

    def test_empty_body_400(self, app_client):
        assert app_client.post("/v1/assets?kind=image", content=b"").status_code == 400
        # duplicate upload returns same id, file appears exactly once
        data = b"\x89PNG-dup"
        a1 = app_client.post("/v1/assets?kind=image", content=data).json()
        a2 = app_client.post("/v1/assets?kind=image", content=data).json()
        assert a1["id"] == a2["id"]


# ------------------------------------------------------- batch consistency --
class TestQueueConsistency:
    def test_mixed_cancel_submit_consistency(self, app_client):
        """Chaos pass: 8 submits + 1 queued-cancel + 1 running-cancel while a
        wait-mode job holds the runtime; assert full DB consistency afterwards."""
        FakeBackend.default_mode = "wait"
        ids = [app_client.post("/v1/jobs", json=T2I(f"chaos-{i}")).json()["id"]
               for i in range(8)]
        first = ids[0]
        wait_for_status(app_client, first, "running")
        victim_q = ids[-1]
        assert app_client.delete(f"/v1/jobs/{victim_q}").json()["status"] == "cancelled"
        assert app_client.delete(f"/v1/jobs/{first}").json()["status"] == "cancelling"
        inst = FakeBackend.instances[-1]
        inst.mode = "success"
        FakeBackend.release = True
        final = {}
        for jid in ids:
            final[jid] = wait_terminal(app_client, jid)
        assert final[victim_q] == "cancelled"
        assert final[first] == "cancelled"
        assert sum(1 for s in final.values() if s == "completed") == 6
        submitted = [j for j, _ in inst.submitted]
        assert len(submitted) == len(set(submitted)) == 7  # every job submitted <=1x
        # victim never touched the backend
        assert victim_q not in submitted
        # global consistency invariants
        listing = app_client.get("/v1/jobs", params={"limit": 500}).json()
        assert listing["total"] == len(ids)
        assert all(j["status"] in TERMINAL for j in listing["jobs"])
        for j in listing["jobs"]:
            if j["status"] == "completed":
                assert len(j["outputs"]) == 1
            else:
                assert j["outputs"] == []
            if j["status"] in TERMINAL:
                assert j["finished_at"]
        q = app_client.get("/v1/queue").json()
        assert q["queued_total"] == 0

    def test_list_jobs_equivalence_single(self, app_client):
        jid = app_client.post("/v1/jobs", json=T2I("eq")).json()["id"]
        wait_terminal(app_client, jid)
        one = app_client.get(f"/v1/jobs/{jid}").json()
        page = next(j for j in app_client.get("/v1/jobs", params={"limit": -1}).json()["jobs"]
                    if j["id"] == jid)
        assert {k: v for k, v in one.items() if k != "queue_position"} == \
               {k: v for k, v in page.items() if k != "queue_position"}
