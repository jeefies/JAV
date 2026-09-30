"""Regression tests for the /review fix batch (commit e95acac follow-up):
callback auth, output containment, persisted cancellation, pagination clamp,
asset delete guard, optional bearer auth."""
import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from jav import config
from jav.scheduler import Scheduler
from jav.runtime import supervisor as sup_mod


def test_internal_callback_requires_secret(app_client):
    r = app_client.post("/v1/internal/task_complete", json={"job_id": "x"})
    assert r.status_code == 403
    r2 = app_client.post("/v1/internal/pipeline_status",
                         json={"status": "unloaded"})
    assert r2.status_code == 403
    r3 = app_client.post("/v1/internal/task_complete", json={"job_id": "x"},
                         headers={"X-JAV-Callback": config.CALLBACK_SECRET})
    assert r3.status_code == 200  # no active backend -> handled False, not forged


def test_ingest_containment_rejects_unmanaged_paths(fresh_store):
    s = Scheduler(fresh_store, None)
    secret_file = config.DATA_DIR / "outside_secret.png"
    secret_file.write_bytes(b"top secret bytes")
    with pytest.raises(ValueError, match="outside managed dirs"):
        s._ingest_outputs("job_nonexistent", [str(secret_file)])
    assert secret_file.exists()  # nothing read/moved


def test_cancel_flag_finalizes_without_submit(fresh_store):
    profiles = sup_mod.config.load_profiles()
    for p in profiles.values():
        p.backend = "fake"
    from conftest import FakeBackend
    sup_mod.register_backend("fake", FakeBackend)
    FakeBackend.instances.clear()
    FakeBackend.default_mode = "success"
    sup = sup_mod.Supervisor(fresh_store, profiles, mem_probe=lambda: 60000,
                             vram_probe=lambda: {}, vram_free_probe=lambda: 60000)
    s = Scheduler(fresh_store, sup)
    payload = {"provider": "zit", "workflow": "t2i", "prompt": "x",
               "negative_prompt": "", "generation": {}, "assets": {}}
    j = fresh_store.create_job(provider="zit", workflow="t2i",
                               runtime_profile="zit", payload=payload, assets=[])
    claimed = fresh_store.claim_next("zit")
    assert claimed["id"] == j["id"]
    assert fresh_store.request_cancel(j["id"])          # starting_runtime
    asyncio.run(s.run_job(claimed))
    assert fresh_store.get_job(j["id"])["status"] == "cancelled"
    assert not FakeBackend.instances or not FakeBackend.instances[-1].submitted


def test_list_jobs_clamps_negative_limit(app_client, fresh_store):
    r = app_client.get("/v1/jobs", params={"limit": -1, "offset": -5})
    assert r.status_code == 200


def test_asset_delete_conflicts_while_referenced(app_client):
    from conftest import FakeBackend
    FakeBackend.default_mode = "wait"
    img = b"\x89PNG" + b"x" * 64
    a = app_client.post("/v1/assets?kind=image", content=img).json()
    busy = app_client.post("/v1/jobs", json={
        "provider": "zit", "workflow": "t2i", "inputs": {"prompt": "busy"},
        "generation": {"width": 64, "height": 64, "steps": 4}})
    assert busy.status_code == 201
    ref = app_client.post("/v1/jobs", json={
        "provider": "zit", "workflow": "i2i",
        "inputs": {"prompt": "p", "image": a["id"]},
        "generation": {"width": 64, "height": 64, "steps": 4}})
    assert ref.status_code == 201
    r2 = app_client.delete(f"/v1/assets/{a['id']}")
    assert r2.status_code == 409
    FakeBackend.release = True


def test_optional_bearer_auth(app_client, fresh_store):
    old, config.API_TOKEN = config.API_TOKEN, "s3cret-token"
    try:
        from jav import server
        profiles = sup_mod.config.load_profiles()
        for p in profiles.values():
            p.backend = "fake"
        from conftest import FakeBackend
        sup_mod.register_backend("fake", FakeBackend)
        sup = sup_mod.Supervisor(fresh_store, profiles, mem_probe=lambda: 60000,
                                 vram_probe=lambda: {}, vram_free_probe=lambda: 60000)
        app = server.create_app(profiles=profiles, store=fresh_store, sup=sup)
        with TestClient(app) as c:
            assert c.get("/v1/health").status_code == 200
            assert c.post("/v1/assets", content=b"x").status_code == 401
            assert c.post("/v1/assets?kind=image", content=b"x",
                          headers={"Authorization": "Bearer wrong"}).status_code == 401
            sha = hashlib.sha256(b"").hexdigest()
            ok = c.post("/v1/assets?kind=image", content=b"",
                        headers={"Authorization": "Bearer s3cret-token"})
            assert ok.status_code in (400, 201)  # passed auth; empty body is 400
            assert c.get("/v1/jobs").status_code == 200  # GET stays open
            # internal endpoints bypass bearer middleware but still need the secret
            assert c.post("/v1/internal/pipeline_status",
                          json={"status": "unloaded"}).status_code == 403
            assert c.post("/v1/internal/pipeline_status",
                          json={"status": "unloaded"},
                          headers={"X-JAV-Callback":
                                   config.CALLBACK_SECRET}).status_code == 200
    finally:
        config.API_TOKEN = old
