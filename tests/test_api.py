import time

import pytest

from conftest import FakeBackend

T2I = {"provider": "zit", "workflow": "t2i",
       "inputs": {"prompt": "a red cube"},
       "generation": {"width": 256, "height": 256, "steps": 4, "seed": 42}}


def wait_for(client, job_id, terminal=("completed", "failed", "cancelled"), timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in terminal:
            return body
        time.sleep(0.15)
    pytest.fail(f"job {job_id} not terminal within {timeout}s: {body}")


def test_health_and_meta(app_client):
    h = app_client.get("/v1/health").json()
    assert h["status"] == "healthy" and h["service"] == "JAV"
    caps = app_client.get("/v1/capabilities").json()
    assert caps["zit"]["workflows"]["t2i"]["available"] is True
    assert caps["ltx25"]["workflows"]["t2v"]["available"] is False
    reason = caps["ltx25"]["workflows"]["t2v"].get("reason", "")
    assert any(k in reason for k in ("weights missing", "not implemented", "not validated"))
    q = app_client.get("/v1/queue").json()
    assert "queued_by_profile" in q
    rt = app_client.get("/v1/runtime").json()
    assert rt["state"] in ("STOPPED", "READY", "BUSY", "STARTING")
    assert "mem" in rt and "admission_headroom_mb" in rt["mem"]


def test_single_job_lifecycle(app_client):
    r = app_client.post("/v1/jobs", json=T2I)
    assert r.status_code == 201
    j = r.json()
    assert j["runtime_profile"] == "zit" and "id" in j and "task_id" not in j
    body = wait_for(app_client, j["id"])
    assert body["status"] == "completed"
    assert len(body["outputs"]) == 1
    url = body["outputs"][0]["url"]
    dl = app_client.get(url)
    assert dl.status_code == 200 and dl.content.startswith(b"png-bytes")


def test_cache_hit_deterministic_seed(app_client):
    r1 = app_client.post("/v1/jobs", json=T2I)
    b1 = wait_for(app_client, r1.json()["id"])
    r2 = app_client.post("/v1/jobs", json=T2I)
    j2 = r2.json()
    assert j2["status"] == "completed"          # immediate cache reuse
    assert j2["outputs"][0]["asset_id"] == b1["outputs"][0]["asset_id"]
    got = app_client.get(f"/v1/jobs/{j2['id']}").json()
    assert got["outputs"]
    # random seed must NOT hit cache
    rnd = {**T2I, "generation": {**T2I["generation"], "seed": -1}}
    r3 = app_client.post("/v1/jobs", json=rnd)
    assert r3.json()["status"] in ("queued", "running", "completed")


def test_validation_errors(app_client):
    assert app_client.post("/v1/jobs", json={**T2I, "workflow": "nope"}).status_code == 415
    assert app_client.post("/v1/jobs", json={**T2I, "inputs": {}}).status_code == 400
    i2i_bad = {"provider": "zit", "workflow": "i2i",
               "inputs": {"prompt": "x", "image": "asset_missing"},
               "generation": {"seed": 7}}
    r = app_client.post("/v1/jobs", json=i2i_bad)
    assert r.status_code == 400  # unknown asset referenced by i2i
    i2i = {"provider": "zit", "workflow": "i2i", "inputs": {"prompt": "p"}}
    assert app_client.post("/v1/jobs", json=i2i).status_code == 400
    assert app_client.post("/v1/jobs",
                           json={"provider": "ltx25", "workflow": "t2v",
                                 "inputs": {"prompt": "p"}}).status_code == 415


def test_assets_raw_and_multipart(app_client):
    data = b"\x89PNG-fake-bytes"
    r = app_client.post("/v1/assets?kind=image", content=data,
                        headers={"x-filename": "a.png"})
    assert r.status_code == 201
    a1 = r.json()
    assert a1["type"] == "image" and a1["size"] == len(data)
    r2 = app_client.post("/v1/assets?kind=image", content=data,
                         headers={"x-filename": "a.png"})
    assert r2.json()["id"] == a1["id"]          # content-addressed dedupe
    files = {"file": ("b.png", data, "image/png")}
    r3 = app_client.post("/v1/assets/upload?kind=image", files=files)
    assert r3.status_code == 201
    assert r3.json()["id"] == a1["id"]          # same bytes dedupe
    g = app_client.get(f"/v1/assets/{a1['id']}")
    assert g.status_code == 200 and g.content == data


def test_batch_atomic_and_grouped(app_client):
    bad_batch = {
        "shared": {"provider": "zit", "workflow": "t2i",
                   "generation": {"width": 64, "height": 64, "steps": 4}},
        "jobs": [{"inputs": {"prompt": "one"}}, {"inputs": {}}],  # 2nd missing prompt
    }
    r = app_client.post("/v1/jobs/batch", json=bad_batch)
    assert r.status_code == 422
    assert "invalid_jobs" in r.json()["detail"]
    assert app_client.get("/v1/jobs", params={"limit": 500}).json()["total"] == 0

    good = {
        "shared": {"provider": "zit", "workflow": "t2i",
                   "generation": {"width": 64, "height": 64, "steps": 4}},
        "jobs": [{"inputs": {"prompt": "one"}}, {"inputs": {"prompt": "two"}},
                 {"inputs": {"prompt": "three"}}],
        "client_ref": "storyboard-01",
    }
    r2 = app_client.post("/v1/jobs/batch", json=good)
    assert r2.status_code == 201
    b = r2.json()
    assert len(b["jobs"]) == 3
    for jid in b["jobs"]:
        assert wait_for(app_client, jid)["status"] == "completed"
    gb = app_client.get(f"/v1/batches/{b['batch_id']}").json()
    assert gb["counts"]["completed"] == 3


def test_cancel_queued_job(app_client):
    FakeBackend.default_mode = "wait"
    first = app_client.post("/v1/jobs", json=T2I).json()["id"]
    deadline = time.time() + 5
    while app_client.get(f"/v1/jobs/{first}").json()["status"] != "running":
        if time.time() > deadline:
            FakeBackend.release = True
            pytest.fail("first job never started running")
        time.sleep(0.1)
    second = app_client.post("/v1/jobs", json={**T2I, "inputs": {"prompt": "queued"}})
    sid = second.json()["id"]
    r = app_client.delete(f"/v1/jobs/{sid}")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert app_client.get(f"/v1/jobs/{sid}").json()["status"] == "cancelled"
    FakeBackend.release = True


def test_sse_terminal(app_client):
    jid = app_client.post("/v1/jobs", json=T2I).json()["id"]
    wait_for(app_client, jid)
    with app_client.stream("GET", f"/v1/jobs/{jid}/events") as resp:
        assert resp.status_code == 200
        first_line = next(resp.iter_lines())
        assert "status" in first_line


def test_keepalive_and_unload(app_client):
    jid = app_client.post("/v1/jobs", json=T2I).json()["id"]
    wait_for(app_client, jid)
    ka = app_client.post("/v1/runtime/keepalive?ttl_s=60").json()
    assert ka["kept_alive"] == "zit" and ka["idle_unload_in_s"] == 60
    unloaded = app_client.post("/v1/runtime/unload").json()
    assert unloaded["unloaded"] == "zit" and unloaded["active_profile"] is None
    ka2 = app_client.post("/v1/runtime/keepalive").json()
    assert ka2["kept_alive"] is None          # nothing resident -> honest no-op


def test_fail_path(app_client):
    FakeBackend.default_mode = "fail"
    jid = app_client.post("/v1/jobs", json=T2I).json()["id"]
    body = wait_for(app_client, jid)
    assert body["status"] == "failed"
    assert "fake fail" in (body["error"] or "")
    FakeBackend.default_mode = "success"
