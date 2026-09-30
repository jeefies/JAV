from datetime import datetime, timedelta, timezone

import pytest

from jav.store import Store


@pytest.fixture
def store(fresh_store):
    return fresh_store


def mk(store, profile="p1", priority=0, age_s=0, status="queued", **kw):
    j = store.create_job(provider="zit", workflow="t2i", runtime_profile=profile,
                         payload={"prompt": "x", "generation": {"seed": 1}},
                         assets=[], priority=priority, **kw)
    if age_s or status != "queued":
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
        store.update_job(j["id"], created_at=ts, status=status)
    return store.get_job(j["id"])


def test_roundtrip(store):
    j = mk(store)
    assert j["status"] == "queued"
    store.set_status(j["id"], "running")
    got = store.get_job(j["id"])
    assert got["status"] == "running" and got["started_at"]
    store.set_status(j["id"], "completed")
    assert store.get_job(j["id"])["finished_at"]


def test_claim_order_priority_then_age(store):
    a = mk(store, priority=0, age_s=100)
    b = mk(store, priority=5, age_s=1)
    c = mk(store, profile="p2")
    assert store.claim_next("p1")["id"] == b["id"]
    assert store.claim_next("p1")["id"] == a["id"]
    assert store.claim_next("p1") is None
    assert store.claim_next("p2")["id"] == c["id"]
    assert store.get_job(b["id"])["status"] == "starting_runtime"


def test_queue_position(store):
    mk(store, age_s=300)
    second = mk(store, age_s=1)
    assert store.queue_position(second["id"]) == 2


def test_cancel_queued_only(store):
    j = mk(store)
    assert store.cancel_queued(j["id"])
    assert store.get_job(j["id"])["status"] == "cancelled"
    assert not store.cancel_queued(j["id"])


def test_recover_interrupted(store):
    j1 = mk(store, status="running")
    j2 = mk(store, status="starting_runtime")
    j3 = mk(store, status="completed")
    ids = store.recover_interrupted()
    assert set(ids) == {j1["id"], j2["id"]}
    assert store.get_job(j1["id"])["status"] == "queued"
    assert store.get_job(j3["id"])["status"] == "completed"


def test_asset_dedupe(store, tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"1234")
    a1 = store.put_asset(sha256="ab" * 32, kind="image", path=str(p), size=4)
    a2 = store.put_asset(sha256="ab" * 32, kind="image", path=str(p), size=4)
    assert a1["id"] == a2["id"]


def test_cache_hit_lookup(store):
    ck = "deadbeef"
    j = mk(store, status="completed", cache_key=ck)
    from pathlib import Path
    f = Path("/tmp/kilo/jav-test/data/outputs/cachetest.png")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"img")
    store.add_output(j["id"], "image", None, str(f))
    hit = store.find_cache_hit(ck)
    assert hit and hit["id"] == j["id"]
    assert store.find_cache_hit("nope") is None


def test_batch_aggregates(store):
    b = store.create_batch("ref", {"size": 2})
    mk(store, batch_id=b["id"])
    mk(store, batch_id=b["id"], status="completed")
    got = store.get_batch(b["id"])
    assert got["counts"] == {"queued": 1, "completed": 1}
    assert len(got["jobs"]) == 2
