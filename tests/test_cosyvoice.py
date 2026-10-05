"""cosyvoice.t2a provider + voice registry + API plumbing tests."""
import json
import time

import pytest
import yaml

from jav import config
from jav import voices
from jav.providers import cosyvoice as cv

T2A = {"provider": "cosyvoice", "workflow": "t2a",
       "inputs": {"text": "那之前公布的两年呢？", "voice_id": "qiyuan",
                  "instruction": "听完后克制地追问"},
       "generation": {"speed": 1.0, "seed": 42}}


def _write_voices(entries):
    p = config.BASE_DIR / "config" / "voices.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False))


def _seed_flags():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "capability_flags.json").write_text(
        json.dumps({"cosyvoice.t2a": True}))


@pytest.fixture
def registry():
    _write_voices([
        {"id": "qiyuan", "name": "齐远", "prompt_text": "参考音频逐字稿。",
         "asset": "asset_qy1234"},
        {"id": "teacher", "name": "叶老师", "prompt_text": "另一段逐字稿。",
         "path": str(config.DATA_DIR / "refs/teacher.wav")},
    ])
    yield
    (config.BASE_DIR / "config" / "voices.yaml").unlink(missing_ok=True)
    voices.load_voices(force=True)


# ---------------- provider unit ----------------
def test_wrap_instruct():
    assert cv.wrap_instruct("克制地追问") == \
        "You are a helpful assistant. 克制地追问<|endofprompt|>"
    s = "custom prefix.<|endofprompt|>"
    assert cv.wrap_instruct(s) == s


def test_normalize_requires_voice(registry):
    with pytest.raises(ValueError, match="voice_id"):
        cv.normalize("t2a", {"text": "hello"}, {})
    with pytest.raises(ValueError, match="unknown voice_id"):
        cv.normalize("t2a", {"text": "hello", "voice_id": "ghost"}, {})
    with pytest.raises(ValueError, match="reference_text"):
        cv.normalize("t2a", {"text": "hi", "reference_audio": "asset_x"}, {})
    with pytest.raises(ValueError, match="text"):
        cv.normalize("t2a", {"voice_id": "qiyuan"}, {})


def test_normalize_registry_and_inline(registry):
    p = cv.normalize("t2a", T2A["inputs"], T2A["generation"])
    assert p["voice_id"] == "qiyuan"
    assert p["assets"] == {"reference_audio": "asset_qy1234"}
    assert p["instruct_text"].endswith("<|endofprompt|>")
    assert p["generation"]["sample_rate"] == 48000

    p2 = cv.normalize("t2a", {"text": "x", "reference_audio": "asset_a",
                              "reference_text": "逐字稿", "instruction": ""}, {})
    assert p2["voice_id"] == "inline:asset_a"
    assert p2["instruct_text"] is None

    # 语速/采样率校验
    with pytest.raises(ValueError, match="speed"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan"}, {"speed": 3})
    with pytest.raises(ValueError, match="sample_rate"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan"}, {"sample_rate": 8000})


def test_cache_key_determinism(registry):
    p1 = cv.normalize("t2a", T2A["inputs"], T2A["generation"])
    p1["job_id"] = "j1"
    k1 = cv.cache_key(p1)
    p2 = cv.normalize("t2a", T2A["inputs"], T2A["generation"])
    assert cv.cache_key(p2) == k1
    p2["instruct_text"] = None
    assert cv.cache_key(p2) != k1
    p2["instruct_text"] = p1["instruct_text"]
    p2["voice_id"] = "teacher"
    assert cv.cache_key(p2) != k1
    # 随机 seed 不缓存（同 zit 姿态）
    p3 = cv.normalize("t2a", T2A["inputs"], {"seed": -1})
    assert cv.cache_key(p3) is None


def test_compile_paths(registry):
    p = cv.normalize("t2a", T2A["inputs"], T2A["generation"])
    p["job_id"] = "job_x"
    task = cv.compile(p, {"reference_audio": "/tmp/kilo/qy.wav"},
                      "/tmp/kilo/out", config.BASE_DIR)
    assert task["mode"] == "t2a"
    assert task["prompt_wav"] == "/tmp/kilo/qy.wav"
    assert task["instruct_text"].startswith("You are a helpful assistant.")

    p2 = cv.normalize("t2a", {"text": "早", "voice_id": "teacher"}, {})
    p2["job_id"] = "job_y"
    task2 = cv.compile(p2, {}, "/tmp/kilo/out", config.BASE_DIR)
    assert task2["prompt_wav"] == str(config.DATA_DIR / "refs/teacher.wav")
    assert task2["instruct_text"] is None


# ---------------- voices registry ----------------
def test_voice_parse_errors():
    cases = [
        ([{"id": "A", "prompt_text": "x", "path": "/mnt/data/AV/a.wav"}], "bad voice id"),
        ([{"id": "a", "prompt_text": "x", "path": "/tmp/a.wav"}], "under /mnt/data/AV"),
        ([{"id": "a", "prompt_text": "x"}], "exactly one"),
        ([{"id": "a", "prompt_text": ""}], "prompt_text"),
        ([{"id": "a", "prompt_text": "x", "asset": "1"},
          {"id": "a", "prompt_text": "y", "asset": "2"}], "duplicate"),
        ([{"id": "a", "prompt_text": "x", "asset": "1", "mode": "fast"}], "unknown key"),
    ]
    for entries, msg in cases:
        _write_voices(entries)
        with pytest.raises(voices.VoiceError, match=msg):
            voices.load_voices(force=True)
    (config.BASE_DIR / "config" / "voices.yaml").unlink(missing_ok=True)


def test_upsert_voice(registry):
    v = voices.upsert_voice({"id": "jiwei", "name": "纪伟", "asset": "asset_jw",
                             "prompt_text": "纪伟参考逐字稿。", "description": "主角"})
    assert v["id"] == "jiwei" and v["asset"] == "asset_jw"
    # 替换同名条目，不重复追加
    voices.upsert_voice({"id": "jiwei", "name": "纪伟旁白", "asset": "asset_jw",
                         "prompt_text": "新逐字稿。"})
    all_v = voices.load_voices()
    assert all_v["jiwei"]["prompt_text"] == "新逐字稿。"
    assert len([k for k in all_v if k == "jiwei"]) == 1


# ---------------- app/API ----------------
def test_capability_gated_until_validated(app_client):
    (config.DATA_DIR / "capability_flags.json").unlink(missing_ok=True)
    import jav.capabilities as caps
    caps._CAPS_CACHE = None
    r = app_client.post("/v1/jobs", json=T2A)
    assert r.status_code == 415 and "not validated" in r.json()["detail"]


def test_t2a_lifecycle_duration_and_download(app_client, registry):
    _seed_flags()
    _write_voices([{"id": "qiyuan", "prompt_text": "逐字稿",
                    "path": str(config.DATA_DIR / "refs/qy.wav")}])
    r = app_client.post("/v1/jobs", json=T2A)
    assert r.status_code == 201, r.text
    jid = r.json()["id"]
    assert r.json()["runtime_profile"] == "cosyvoice"
    deadline = time.time() + 15
    body = None
    while time.time() < deadline:
        body = app_client.get(f"/v1/jobs/{jid}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.15)
    assert body["status"] == "completed", body
    o = body["outputs"][0]
    assert o["kind"] == "audio"
    assert o["duration_s"] == 2.5          # worker meta 落进 asset 再透出
    dl = app_client.get(o["url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("audio/wav")
    assert dl.content.startswith(b"wav-bytes")


def test_t2a_cache_hit_reuse(app_client, registry):
    _seed_flags()
    _write_voices([{"id": "qiyuan", "prompt_text": "逐字稿",
                    "path": str(config.DATA_DIR / "refs/qy.wav")}])
    r1 = app_client.post("/v1/jobs", json=T2A)
    j1 = r1.json()["id"]
    deadline = time.time() + 15
    while time.time() < deadline:
        if app_client.get(f"/v1/jobs/{j1}").json()["status"] == "completed":
            break
        time.sleep(0.15)
    r2 = app_client.post("/v1/jobs", json=T2A)
    j2 = r2.json()
    assert j2["status"] == "completed"           # §7 成功音频缓存复用
    assert j2["outputs"][0]["asset_id"] == \
        app_client.get(f"/v1/jobs/{j1}").json()["outputs"][0]["asset_id"]
    # 换语速 = 新参数组合，不复用
    r3 = app_client.post("/v1/jobs", json={**T2A, "generation": {"speed": 1.2, "seed": 42}})
    assert r3.json()["status"] == "queued"


def test_voices_api(app_client, registry):
    a = app_client.post("/v1/assets?kind=audio", content=b"RIFFfake",
                        headers={"x-filename": "jiwei_ref.wav"})
    assert a.status_code == 201, a.text
    aid = a.json()["id"]
    r = app_client.post("/v1/voices", json={"id": "jiwei", "name": "纪伟",
                                            "prompt_asset": aid,
                                            "prompt_text": "纪伟逐字稿。"})
    assert r.status_code == 201, r.text
    lst = app_client.get("/v1/voices").json()["voices"]
    assert {"id": "jiwei", "name": "纪伟", "source": "asset",
            "description": ""} in lst or any(v["id"] == "jiwei" for v in lst)
    # 非音频资产拒绝
    img = app_client.post("/v1/assets?kind=image", content=b"PNG",
                          headers={"x-filename": "x.png"}).json()["id"]
    bad = app_client.post("/v1/voices", json={"id": "oops", "prompt_asset": img,
                                              "prompt_text": "x"})
    assert bad.status_code == 400
    # 注册后的 voice 可直接用于 t2a
    _seed_flags()
    job = app_client.post("/v1/jobs", json={
        "provider": "cosyvoice", "workflow": "t2a",
        "inputs": {"text": "旁白测试", "voice_id": "jiwei"},
        "generation": {"seed": 7}})
    assert job.status_code == 201, job.text
    (config.BASE_DIR / "config" / "voices.yaml").unlink(missing_ok=True)


def test_voice_sample_endpoint(app_client, registry):
    # asset 型：注册后即可试听，内容与资产字节一致
    a = app_client.post("/v1/assets?kind=audio", content=b"RIFFaud",
                        headers={"x-filename": "qy.wav"})
    aid = a.json()["id"]
    voices.upsert_voice({"id": "qy-s", "asset": aid, "prompt_text": "稿。"})
    r = app_client.get("/v1/voices/qy-s/sample")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content == b"RIFFaud"
    # path 型：文件缺失 410，存在则原样返回
    assert app_client.get("/v1/voices/teacher/sample").status_code == 410
    p = config.DATA_DIR / "refs/teacher.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"wavdata")
    r3 = app_client.get("/v1/voices/teacher/sample")
    assert r3.status_code == 200 and r3.content == b"wavdata"
    assert app_client.get("/v1/voices/ghost/sample").status_code == 404
    p.unlink(missing_ok=True)


# ---------------- wants.md v2: strict params ----------------
def test_strict_param_rejection(registry):
    # 未知 input / generation key 明确报错，不能接受后忽略（§3）
    with pytest.raises(ValueError, match="unsupported input"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan", "emotion": "sad"}, {})
    with pytest.raises(ValueError, match="unsupported generation"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan"}, {"temp": 0.7})
    with pytest.raises(ValueError, match="instruction too long"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan",
                             "instruction": "长" * 501}, {})
    with pytest.raises(ValueError, match="duration_limit_s"):
        cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan", "duration_limit_s": 0}, {})
    p = cv.normalize("t2a", {"text": "x", "voice_id": "qiyuan", "duration_limit_s": 4.0}, {})
    assert p["duration_limit_s"] == 4.0 and p["instruction"] == ""


# ---------------- voices registry v2: no silent overwrite ----------------
def test_register_409_replace_history(registry):
    v, created = voices.register_voice({"id": "jiwei", "asset": "a1",
                                        "prompt_text": "第一版逐字稿。", "name": "纪伟"})
    assert created and v["version"] == 1
    # 重复注册默认拒绝——不默默覆盖（§2）
    with pytest.raises(voices.VoiceError) as ei:
        voices.register_voice({"id": "jiwei", "asset": "a2", "prompt_text": "换人"})
    assert ei.value.status == 409 and "replace=true" in str(ei.value)
    assert voices.get_voice("jiwei")["asset"] == "a1"     # 未被动过
    # 显式 replace：版本 +1，旧内容进 history
    v2, created2 = voices.register_voice({"id": "jiwei", "asset": "a2",
                                          "prompt_text": "新逐字稿", "name": "纪伟B"},
                                         replace=True, note="候选B胜出")
    assert not created2 and v2["version"] == 2
    assert v2["asset"] == "a2" and v2["revisions"] if "revisions" in v2 else True
    hist = v2["history"]
    assert hist and hist[-1]["asset"] == "a1" and hist[-1]["note"] == "候选B胜出"
    got = voices.delete_voice("jiwei")
    assert got["asset"] == "a2"
    with pytest.raises(voices.VoiceError) as ei:
        voices.delete_voice("jiwei")
    assert ei.value.status == 404


def test_voice_metadata_fields(registry):
    v, _ = voices.register_voice({
        "id": "comm-m1", "asset": "a9", "prompt_text": "逐字稿",
        "kind": "community", "role": "qiyuan", "tags": ["male", "bright"],
        "license": "CC-BY-4.0", "provenance": "https://example.org/voice-pack",
        "model": "Fun-CosyVoice3-0.5B"})
    pv = voices.public_view({"comm-m1": v})[0]
    assert pv["kind"] == "community" and pv["role"] == "qiyuan"
    assert pv["license"] == "CC-BY-4.0" and pv["version"] == 1
    dv = voices.detail_view(v)
    assert dv["provenance"].startswith("https://") and dv["transcript_chars"] > 0
    assert "path" not in dv and "prompt_text" not in dv   # 不泄漏
    with pytest.raises(voices.VoiceError, match="kind"):
        voices.register_voice({"id": "bad-k", "asset": "a", "prompt_text": "x",
                               "kind": "magic"})


# ---------------- API v2 ----------------
def _mk_asset(app_client):
    a = app_client.post("/v1/assets?kind=audio", content=b"RIFFaud",
                        headers={"x-filename": "ref.wav"})
    return a.json()["id"]


def test_voices_api_v2(app_client, registry):
    aid = _mk_asset(app_client)
    body = {"id": "qy2", "prompt_asset": aid, "prompt_text": "逐字稿。"}
    r = app_client.post("/v1/voices", json=body)
    assert r.status_code == 201, r.text
    assert r.json()["version"] == 1 and r.json()["replaced"] is False
    # 默默覆盖被禁止：409
    assert app_client.post("/v1/voices", json=body).status_code == 409
    rep = app_client.post("/v1/voices", json={**body, "replace": True, "note": "v2"})
    assert rep.status_code == 200 and rep.json()["version"] == 2
    # 详情：版本列表、无路径/逐字稿泄漏
    d = app_client.get("/v1/voices/qy2")
    assert d.status_code == 200
    dj = d.json()
    assert dj["version"] == 2 and len(dj["revisions"]) == 1
    assert dj["reference"]["available"] is True and dj["reference"]["bytes"] == 7
    assert "prompt_text" not in dj and "path" not in dj
    # 不兼容模型声明 422（不静默入库）
    bad = app_client.post("/v1/voices", json={**body, "id": "qy3",
                                              "model": "CosyVoice2-0.5B"})
    assert bad.status_code == 422
    # preview 正名 + sample 别名兼容
    for ep in ("preview", "sample"):
        pr = app_client.get(f"/v1/voices/qy2/{ep}")
        assert pr.status_code == 200 and pr.content == b"RIFFaud"
    assert app_client.get("/v1/voices/ghost/preview").status_code == 404
    assert app_client.get("/v1/voices/ghost").status_code == 404
    # 删除后引用任务快速失败
    assert app_client.delete("/v1/voices/qy2").json()["deleted"] is True
    assert app_client.get("/v1/voices/qy2").status_code == 404
    _seed_flags()
    job = app_client.post("/v1/jobs", json={
        "provider": "cosyvoice", "workflow": "t2a",
        "inputs": {"text": "喂", "voice_id": "qy2"}})
    assert job.status_code == 400 and "unknown voice_id" in job.text
    assert app_client.delete("/v1/voices/qy2").status_code == 404


def test_strict_params_api(app_client, registry):
    _seed_flags()
    _write_voices([{"id": "qiyuan", "prompt_text": "稿",
                    "path": str(config.DATA_DIR / "refs/qy.wav")}])
    r = app_client.post("/v1/jobs", json={
        "provider": "cosyvoice", "workflow": "t2a",
        "inputs": {"text": "x", "voice_id": "qiyuan", "pitch": 3}})
    assert r.status_code == 400 and "unsupported input" in r.text
    r2 = app_client.post("/v1/jobs", json={
        "provider": "cosyvoice", "workflow": "t2a",
        "inputs": {"text": "x", "voice_id": "qiyuan"},
        "generation": {"temp": 1}})
    assert r2.status_code == 400 and "unsupported generation" in r2.text


def test_idempotent_submit_and_replay(app_client, registry):
    _seed_flags()
    _write_voices([{"id": "qiyuan", "prompt_text": "稿",
                    "path": str(config.DATA_DIR / "refs/qy.wav")}])
    spec = {**T2A, "client_ref": "ep01-line-07"}
    r1 = app_client.post("/v1/jobs", json=spec)
    assert r1.status_code == 201
    j1 = r1.json()["id"]
    r2 = app_client.post("/v1/jobs", json=spec)   # 响应丢失后重试同编号
    assert r2.status_code == 200 and r2.json()["idempotent_replay"] is True
    assert r2.json()["id"] == j1                  # 不重复生成/计费
    q = app_client.get("/v1/jobs?client_ref=ep01-line-07")
    assert q.json()["total"] == 1 and q.json()["jobs"][0]["id"] == j1
    # 批量：同 batch client_ref 重放
    batch = {"shared": {"provider": "cosyvoice", "workflow": "t2a"},
             "client_ref": "ep01-all",
             "jobs": [{"inputs": {"text": "甲", "voice_id": "qiyuan"}},
                      {"inputs": {"text": "乙", "voice_id": "qiyuan"},
                       "client_ref": "ep01-b"}]}
    b1 = app_client.post("/v1/jobs/batch", json=batch)
    assert b1.status_code == 201, b1.text
    b2 = app_client.post("/v1/jobs/batch", json=batch)
    assert b2.status_code == 200 and b2.json()["idempotent_replay"] is True
    assert b2.json()["batch_id"] == b1.json()["batch_id"]
    assert sorted(b2.json()["jobs"]) == sorted(b1.json()["jobs"])


def test_audio_receipt_fields(app_client, registry):
    _seed_flags()
    _write_voices([{"id": "qiyuan", "prompt_text": "稿",
                    "path": str(config.DATA_DIR / "refs/qy.wav")}])
    r = app_client.post("/v1/jobs", json=T2A)
    jid = r.json()["id"]
    deadline = time.time() + 15
    while time.time() < deadline:
        body = app_client.get(f"/v1/jobs/{jid}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.15)
    o = body["outputs"][0]
    assert len(o["sha256"]) == 64 and o["size_bytes"] > 0
    assert o["meta"]["mode"] in ("zero_shot", "instruct2")
    assert o["meta"]["voice_id"] == "qiyuan"
    assert "duration_s" in o["meta"]


def test_capabilities_model_block(app_client):
    caps = app_client.get("/v1/capabilities").json()
    m = caps["cosyvoice"]["model"]
    assert m["name"] == "Fun-CosyVoice3-0.5B" and m["native_sample_rate"] == 24000
    assert m["weights_fingerprint"] is None or len(m["weights_fingerprint"]) == 12
    assert m["features"]["instruction"] is True
