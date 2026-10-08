"""GET /v1/openapi.json: curated public contract view (2026-10-08)."""


def _spec(app_client):
    r = app_client.get("/v1/openapi.json")
    assert r.status_code == 200
    return r.json()


def test_version_and_default_paths_disabled(app_client):
    spec = _spec(app_client)
    assert spec["openapi"].startswith("3.1")
    # FastAPI defaults must not leak the uncurated view
    assert app_client.get("/openapi.json").status_code == 404
    assert app_client.get("/docs").status_code == 404


def test_internal_stripped_public_present(app_client):
    paths = _spec(app_client)["paths"]
    assert not [p for p in paths if p.startswith("/v1/internal")]
    for p in ("/v1/jobs", "/v1/jobs/batch", "/v1/voices", "/v1/capabilities",
              "/v1/queue", "/v1/assets", "/v1/openapi.json"):
        assert p in paths, p


def test_security_scheme_and_per_op(app_client):
    spec = _spec(app_client)
    assert spec["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    jobs = spec["paths"]["/v1/jobs"]
    assert jobs["post"]["security"] == [{"BearerAuth": []}]
    assert "security" not in jobs["get"]
    # every non-GET public op carries bearer security
    for p, item in spec["paths"].items():
        for m, op in item.items():
            if m in ("get", "head", "options", "trace"):
                continue
            assert op.get("security") == [{"BearerAuth": []}], (m, p)


def test_tags_and_summaries(app_client):
    spec = _spec(app_client)
    tag_names = {t["name"] for t in spec["tags"]}
    for p, item in spec["paths"].items():
        for m, op in item.items():
            assert len(op["tags"]) == 1 and op["tags"][0] in tag_names, (m, p)
    assert spec["paths"]["/v1/jobs"]["post"]["summary"] == "提交单个任务"
    assert spec["paths"]["/v1/jobs"]["post"]["x-summary-en"] == "Submit a job"
    assert all("x-summary-en" in op
               for item in spec["paths"].values() for op in item.values())
    assert spec["paths"]["/v1/jobs/batch"]["post"]["tags"] == ["batches"]
    assert spec["paths"]["/v1/voices/{voice_id}/preview"]["get"]["tags"] == ["voices"]
    assert spec["paths"]["/v1/queue"]["get"]["tags"] == ["system"]


def test_body_schema_from_pydantic(app_client):
    spec = _spec(app_client)
    ref = spec["paths"]["/v1/jobs"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    name = ref.split("/")[-1]
    assert name in spec["components"]["schemas"]
    assert "client_ref" in spec["components"]["schemas"][name]["properties"]


def test_no_secrets_and_stable_object(app_client):
    raw = app_client.get("/v1/openapi.json")
    assert raw.status_code == 200
    assert "JAV_API_TOKEN=" not in raw.text
    assert raw.json() == _spec(app_client)
    # curate runs once per process: the app-level cache holds that mapping
    assert app_client.app._jav_openapi_spec == raw.json()
