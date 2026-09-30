"""JAV Python SDK — single-file client for notebooks.

    from jav import Client
    c = Client("http://127.0.0.1:8765")          # or via SSH tunnel
    job = c.submit("zit", "t2i", inputs={"prompt": "a red cube"})
    print(job.wait().status, job.outputs()[0])

Batch:

    batch = c.submit_batch(
        {"provider": "zit", "workflow": "i2v-less t2i", "generation": {"steps": 9}},
        [{"inputs": {"prompt": p}} for p in prompts])
    batch.wait()

Assets are uploaded by path (content-addressed, dedup server-side).
"""
from __future__ import annotations

import mimetypes
import time
from pathlib import Path

import requests

TERMINAL = ("completed", "failed", "cancelled")


class Job:
    def __init__(self, client: "Client", data: dict):
        self.client = client
        self.__dict__.update({k: v for k, v in data.items() if k != "outputs"})
        self.id = data["id"]
        self.data = data

    @property
    def status(self) -> str:
        return self.refresh()["status"]

    def refresh(self) -> dict:
        self.data = self.client._get(f"/v1/jobs/{self.id}")
        return self.data

    def wait(self, timeout: float = 1800, interval: float = 2.0) -> "JobResult":
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                d = self.refresh()
            except Exception:
                # transient poll failure (server busy/swap thrash): keep waiting
                time.sleep(interval)
                continue
            if d["status"] in TERMINAL:
                return JobResult(self, d)
            time.sleep(interval)
        raise TimeoutError(f"job {self.id} not terminal within {timeout}s")

    def cancel(self) -> dict:
        return self.client._request("DELETE", f"/v1/jobs/{self.id}")


class JobResult:
    def __init__(self, job: Job, data: dict):
        self.job = job
        self.status = data["status"]
        self.error = data.get("error")
        self.outputs = data.get("outputs", [])

    def download(self, path: str | Path, asset_id: str | None = None) -> Path:
        url = f"/v1/jobs/{self.job.id}/output"
        params = {"asset_id": asset_id} if asset_id else None
        r = requests.get(self.job.client.base + url, params=params, timeout=300)
        r.raise_for_status()
        p = Path(path)
        p.write_bytes(r.content)
        return p


class Batch:
    def __init__(self, client: "Client", data: dict):
        self.client = client
        self.batch_id = data["batch_id"]
        self.jobs = [Job(client, {"id": j}) for j in data["jobs"]]

    def wait(self, timeout: float = 3600, interval: float = 3.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            b = self.client._get(f"/v1/batches/{self.batch_id}")
            counts = b["counts"]
            done = sum(n for s, n in counts.items() if s in TERMINAL)
            if counts and done == sum(counts.values()):
                return b
            time.sleep(interval)
        raise TimeoutError(f"batch {self.batch_id} incomplete within {timeout}s")

    def cancel(self) -> dict:
        return self.client._request("DELETE", f"/v1/batches/{self.batch_id}")


class Client:
    def __init__(self, base: str = "http://127.0.0.1:8765", token: str | None = None):
        self.base = base.rstrip("/")
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # ---- low level ----
    def _request(self, method: str, path: str, **kw) -> dict:
        r = self.session.request(method, self.base + path, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str, **kw) -> dict:
        return self._request("GET", path, **kw)

    # ---- assets ----
    def upload_asset(self, path: str | Path, kind: str | None = None) -> str:
        p = Path(path)
        if kind is None:
            mime = mimetypes.guess_type(p.name)[0] or ""
            kind = mime.split("/")[0] if mime.split("/")[0] in ("image", "video", "audio") else None
        data = p.read_bytes()
        headers = {"x-filename": p.name}
        if kind:
            headers["content-type"] = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        r = self.session.post(self.base + f"/v1/assets?kind={kind or ''}",
                              data=data, headers=headers, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"asset upload failed {r.status_code}: {r.text[:200]}")
        return r.json()["id"]

    # ---- jobs ----
    def submit(self, provider: str, workflow: str, inputs: dict | None = None,
               generation: dict | None = None, client_ref: str | None = None,
               priority: int = 0) -> Job:
        payload = {"provider": provider, "workflow": workflow,
                   "inputs": inputs or {}, "generation": generation or {},
                   "priority": priority}
        if client_ref:
            payload["client_ref"] = client_ref
        return Job(self, self._request("POST", "/v1/jobs", json=payload))

    def submit_batch(self, shared: dict, jobs: list[dict],
                     client_ref: str | None = None) -> Batch:
        body = {"shared": shared, "jobs": jobs}
        if client_ref:
            body["client_ref"] = client_ref
        return Batch(self, self._request("POST", "/v1/jobs/batch", json=body))

    def job(self, job_id: str) -> Job:
        return Job(self, self._get(f"/v1/jobs/{job_id}"))

    def jobs(self, status: str | None = None, provider: str | None = None,
             batch_id: str | None = None, limit: int = 100) -> list[dict]:
        params = {k: v for k, v in (("status", status), ("provider", provider),
                                    ("batch_id", batch_id), ("limit", limit)) if v}
        return self._get("/v1/jobs", params=params)["jobs"]

    # ---- meta ----
    def capabilities(self) -> dict:
        return self._get("/v1/capabilities")

    def runtime(self) -> dict:
        return self._get("/v1/runtime")

    def keepalive(self, ttl_s: int | None = None) -> dict:
        """Keep the active runtime resident (refresh idle-unload timer)."""
        path = "/v1/runtime/keepalive" + (f"?ttl_s={int(ttl_s)}" if ttl_s else "")
        return self._request("POST", path)

    def unload(self) -> dict:
        return self._request("POST", "/v1/runtime/unload")

    def queue(self) -> dict:
        return self._get("/v1/queue")

    def health(self) -> dict:
        return self._get("/v1/health")
