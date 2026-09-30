"""Pydantic domain models — the single API contract source (JAV-DESIGN 1)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Providers = Literal["zit", "mh3", "ltx25"]


class ProviderError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status

PROVIDER_WORKFLOWS: dict[str, dict[str, str]] = {
    "zit": {"t2i": "zit", "i2i": "zit", "inpaint": "zit"},
    "mh3": {"t2v": "mh3.fl2va", "i2v": "mh3.fl2va", "fl2v": "mh3.fl2va",
            "ref2v": "mh3.ref2va", "fun_control": "mh3.ref2va",
            "multiframe": "mh3.ref2va"},
    "ltx25": {"t2v": "ltx25", "i2v": "ltx25", "flf2v": "ltx25", "a2v": "ltx25",
              "bbox_control": "ltx25",
              "union_control": "ltx25", "motion_control": "ltx25",
              "inpaint": "ltx25", "outpaint": "ltx25", "ic_lora": "ltx25"},
}


class JobSubmit(BaseModel):
    provider: Providers
    workflow: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    client_ref: str | None = None

    @field_validator("workflow")
    @classmethod
    def _wf(cls, v):
        return v.strip().lower()


class BatchSpec(BaseModel):
    """shared defaults + per-job overrides (JAV-DESIGN 3.3)."""
    shared: dict[str, Any] = Field(default_factory=dict)
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    client_ref: str | None = None

    def merged(self, job: dict) -> dict:
        out: dict[str, Any] = {}
        for key in ("provider", "workflow", "priority", "client_ref"):
            if key in self.shared:
                out[key] = self.shared[key]
        for key in ("inputs", "generation"):
            if isinstance(self.shared.get(key), dict):
                out[key] = dict(self.shared[key])
        out["client_ref"] = job.get("client_ref", self.client_ref)
        for key, val in job.items():
            if key in ("inputs", "generation") and isinstance(val, dict):
                merged = dict(out.get(key, {}))
                merged.update(val)
                out[key] = merged
            else:
                out[key] = val
        return out
