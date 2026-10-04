"""Provider registry: validation, defaults, compile-to-runtime-task."""
from __future__ import annotations

from .. import config
from ..models import PROVIDER_WORKFLOWS, ProviderError
from . import cosyvoice, ltx25, mh3, zit

PROVIDERS = {"zit": zit, "ltx25": ltx25, "mh3": mh3, "cosyvoice": cosyvoice}


def resolve_profile(provider: str, workflow: str) -> str:
    profile = PROVIDER_WORKFLOWS.get(provider, {}).get(workflow)
    if not profile:
        raise ProviderError(f"unsupported workflow {provider}.{workflow}", 400)
    return profile


def normalize(provider: str, workflow: str, inputs: dict, generation: dict) -> dict:
    """Apply provider defaults; returns payload dict (validated, complete)."""
    return PROVIDERS[provider].normalize(workflow, inputs, generation)


def asset_slots(payload: dict) -> dict[str, str]:
    return PROVIDERS[payload["provider"]].asset_slots(payload)


def cache_key(payload: dict) -> str | None:
    return PROVIDERS[payload["provider"]].cache_key(payload)


def compile(payload: dict, asset_paths: dict[str, str], output_dir) -> dict:
    """payload -> runtime task dict (worker/comfy level)."""
    return PROVIDERS[payload["provider"]].compile(
        payload, asset_paths, output_dir, base_dir=config.BASE_DIR)
