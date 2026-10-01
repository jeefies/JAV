"""Workflow Compiler: template (ComfyUI API-format) + parameter manifest.

Clients never see ComfyUI node ids; ComfyUI upgrades only touch
workflows/*.api.json + *.manifest.yaml (JAV-DESIGN 5).
"""
from __future__ import annotations

import json
import re

from .. import config


class TemplateError(RuntimeError):
    pass


def load_pair(provider: str, workflow: str) -> tuple[dict, dict]:
    root = config.BASE_DIR / "jav" / "workflows" / provider
    tpl_path, man_path = root / f"{workflow}.api.json", root / f"{workflow}.manifest.json"
    if not tpl_path.exists() or not man_path.exists():
        raise TemplateError(
            f"{provider}.{workflow}: workflow template/manifest not installed "
            f"({tpl_path.name}, {man_path.name})")
    return json.loads(tpl_path.read_text()), json.loads(man_path.read_text())


def _set_deep(graph: dict, path: str, value) -> None:
    """path like '83.inputs.text' -> graph['83']['inputs']['text']=value.
    A dict value written onto a leaf input key is expanded to the v3 dynamic
    combo API form: discriminator + '<key>.<field>' dotted siblings."""
    keys = [t for t in re.split(r"[.\[\]]", path) if t]
    node = graph
    for k in keys[:-1]:
        node = node[k]
    key = keys[-1]
    if isinstance(value, dict) and key in value:
        node[key] = value[key]
        for k, v in value.items():
            if k != key:
                node[f"{key}.{k}"] = v
        return
    node[key] = value


def compile_graph(provider: str, workflow: str, params: dict,
                  asset_names: dict[str, str] | None = None) -> dict:
    """params: manifest-name -> scalar value (asset names pre-uploaded to
    ComfyUI input/). Returns a ready-to-POST /prompt graph."""
    graph, manifest = load_pair(provider, workflow)
    for name, rule in manifest.get("params", {}).items():
        if name in params:
            _set_deep(graph, rule["node"], params[name])
        elif "default" in rule:
            _set_deep(graph, rule["node"], rule["default"])
    for name, rule in manifest.get("assets", {}).items():
        if name in (asset_names or {}):
            _set_deep(graph, rule["node"], asset_names[name])
    return graph
