"""Voice registry for cosyvoice.t2a (配音角色固定音色).

config/voices.yaml maps stable per-character voice ids to a reference dry
voice (asset id or a file path under /mnt/data/AV) + its exact transcript,
which CosyVoice3 turns into a zero-shot speaker embedding. 与 profiles.yaml
一样：字段级、mtime 缓存、未知 key 直接报错。

wants.md 管理要求：
  - 每条音色带 kind(local/community/cloud)、role(角色映射)、tags(声线描述)、
    license(使用许可)、provenance(来源)、model(兼容模型) 元数据（全部可选，
    旧条目照常解析）。
  - 重复注册默认 409 拒绝，**不会默默覆盖**；显式 replace=true 才替换，并且
    旧内容进 history（版本递增，保留最近 20 版）。
  - 支持查询（GET 列表 / 单条详情含版本）、删除；试听走 asset/path 解析。
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
# 干声参考只允许这两个来源：JAV 资产库，或 AV 数据盘内的既有文件（红线同源）
_ALLOWED_PATH_ROOTS = (config.DATA_DIR.resolve(), Path("/mnt/data/AV").resolve())
_KINDS = ("local", "community", "cloud")
_HISTORY_KEEP = 20

_ENTRY_KEYS = {"id", "name", "prompt_text", "asset", "path", "description",
               "kind", "role", "tags", "license", "provenance", "model",
               "version", "created_at", "updated_at", "history"}


class VoiceError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _registry_path() -> Path:
    return config.BASE_DIR / "config" / "voices.yaml"


_CACHE: tuple[int, dict] | None = None  # (mtime_ns, voices)
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_path(vid: str, path: str) -> None:
    rp = Path(path)
    if not rp.is_absolute() or not any(
            rp == r or rp.is_relative_to(r) for r in _ALLOWED_PATH_ROOTS):
        raise VoiceError(f"voices.yaml: voice {vid!r} path must be an absolute "
                         f"file under /mnt/data/AV: {path!r}")


def _rev_snapshot(d: dict) -> dict:
    """One stored revision of a voice entry (for history)."""
    return {"version": int(d.get("version", 1)), "at": d.get("updated_at") or d.get("created_at", ""),
            "asset": d.get("asset"), "path": d.get("path"),
            "prompt_text": d.get("prompt_text", ""), "name": d.get("name", ""),
            "description": d.get("description", "")}


def _parse_entry(d: dict, seen: dict[str, dict]) -> None:
    if not isinstance(d, dict):
        raise VoiceError(f"voices.yaml: entry must be a mapping, got {type(d).__name__}")
    unknown = set(d) - _ENTRY_KEYS
    if unknown:
        raise VoiceError(f"voices.yaml: unknown key(s) {sorted(unknown)} "
                         f"in voice {d.get('id')!r}")
    vid = str(d.get("id", "")).strip()
    if not VOICE_ID_RE.match(vid):
        raise VoiceError(f"voices.yaml: bad voice id {vid!r} "
                         "(want [a-z0-9][a-z0-9_-] up to 32 chars)")
    if vid in seen:
        raise VoiceError(f"voices.yaml: duplicate voice id {vid!r}")
    text = str(d.get("prompt_text", "")).strip()
    if not text:
        raise VoiceError(f"voices.yaml: voice {vid!r} needs a non-empty prompt_text "
                         "(逐字参考稿)")
    asset = str(d.get("asset", "")).strip()
    path = str(d.get("path", "")).strip()
    if bool(asset) == bool(path):
        raise VoiceError(f"voices.yaml: voice {vid!r} needs exactly one of "
                         "'asset' / 'path'")
    if path:
        _validate_path(vid, path)
    kind = str(d.get("kind", "local")).strip().lower() or "local"
    if kind not in _KINDS:
        raise VoiceError(f"voices.yaml: voice {vid!r} kind must be one of {_KINDS}")
    tags = d.get("tags", []) or []
    if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
        raise VoiceError(f"voices.yaml: voice {vid!r} tags must be a list of strings")
    version = int(d.get("version", 1))
    if version < 1:
        raise VoiceError(f"voices.yaml: voice {vid!r} version must be >= 1")
    history = d.get("history", []) or []
    if not isinstance(history, list) or any(not isinstance(h, dict) for h in history):
        raise VoiceError(f"voices.yaml: voice {vid!r} history must be a list of mappings")
    seen[vid] = {
        "id": vid,
        "name": str(d.get("name", "")).strip() or vid,
        "prompt_text": text,
        "asset": asset or None,
        "path": path or None,
        "description": str(d.get("description", "")).strip(),
        "kind": kind,
        "role": str(d.get("role", "")).strip(),
        "tags": [x.strip() for x in tags if str(x).strip()],
        "license": str(d.get("license", "")).strip(),
        "provenance": str(d.get("provenance", "")).strip(),
        "model": str(d.get("model", "")).strip(),
        "version": version,
        "created_at": str(d.get("created_at", "") or ""),
        "updated_at": str(d.get("updated_at", "") or ""),
        "history": history,
    }


def _parse(raw: list | None) -> dict[str, dict]:
    voices: dict[str, dict] = {}
    for d in raw or []:
        _parse_entry(d, voices)
    return voices


def load_voices(force: bool = False) -> dict[str, dict]:
    """mtime-cached registry (voice edits take effect without restart)."""
    global _CACHE
    p = _registry_path()
    with _LOCK:
        mtime = p.stat().st_mtime_ns if p.exists() else 0
        if not force and _CACHE is not None and _CACHE[0] == mtime:
            return _CACHE[1]
        raw: list | None = None
        if p.exists():
            try:
                import yaml
            except ImportError:
                raise VoiceError("PyYAML not installed: cannot read voices.yaml", 500)
            raw = yaml.safe_load(p.read_text())
        voices = _parse(raw)
        _CACHE = (mtime, voices)
        return voices


def get_voice(voice_id: str) -> dict | None:
    return load_voices().get(voice_id)


def resolve_reference(v: dict) -> str | None:
    """Filesystem path of the reference take (asset id resolved by caller)."""
    return v["path"]


def public_view(voices: dict[str, dict]) -> list[dict]:
    """API projection: no filesystem paths / transcripts leakage."""
    return [{"id": v["id"], "name": v["name"], "description": v["description"],
             "source": "asset" if v["asset"] else "file",
             "kind": v["kind"], "role": v["role"], "tags": v["tags"],
             "license": v["license"], "model": v["model"],
             "version": v["version"], "updated_at": v["updated_at"],
             "revisions": len(v["history"])}
            for v in voices.values()]


def detail_view(v: dict) -> dict:
    """GET /v1/voices/{id}: everything safe — still no paths or transcripts."""
    return {
        "id": v["id"], "name": v["name"], "description": v["description"],
        "source": "asset" if v["asset"] else "file",
        "asset": v["asset"], "kind": v["kind"], "role": v["role"],
        "tags": v["tags"], "license": v["license"], "provenance": v["provenance"],
        "model": v["model"], "version": v["version"],
        "created_at": v["created_at"], "updated_at": v["updated_at"],
        "transcript_chars": len(v["prompt_text"]),
        "revisions": [{k: h.get(k) for k in ("version", "at", "name", "description")}
                      for h in v["history"]],
    }


def _read_raw() -> list:
    import yaml
    p = _registry_path()
    return yaml.safe_load(p.read_text()) if p.exists() else []


def _write_raw(entries: list) -> None:
    import yaml
    p = _registry_path()
    tmp = p.with_name(f".voices.{os.getpid()}.tmp")
    tmp.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False))
    os.replace(tmp, p)
    global _CACHE
    _CACHE = None


def register_voice(entry: dict, replace: bool = False, note: str = "") -> tuple[dict, bool]:
    """Register a voice. Returns (entry, was_replaced).

    wants.md §2: 重复注册有明确处理规则——默认 409，显式 replace=true 才
    覆盖，且旧版本进 history（版本自动 +1）。"""
    with _LOCK:
        raw = _read_raw()
        vid = str(entry.get("id", "")).strip()
        existing = next((d for d in raw if isinstance(d, dict) and d.get("id") == vid), None)
        created = existing is None
        if existing is not None and not replace:
            raise VoiceError(
                f"voice {vid!r} already exists (version {int(existing.get('version', 1))}); "
                "resubmit with replace=true to swap it (old revision kept in history)", 409)
        cleaned = {k: v for k, v in entry.items()
                   if v not in (None, "") and k not in ("version", "created_at",
                                                         "updated_at", "history")}
        now = _now()
        new_entry = dict(cleaned)
        new_entry["version"] = (int(existing.get("version", 1)) + 1) if existing else 1
        new_entry["created_at"] = (existing or {}).get("created_at") or now
        new_entry["updated_at"] = now
        hist = list((existing or {}).get("history") or [])
        if existing is not None:
            snap = _rev_snapshot(existing)
            if note:
                snap["note"] = note
            hist.append(snap)
        new_entry["history"] = hist[-_HISTORY_KEEP:]
        merged = [d for d in raw if not (isinstance(d, dict) and d.get("id") == vid)]
        merged.append(new_entry)
        _parse(merged)  # validates the WHOLE merged list, not just the new entry
        _write_raw(merged)
        return _parse(_read_raw())[vid], created


def delete_voice(voice_id: str) -> dict:
    """Remove an entry (404 if unknown). Generated audio/assets already produced
    with it are untouched; future jobs referencing it fail fast at submit."""
    with _LOCK:
        raw = _read_raw()
        remaining = [d for d in raw if not (isinstance(d, dict) and d.get("id") == voice_id)]
        if len(remaining) == len(raw):
            raise VoiceError(f"voice {voice_id!r} not found", 404)
        _parse(remaining)
        _write_raw(remaining)
        return next(d for d in raw if d.get("id") == voice_id)


def upsert_voice(entry: dict) -> dict:
    """Back-compat helper (tests/tools): force-replace semantics."""
    v, _ = register_voice(entry, replace=True)
    return v


def file_sig(path: str) -> str:
    """cheap content signature for worker-side cache invalidation (mtime+size)."""
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns:x}-{st.st_size:x}"
    except OSError:
        return "missing"
