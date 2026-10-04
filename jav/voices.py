"""Voice registry for cosyvoice.t2a (配音角色固定音色).

config/voices.yaml maps stable per-character voice ids to a reference dry
voice (asset id or a file path under /mnt/data/AV) + its exact transcript,
which CosyVoice3 turns into a zero-shot speaker embedding. 与 profiles.yaml
一样：字段级、mtime 缓存、未知 key 直接报错。
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from . import config

VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
# 干声参考只允许这两个来源：JAV 资产库，或 AV 数据盘内的既有文件（红线同源）
_ALLOWED_PATH_ROOTS = (config.DATA_DIR.resolve(), Path("/mnt/data/AV").resolve())


class VoiceError(ValueError):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _registry_path() -> Path:
    return config.BASE_DIR / "config" / "voices.yaml"


_CACHE: tuple[int, dict] | None = None  # (mtime_ns, voices)
_LOCK = threading.Lock()


def _parse(raw: list | None) -> dict[str, dict]:
    voices: dict[str, dict] = {}
    for d in raw or []:
        if not isinstance(d, dict):
            raise VoiceError(f"voices.yaml: entry must be a mapping, got {type(d).__name__}")
        known = {"id", "name", "prompt_text", "asset", "path", "description"}
        unknown = set(d) - known
        if unknown:
            raise VoiceError(f"voices.yaml: unknown key(s) {sorted(unknown)} "
                             f"in voice {d.get('id')!r}")
        vid = str(d.get("id", "")).strip()
        if not VOICE_ID_RE.match(vid):
            raise VoiceError(f"voices.yaml: bad voice id {vid!r} "
                             "(want [a-z0-9][a-z0-9_-] up to 32 chars)")
        if vid in voices:
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
            rp = Path(path)
            if not rp.is_absolute() or not any(
                    rp == r or rp.is_relative_to(r) for r in _ALLOWED_PATH_ROOTS):
                raise VoiceError(f"voices.yaml: voice {vid!r} path must be an absolute "
                                 f"file under /mnt/data/AV: {path!r}")
        voices[vid] = {
            "id": vid,
            "name": str(d.get("name", "")).strip() or vid,
            "prompt_text": text,
            "asset": asset or None,
            "path": path or None,
            "description": str(d.get("description", "")).strip(),
        }
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


def public_view(voices: dict[str, dict]) -> list[dict]:
    """API projection: no filesystem paths / transcripts leakage."""
    return [{"id": v["id"], "name": v["name"], "description": v["description"],
             "source": "asset" if v["asset"] else "file"}
            for v in voices.values()]


def upsert_voice(entry: dict) -> dict:
    """Register/replace a voice entry atomically (used by POST /v1/voices)."""
    p = _registry_path()
    with _LOCK:
        try:
            import yaml
        except ImportError:
            raise VoiceError("PyYAML not installed: cannot write voices.yaml", 500)
        raw = yaml.safe_load(p.read_text()) if p.exists() else []
        # validate the merged list, not just the incoming entry
        cleaned = {k: v for k, v in entry.items() if v not in (None, "")}
        merged = [d for d in (raw or []) if d.get("id") != cleaned.get("id")]
        merged.append(cleaned)
        _parse(merged)  # raises VoiceError on any invalidity
        tmp = p.with_name(f".voices.{os.getpid()}.tmp")
        tmp.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False))
        os.replace(tmp, p)
        global _CACHE
        _CACHE = None
    return _parse(yaml.safe_load(p.read_text()))[cleaned["id"]]
