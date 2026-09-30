"""Asset API: content-addressed uploads (JAV-DESIGN 6).

Accepts raw-body PUT/POST bytes (SDK-friendly) or multipart form."""
from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .. import config
from ..store import now

router = APIRouter(prefix="/v1")

EXT_KIND = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
            ".bmp": "image", ".gif": "image", ".mp4": "video", ".mov": "video",
            ".webm": "video", ".mkv": "video", ".mp3": "audio", ".wav": "audio",
            ".flac": "audio", ".ogg": "audio"}


async def _ingest(data: bytes, kind: str | None, filename: str | None,
                 mime: str | None, ctx) -> dict:
    if not data:
        raise HTTPException(400, detail="empty body")
    if len(data) > 2 * 1024**3:
        raise HTTPException(413, detail="asset exceeds 2GiB")
    sha = hashlib.sha256(data).hexdigest()
    existing = ctx.store.get_asset(f"asset_{sha[:16]}")
    if existing:
        return existing
    if not kind:
        ext = Path(filename or "").suffix.lower()
        kind = EXT_KIND.get(ext) or ("image" if (mime or "").startswith("image/") else
                                     "video" if (mime or "").startswith("video/") else
                                     "audio" if (mime or "").startswith("audio/") else None)
    if not kind:
        raise HTTPException(400, detail="cannot infer asset kind; pass ?kind=")
    ext = Path(filename).suffix.lower() if filename else {
        "image": ".png", "video": ".mp4", "audio": ".mp3"}.get(kind, ".bin")
    dest_dir = config.ASSETS_DIR / sha[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sha[:24]}{ext}"
    if not dest.exists():
        dest.write_bytes(data)
    return ctx.store.put_asset(sha256=sha, kind=kind, path=str(dest),
                               size=len(data), mime=mime,
                               asset_id=f"asset_{sha[:16]}")


@router.post("/assets", status_code=201)
async def upload_asset_raw(request: Request, kind: str | None = None):
    """raw-body upload (SDK/curl friendly)"""
    ctx = request.app.ctx
    data = await request.body()
    ctype = request.headers.get("content-type")
    asset = await _ingest(data, kind, request.headers.get("x-filename"), ctype, ctx)
    return {"id": asset["id"], "type": asset["kind"], "sha256": asset["sha256"],
            "size": asset["size"], "created_at": asset["created_at"]}


@router.post("/assets/upload", status_code=201)
async def upload_asset_multipart(request: Request, kind: str | None = None,
                                 file: UploadFile = File(...)):
    ctx = request.app.ctx
    data = await file.read()
    asset = await _ingest(data, kind, file.filename, file.content_type, ctx)
    return {"id": asset["id"], "type": asset["kind"], "sha256": asset["sha256"],
            "size": asset["size"], "created_at": asset["created_at"]}


@router.get("/assets/{asset_id}")
async def get_asset(asset_id: str, request: Request):
    ctx = request.app.ctx
    asset = ctx.store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, detail="asset not found")
    if not Path(asset["path"]).exists():
        raise HTTPException(410, detail="asset file missing")
    return FileResponse(asset["path"], media_type=asset.get("mime") or "application/octet-stream",
                        filename=Path(asset["path"]).name)


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, request: Request):
    ctx = request.app.ctx
    asset = ctx.store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, detail="asset not found")
    try:
        Path(asset["path"]).unlink(missing_ok=True)
    except PermissionError:
        pass
    ctx.store._exec("DELETE FROM assets WHERE id=?", (asset_id,))
    return {"id": asset_id, "deleted": True}
