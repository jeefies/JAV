"""Asset API: content-addressed uploads (JAV-DESIGN 6).

Accepts raw-body PUT/POST bytes (SDK-friendly) or multipart form.
Uploads are streamed to disk with an incremental byte cap + SHA256 (never
buffered whole on the event loop)."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .. import config
from ..models import EXT_KIND

router = APIRouter(prefix="/v1")

MAX_ASSET_BYTES = 2 * 1024**3
CHUNK = 1024 * 1024


async def _spool(request: Request | None, upload: UploadFile | None) -> tuple[Path, int]:
    """Stream the request body to a temp file; enforce the 2GiB cap during
    the transfer, not after buffering it."""
    tmp_dir = config.ASSETS_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}.part"
    total = 0
    h = hashlib.sha256()
    try:
        if upload is not None:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ASSET_BYTES:
                    raise HTTPException(413, detail="asset exceeds 2GiB")
                h.update(chunk)
                await _write(tmp, chunk)
        else:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_ASSET_BYTES:
                    raise HTTPException(413, detail="asset exceeds 2GiB")
                h.update(chunk)
                await _write(tmp, chunk)
        if total == 0:
            raise HTTPException(400, detail="empty body")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, total


async def _write(tmp: Path, chunk: bytes):
    with open(tmp, "ab") as fh:
        fh.write(chunk)


async def _ingest_request(request: Request, upload: UploadFile | None,
                          kind: str | None, filename: str | None,
                          mime: str | None, ctx) -> dict:
    if request is not None:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_ASSET_BYTES:
            raise HTTPException(413, detail="asset exceeds 2GiB")
    tmp, total = await _spool(request, upload)
    with open(tmp, "rb") as fh:
        sha_hex = hashlib.file_digest(fh, "sha256").hexdigest()
    existing = ctx.store.get_asset(f"asset_{sha_hex[:16]}")
    if existing:
        tmp.unlink(missing_ok=True)
        return existing
    if not kind:
        ext = Path(filename or "").suffix.lower()
        kind = EXT_KIND.get(ext) or ("image" if (mime or "").startswith("image/") else
                                     "video" if (mime or "").startswith("video/") else
                                     "audio" if (mime or "").startswith("audio/") else None)
    if not kind:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, detail="cannot infer asset kind; pass ?kind=")
    ext = Path(filename).suffix.lower() if filename else {
        "image": ".png", "video": ".mp4", "audio": ".mp3"}.get(kind, ".bin")
    dest_dir = config.ASSETS_DIR / sha_hex[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sha_hex[:24]}{ext}"
    if dest.exists():
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, dest)
    return ctx.store.put_asset(sha256=sha_hex, kind=kind, path=str(dest),
                               size=total, mime=mime,
                               asset_id=f"asset_{sha_hex[:16]}")


@router.post("/assets", status_code=201)
async def upload_asset_raw(request: Request, kind: str | None = None):
    """raw-body upload (SDK/curl friendly)"""
    ctx = request.app.ctx
    asset = await _ingest_request(request, None, kind,
                                  request.headers.get("x-filename"),
                                  request.headers.get("content-type"), ctx)
    return {"id": asset["id"], "type": asset["kind"], "sha256": asset["sha256"],
            "size": asset["size"], "created_at": asset["created_at"]}


@router.post("/assets/upload", status_code=201)
async def upload_asset_multipart(request: Request, kind: str | None = None,
                                 file: UploadFile = File(...)):
    ctx = request.app.ctx
    asset = await _ingest_request(None, file, kind, file.filename,
                                  file.content_type, ctx)
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
    refs = ctx.store.asset_references(asset_id)
    if refs["active_jobs"] or refs["outputs"]:
        raise HTTPException(409, detail=(
            f"asset in use: {refs['active_jobs']} non-terminal jobs, "
            f"{refs['outputs']} recorded outputs"))
    try:
        Path(asset["path"]).unlink(missing_ok=True)
    except PermissionError:
        pass
    ctx.store._exec("DELETE FROM assets WHERE id=?", (asset_id,))
    return {"id": asset_id, "deleted": True}
