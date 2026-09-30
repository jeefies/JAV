"""Weight downloader with redline compliance (JAV-DESIGN 11.2):

  - source: hf-mirror; downloads land in staging on /mnt/data, verified
    (size + streamed sha256 vs HF LFS oid), then moved into ComfyUI/models
    category dirs; registered in data/weights.json
  - refuses to start if free disk < required bytes + margin
  - resumable: files already at destination with matching sha are skipped
  - partial downloads (.incomplete) live in staging, never /tmp

usage:
  python3 tools/download_weights.py --repo Lightricks/LTX-2.5 \
      --match ltx-2.5-22b-distilled-transformer-comfy-int8-convrot \
      --match gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot \
      --match ltx-2.5-audio-vae-bf16 --match ltx-2.5-video-vae-bf16
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/mnt/data/AV/JAV")

from jav import config  # noqa: E402

MODELS = config.COMFYUI_DIR / "models"
STAGING = config.DATA_DIR / "staging"
REGISTRY = config.DATA_DIR / "weights.json"
MARGIN_BYTES = 20 * 1024**3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_manifest(repo: str, matches: list[str]) -> list[dict]:
    import requests
    url = f"https://hf-mirror.com/api/models/{repo}/tree/main?recursive=true"
    r = requests.get(url, timeout=30, headers={"User-Agent": "jav-weight-sync/1.0"})
    r.raise_for_status()
    data = r.json()
    picked = []
    for f in data:
        if f["type"] != "file":
            continue
        if not any(m in f["path"] for m in matches):
            continue
        lfs = f.get("lfs", {})
        if not lfs.get("size"):
            continue
        picked.append({"path": f["path"], "size": lfs["size"],
                       "sha256": lfs.get("oid", "")})
    return picked


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--match", action="append", required=True)
    args = ap.parse_args()

    manifest = fetch_manifest(args.repo, args.match)
    if not manifest:
        log("no files matched"); sys.exit(1)
    required = sum(f["size"] for f in manifest)
    log(f"{len(manifest)} files, {required/1e9:.1f} GB total")

    free = shutil.disk_usage("/mnt/data").free
    if free < required + MARGIN_BYTES:
        log(f"ABORT: free {free/1e9:.1f} GB < required {required/1e9:.1f} GB + 20 GB margin "
            f"— consult JAV-DESIGN.md 11.3 deletion list")
        sys.exit(2)

    from huggingface_hub import hf_hub_download
    STAGING.mkdir(parents=True, exist_ok=True)
    registry = load_registry()

    for f in manifest:
        dest = MODELS / f["path"]
        if dest.exists():
            if f["sha256"] and sha256_file(dest) == f["sha256"]:
                log(f"skip (verified at destination): {f['path']}")
                registry[str(dest)] = f | {"ts": time.strftime("%F %T")}
                continue
            log(f"destination exists but sha mismatch — redownloading: {f['path']}")
            dest.unlink()
        log(f"downloading {f['path']} ({f['size']/1e9:.2f} GB)...")
        t0 = time.time()
        local = hf_hub_download(repo_id=args.repo, filename=f["path"],
                                local_dir=str(STAGING),
                                endpoint="https://hf-mirror.com")
        local = Path(local)
        got = local.stat().st_size
        if got != f["size"]:
            local.unlink(missing_ok=True)
            log(f"FAIL size {got} != {f['size']}")
            sys.exit(3)
        if f["sha256"]:
            h = sha256_file(local)
            if h != f["sha256"]:
                local.unlink(missing_ok=True)
                log(f"FAIL sha {h[:12]} != {f['sha256'][:12]}")
                sys.exit(3)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local), dest)
        dt = time.time() - t0
        registry[str(dest)] = f | {"repo": args.repo, "ts": time.strftime("%F %T")}
        log(f"OK {dest.name} ({got/1e9:.2f} GB, {dt:.0f}s, {got/1e6/max(dt,1):.0f} MB/s)")

    REGISTRY.write_text(json.dumps(registry, indent=1))
    log("DOWNLOAD_ALL_DONE")
    # clean staging leftovers (.incomplete partials etc.)
    for p in STAGING.rglob("*"):
        if p.is_file() and (p.name.endswith(".incomplete") or ".no_exist" in str(p)):
            p.unlink(missing_ok=True)
    log("DOWNLOAD_COMPLETE")


if __name__ == "__main__":
    main()
