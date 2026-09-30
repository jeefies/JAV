"""One-shot integrity verification for every registered weight file.

Per data/weights.json entry: exists -> exact size -> sha256 ->
safetensors header parses. Exit nonzero on any failure; prints a repair
command per bad file (delete + re-run download_weights for that match).

usage: python3 tools/verify_weights.py [--no-hash]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REGISTRY = Path("/mnt/data/AV/JAV/data/weights.json")


def sha256_stream(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 streaming (existence + size + header only)")
    args = ap.parse_args()
    reg = json.loads(REGISTRY.read_text())
    fails = []
    for path_s, meta in sorted(reg.items()):
        p = Path(path_s)
        name = p.name[:58]
        if not p.exists():
            print(f"MISSING  {name}")
            fails.append((path_s, meta, "missing"))
            continue
        size = p.stat().st_size
        if size != meta["size"]:
            print(f"SIZEBAD  {name} ({size} != {meta['size']})")
            fails.append((path_s, meta, "size"))
            continue
        if not args.no_hash:
            if sha256_stream(p) != meta["sha256"]:
                print(f"HASHBAD  {name}")
                fails.append((path_s, meta, "sha256"))
                continue
        if p.suffix == ".safetensors":
            try:
                from safetensors import safe_open
                with safe_open(p, framework="pt") as f:
                    n = len(f.keys())
                print(f"OK       {name} {size/1e9:7.2f}G  {n} tensors")
            except Exception as e:
                print(f"HDRFAIL  {name}: {str(e)[:80]}")
                fails.append((path_s, meta, "header"))
        else:
            print(f"OK       {name} {size/1e9:7.2f}G")
    total = sum(v["size"] for v in reg.values())
    print(f"---\n{len(reg)} files, {total/1e9:.1f} GB, failures: {len(fails)}")
    for path_s, meta, why in fails:
        stem = Path(path_s).stem
        print(f"REPAIR[{why}]: rm -f '{path_s}' && "
              f"python3 /mnt/data/AV/JAV/tools/download_weights.py "
              f"--repo {meta.get('repo','?')} --match {stem}")
    if fails:
        sys.exit(1)
    print("VERIFY_ALL_OK")


if __name__ == "__main__":
    main()
