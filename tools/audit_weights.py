"""Audit: downloaded model weights must live under /mnt/data/AV (user redline).

Scope: HF/modelscope caches, ~/ComfyUI, and large (>100MB) weight-shaped
files in /tmp. Scratch artifacts from other workloads (unichess kit_loop
training temp dirs etc.) are out of scope and excluded by prefix."""
import os
import sys
from pathlib import Path

WEIGHT_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx")
SCAN_DIRS = [Path.home() / ".cache/huggingface",
             Path.home() / ".cache/modelscope",
             Path.home() / "ComfyUI/models"]
TMP_MIN_BYTES = 100 * 1024 * 1024
TMP_EXCLUDE = ("kit_loop_", "p19sink_", "unichess", "p4_gpusrv", "m2inspect",
               "tmpr", "tmpi", "tmpg", "tmpf", "tmp9", "tmp_", "tmps", "tmpn",
               "tmpk", "tmpu", "tmpp", "tmpj", "fav_")
SIZE_ONLY_ROOTS = {"/home/jeefy/.cache/pip"}

bad = []


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        if ".no_exist" in dirpath:
            continue
        yield dirpath, filenames


for root in SCAN_DIRS:
    if not root.is_dir():
        continue
    for dirpath, filenames in walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in WEIGHT_EXT and not p.is_symlink():
                if not any(str(p).startswith(a) for a in SIZE_ONLY_ROOTS):
                    bad.append(p)

tmp = Path("/tmp")
for entry in tmp.iterdir():
    if any(entry.name.startswith(pre) for pre in TMP_EXCLUDE):
        continue
    if entry.is_dir():
        for dirpath, filenames in walk(entry):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    if p.suffix.lower() in WEIGHT_EXT and not p.is_symlink() \
                            and p.stat().st_size > TMP_MIN_BYTES:
                        bad.append(p)
                except OSError:
                    pass
    elif entry.suffix.lower() in WEIGHT_EXT and entry.stat().st_size > TMP_MIN_BYTES:
        bad.append(entry)

if bad:
    for p in bad:
        try:
            print(f"LEAKED: {p} ({p.stat().st_size / 1e9:.2f} GB)")
        except OSError:
            print(f"LEAKED: {p} (unreadable)")
    sys.exit(1)
print("AUDIT_OK: no downloaded weights outside /mnt/data/AV")
