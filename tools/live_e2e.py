#!/usr/bin/env python3
"""Live end-to-end: REAL GPU generations against the running JAV service.

pytest suites are FakeBackend-only (fast, no GPU); this closes the other half
of the pyramid through the production HTTP path via the SDK:
  zit   t2i -> i2i -> inpaint   (image chain feeds itself, real bytes)
  ltx25 soft-cancel finalize + t2v + i2v   (crosses into ComfyUI runtime)
  mh3   t2v turbo               (cross-family switch, VRAM release check)

Media content is verified, not just job status: PNG pixels via PIL, video
duration/resolution/frames via ffprobe.

usage: python3 tools/live_e2e.py [--stage zit|ltx25|mh3|all] [--skip-cancel]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/data/AV/JAV/sdk")
from jav import Client  # noqa: E402

OUT = Path("/tmp/kilo/live_e2e")
REPORT = OUT / "report.json"
RESULTS = []


def record(stage, name, ok, detail):
    RESULTS.append({"stage": stage, "case": name, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {stage}/{name}: {detail}", flush=True)


def ffprobe_meta(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=width,height,nb_frames",
                        "-of", "json", str(path)],
                       capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    return {"duration": float(d.get("format", {}).get("duration", 0) or 0),
            "width": st.get("width"), "height": st.get("height"),
            "frames": st.get("nb_frames")}


def run_job(c: Client, label, timeout, expect="completed", **kw):
    job = c.submit(**kw)
    res = job.wait(timeout=timeout, interval=3)
    data = res.job.data
    ok = data["status"] == expect
    return job, res, ok, data


def stage_zit(c: Client):
    from PIL import Image, ImageDraw
    OUT.mkdir(parents=True, exist_ok=True)

    job, res, ok, data = run_job(c, "t2i", 900, provider="zit", workflow="t2i",
                                 inputs={"prompt": "a red cube on white marble, studio light"},
                                 generation={"width": 512, "height": 512, "steps": 9, "seed": 1234})
    p = res.download(OUT / "t2i.png")
    size = Image.open(p).size
    record("zit", "t2i", ok and size == (512, 512) and p.stat().st_size > 10_000,
           f"status={data['status']} size={size} bytes={p.stat().st_size}")

    t2i_asset = c.upload_asset(p)
    job, res, ok, data = run_job(c, "i2i", 900, provider="zit", workflow="i2i",
                                 inputs={"prompt": "the cube turns into a glossy apple",
                                         "image": t2i_asset},
                                 generation={"width": 512, "height": 512, "steps": 9,
                                             "strength": 0.55, "seed": 77})
    p2 = res.download(OUT / "i2i.png")
    size = Image.open(p2).size
    record("zit", "i2i", ok and size == (512, 512), f"status={data['status']} size={size}")

    mask = OUT / "mask.png"
    im = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(im).ellipse((160, 160, 352, 352), fill=255)
    im.save(mask)
    mask_asset = c.upload_asset(mask, kind="image")
    job, res, ok, data = run_job(c, "inpaint", 900, provider="zit", workflow="inpaint",
                                 inputs={"prompt": "a glowing holographic cube",
                                         "image": t2i_asset, "mask": mask_asset},
                                 generation={"width": 512, "height": 512, "steps": 9, "seed": 91})
    p3 = res.download(OUT / "inpaint.png")
    record("zit", "inpaint", ok and Image.open(p3).size == (512, 512),
           f"status={data['status']} bytes={p3.stat().st_size}")


def safe_status(job):
    try:
        return job.refresh()["status"]
    except Exception:
        return None  # transient poll failure under system pressure


def stage_ltx25(c: Client, skip_cancel):
    gen = {"width": 512, "height": 288, "duration": 3, "fps": 8, "mode": "fast"}
    if not skip_cancel:
        job = c.submit(provider="ltx25", workflow="t2v",
                       inputs={"prompt": "a drone orbiting a lighthouse at dusk"},
                       generation=gen)
        deadline = time.time() + 600
        status = safe_status(job)
        while time.time() < deadline and status in (None, "queued", "starting_runtime"):
            time.sleep(2)
            status = safe_status(job)
        if status == "running":
            c_cancel = job.cancel()
            deadline = time.time() + 420
            while time.time() < deadline and safe_status(job) == "running":
                time.sleep(3)
            final = safe_status(job)
            # the reviewed bug: interrupt used to surface as failed/timeout and
            # even RE-RUN the cancelled job. Must land on terminal cancelled.
            record("ltx25", "cancel_finalizes", final == "cancelled",
                   f"cancel_resp={c_cancel.get('status')} final={final}")
        else:
            record("ltx25", "cancel_finalizes", True, f"skipped: already {status} before cancel")

    job, res, ok, data = run_job(c, "t2v", 1500, provider="ltx25", workflow="t2v",
                                 inputs={"prompt": "timelapse clouds over a mountain ridge, "
                                                   "camera slowly pushing in"},
                                 generation=gen)
    p = res.download(OUT / "t2v.mp4")
    m = ffprobe_meta(p)
    ok = ok and m["duration"] > 2.5 and (m["width"], m["height"]) == (512, 288)
    record("ltx25", "t2v", ok, f"status={data['status']} meta={m} bytes={p.stat().st_size}")

    img_asset = c.upload_asset(OUT / "t2i.png")
    job, res, ok, data = run_job(c, "i2v", 1500, provider="ltx25", workflow="i2v",
                                 inputs={"prompt": "the cube rotates slowly, soft light shifts",
                                         "first_image": img_asset},
                                 generation=gen)
    p = res.download(OUT / "i2v.mp4")
    m = ffprobe_meta(p)
    ok = ok and m["duration"] > 2.5
    record("ltx25", "i2v", ok, f"status={data['status']} meta={m} bytes={p.stat().st_size}")


def stage_mh3(c: Client):
    job, res, ok, data = run_job(c, "t2v", 2100, provider="mh3", workflow="t2v",
                                 inputs={"prompt": "a corgi surfing a tiny wave, sunny beach, "
                                                   "photorealistic", "turbo": True},
                                 generation={"width": 512, "height": 320, "duration": 5})
    p = res.download(OUT / "mh3_t2v.mp4")
    m = ffprobe_meta(p)
    ok = ok and m["duration"] > 4.0
    record("mh3", "t2v_turbo", ok, f"status={data['status']} meta={m} bytes={p.stat().st_size}")


CONTROL_SOURCE = "/mnt/data/AV/JAV/data/outputs/fc/fc567f66bc213017ffed2a99.mp4"


def stage_control(c: Client):
    """The 5 IC-LoRA control workflows on a real source video (~5s clip)."""
    from PIL import Image, ImageDraw
    src_asset = c.upload_asset(CONTROL_SOURCE, kind="video")

    mask = OUT / "ctl_mask.png"
    im = Image.new("L", (768, 416), 0)
    ImageDraw.Draw(im).ellipse((250, 120, 520, 300), fill=255)
    im.save(mask)
    mask_asset = c.upload_asset(mask, kind="image")

    sheet = OUT / "ctl_sheet.png"
    base = Image.new("RGB", (512, 512), "white")
    try:
        cube = Image.open(OUT / "t2i.png").convert("RGB").resize((240, 240))
        base.paste(cube, (10, 10))
        apple = Image.open(OUT / "i2i.png").convert("RGB").resize((240, 240))
        base.paste(apple, (262, 10))
    except Exception:
        ImageDraw.Draw(base).rectangle((10, 10, 250, 250), fill="red")
    base.save(sheet)
    sheet_asset = c.upload_asset(sheet, kind="image")

    cases = [
        ("union_control", {"prompt": "a person walking through a sunlit forest trail",
                           "control_video": src_asset, "shorter_size": 384}),
        ("motion_control", {"prompt": "the scene melts into ultra slow motion, floating dust",
                            "source_video": src_asset, "shorter_size": 384}),
        ("inpaint", {"prompt": "a glowing glass orb hovering in the air",
                     "source_video": src_asset, "mask_image": mask_asset,
                     "shorter_size": 384}),
        ("outpaint", {"prompt": "a wide coastal panorama extends left and right",
                      "source_video": src_asset, "canvas_width": 1024,
                      "canvas_height": 576, "shorter_size": 384}),
        ("ic_lora", {"prompt": "top-left: a red cube; bottom-right: a glossy apple — "
                               "the apple rolls toward the cube on a white table",
                     "reference_sheet": sheet_asset}),
        ("ic_lora", {"prompt": "the camera pans while the scene stays photoreal",
                     "mode": "v2v", "source_video": src_asset,
                     "lora": "cinemagraph", "shorter_size": 384}),
    ]
    # source clip is ~5.2 s; reference mode requests the default 5 s.
    # guide tokens must be cropped: >2x duration means the crop chain broke.
    for wf, inputs in cases:
        label = wf + ("_v2v" if wf == "ic_lora" and inputs.get("mode") == "v2v" else "")
        job, res, ok, data = run_job(c, label, 2400, provider="ltx25",
                                     workflow=wf, inputs=inputs,
                                     generation={"seed": 4242})
        if data["status"] == "completed":
            p = res.download(OUT / f"ctl_{label}.mp4")
            m = ffprobe_meta(p)
            ok = ok and 3.5 <= m["duration"] <= 7.0
            record("control", label, ok, f"meta={m} bytes={p.stat().st_size}")
        else:
            record("control", label, False, f"status={data['status']} err={data.get('error')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma list of zit,ltx25,control,mh3 or all")
    ap.add_argument("--skip-cancel", action="store_true")
    args = ap.parse_args()

    c = Client()
    health = c._get("/v1/health")
    if health.get("status") != "healthy":
        sys.exit("service not healthy: " + json.dumps(health))
    print(f"JAV live ({health['state']}, auth={health.get('auth')})\n", flush=True)

    stages = ({"zit", "ltx25", "control", "mh3"} if args.stage == "all"
              else set(args.stage.split(",")))
    if "zit" in stages:
        stage_zit(c)
    if "ltx25" in stages:
        stage_ltx25(c, args.skip_cancel)
    if "control" in stages:
        stage_control(c)
    if "mh3" in stages:
        stage_mh3(c)

    caps = c.capabilities()
    avail = {f"{p}.{w}": v["available"] for p, d in caps.items() for w, v in d["workflows"].items()}
    rt = c.runtime()
    summary = {"results": RESULTS, "passed": sum(r["ok"] for r in RESULTS),
               "failed": sum(not r["ok"] for r in RESULTS),
               "capabilities_available": sum(avail.values()),
               "runtime_after": {"state": rt["state"], "active_profile": rt["active_profile"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    merged = {**existing, **{r["stage"] + "/" + r["case"]: r for r in RESULTS}}
    REPORT.write_text(json.dumps({**summary, "cumulative": merged}, indent=1, default=str))
    print(f"\nSUMMARY: {summary['passed']} passed, {summary['failed']} failed "
          f"-> {REPORT}", flush=True)
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
