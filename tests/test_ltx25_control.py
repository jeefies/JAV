"""LTX-2.5 IC-LoRA control workflows: normalize/compile/validation logic."""
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/AV/JAV")

import pytest

from jav import config
from jav.models import ProviderError
from jav.providers import ltx25


@pytest.fixture(autouse=True)
def real_workflows(monkeypatch):
    # conftest relocates BASE_DIR to a tmp tree; template compilation must
    # read the real jav/workflows/ graphs
    monkeypatch.setattr(config, "BASE_DIR", Path("/mnt/data/AV/JAV"))


def compile_ok(workflow, inputs, generation=None):
    p = ltx25.normalize(workflow, inputs, generation or {})
    t = ltx25.compile(p, {k: f"/tmp/{k}.bin" for k in p["assets"]}, "/tmp", "/base")
    return p, t["graph"]


def test_union_control_compile():
    _, g = compile_ok("union_control", {"prompt": "x", "control_video": "asset_u"})
    assert g["30"]["class_type"] == "LTXICLoRALoaderModelOnly"
    assert g["30"]["inputs"]["lora_name"] == \
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    assert g["35"]["inputs"]["resize_type"] == "scale to multiple"
    assert g["35"]["inputs"]["resize_type.multiple"] == 64  # factor-2 lora
    assert g["12"]["inputs"]["model"] == ["30", 0]
    assert g["38"]["inputs"]["latent_downscale_factor"] == ["30", 1]
    assert g["31"]["inputs"]["file"] == "asset:control_video"
    assert g["8"]["inputs"]["width"] == ["37", 0]  # canvas follows source video


def test_motion_control_uses_slow_motion_lora():
    _, g = compile_ok("motion_control", {"prompt": "x", "source_video": "asset_s"})
    assert g["30"]["inputs"]["lora_name"].startswith("ltx-2.5-22b-lora-slow-motion")
    assert g["35"]["inputs"]["resize_type.multiple"] == 32
    assert "33" not in g  # no annotator in V2V path


def test_inpaint_mask_chain():
    _, g = compile_ok("inpaint", {"prompt": "x", "source_video": "a",
                                  "mask_image": "m", "dilate_radius": 9})
    assert g["42"]["inputs"]["spatial_radius"] == 9
    assert g["36"]["inputs"]["images"] == ["35", 0]
    assert g["36"]["inputs"]["mask"] == ["42", 0]
    assert g["38"]["inputs"]["image"] == ["36", 0]


def test_inpaint_rejects_missing_mask():
    with pytest.raises(ProviderError, match="mask_image"):
        ltx25.normalize("inpaint", {"prompt": "x", "source_video": "a"}, {})


def test_outpaint_canvas_validation():
    with pytest.raises(ProviderError, match="canvas_width"):
        ltx25.normalize("outpaint", {"prompt": "x", "source_video": "a",
                                     "canvas_width": 1000, "canvas_height": 576}, {})
    _, g = compile_ok("outpaint", {"prompt": "x", "source_video": "a",
                                   "canvas_width": 1024, "canvas_height": 576})
    assert g["43"]["inputs"]["target_width"] == 1024


def test_ic_lora_modes():
    _, g_ref = compile_ok("ic_lora", {"prompt": "x", "reference_sheet": "s"})
    assert g_ref["32"]["class_type"] == "RepeatImageBatch"
    # reference mode has no source video: length/fps are user params
    assert g_ref["32"]["inputs"]["amount"] == 121          # 5 s @ 24 fps
    assert g_ref["8"]["inputs"]["length"] == 121
    assert g_ref["19"]["inputs"]["fps"] == 24
    assert g_ref["7"]["inputs"]["frame_rate"] == 24
    _, g_v2v = compile_ok("ic_lora", {"prompt": "x", "mode": "v2v",
                                      "source_video": "s", "lora": "clean_plate"})
    assert g_v2v["30"]["inputs"]["lora_name"].endswith("clean-plate-1.0.safetensors")
    assert g_v2v["31"]["class_type"] == "LoadVideo"


def test_ic_lora_lora_allowlist():
    with pytest.raises(ProviderError, match="not in"):
        ltx25.normalize("ic_lora", {"prompt": "x", "reference_sheet": "s",
                                    "lora": "union"}, {})
    with pytest.raises(ProviderError, match="not in"):
        # cinemagraph is a V2V edit lora, not valid for reference-sheet mode
        ltx25.normalize("ic_lora", {"prompt": "x", "reference_sheet": "s",
                                    "lora": "cinemagraph"}, {})
    with pytest.raises(ProviderError, match="mode"):
        ltx25.normalize("ic_lora", {"prompt": "x", "mode": "weird",
                                    "reference_sheet": "s"}, {})


def test_control_param_bounds():
    with pytest.raises(ProviderError, match="shorter_size"):
        ltx25.normalize("motion_control", {"prompt": "x", "source_video": "a",
                                           "shorter_size": 513}, {})
    with pytest.raises(ProviderError, match="strength"):
        ltx25.normalize("motion_control", {"prompt": "x", "source_video": "a"},
                        {"strength": 1.5})
    with pytest.raises(ProviderError, match="canny"):
        ltx25.normalize("union_control", {"prompt": "x", "control_video": "a",
                                          "canny_low": 0.9, "canny_high": 0.2}, {})


def test_control_payload_skips_frames_math():
    p = ltx25.normalize("inpaint", {"prompt": "x", "source_video": "a",
                                    "mask_image": "m"}, {})
    assert "num_frames" not in p["generation"]  # duration follows the source video
    assert p["control"] is True


def test_video_fps_follows_generation():
    # regression: container playback rate must match request fps (was 30)
    p = ltx25.normalize("t2v", {"prompt": "x"}, {"fps": 8, "duration": 3})
    _, g = p, ltx25.compile(p, {}, "/tmp", "/base")["graph"]
    cv = next(n for n in g.values() if n["class_type"] == "CreateVideo")
    assert cv["inputs"]["fps"] == 8
    assert g["7"]["inputs"]["frame_rate"] == 8


def test_control_graphs_crop_guides():
    # IC-LoRA guide tokens must be stripped before decode or outputs land 2x
    # the requested duration (regression from live e2e 2026-09-30)
    for wf, inputs in (
        ("union_control", {"prompt": "x", "control_video": "a"}),
        ("motion_control", {"prompt": "x", "source_video": "a"}),
        ("inpaint", {"prompt": "x", "source_video": "a", "mask_image": "m"}),
        ("outpaint", {"prompt": "x", "source_video": "a",
                      "canvas_width": 1024, "canvas_height": 576}),
        ("ic_lora", {"prompt": "x", "reference_sheet": "s"}),
        ("ic_lora", {"prompt": "x", "mode": "v2v", "source_video": "a"}),
    ):
        p = ltx25.normalize(wf, inputs, {})
        g = ltx25.compile(p, {k: "/tmp/x" for k in p["assets"]}, "/tmp", "/b")["graph"]
        cid, crop = next((k, n) for k, n in g.items()
                         if n["class_type"] == "LTXVCropGuides")
        dec = next(n for n in g.values() if n["class_type"] == "VAEDecodeTiled")
        assert dec["inputs"]["samples"] == [cid, 2]
        assert crop["inputs"]["latent"] == ["16", 0]
