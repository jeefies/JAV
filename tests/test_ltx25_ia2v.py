"""ltx25.ia2v: first-image + locked recording dual-condition workflow (2026-10-06)."""
from pathlib import Path

import pytest

from jav import config
from jav.capabilities import IMPLEMENTED
from jav.models import PROVIDER_WORKFLOWS, ProviderError
from jav.providers import ltx25


@pytest.fixture(autouse=True)
def _tpl_root(monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", Path("/mnt/data/AV/JAV"))


def _compile(inputs, generation=None):
    p = ltx25.normalize("ia2v", inputs, generation or {})
    t = ltx25.compile(p, {k: f"/tmp/{k}.bin" for k in p["assets"]}, "/tmp", "/base")
    return p, t["graph"]


def test_registered_everywhere():
    assert PROVIDER_WORKFLOWS["ltx25"]["ia2v"] == "ltx25"
    assert "ltx25.ia2v" in IMPLEMENTED


def test_requires_both_assets():
    with pytest.raises(ProviderError):
        ltx25.normalize("ia2v", {"prompt": "x", "audio": "a1"}, {})
    with pytest.raises(ProviderError):
        ltx25.normalize("ia2v", {"prompt": "x", "first_image": "a1"}, {})


def test_image_aliases():
    p, _ = _compile({"prompt": "x", "image": "asset_1", "audio": "asset_2"})
    assert p["assets"] == {"first_image": "asset_1", "audio": "asset_2"}
    p, _ = _compile({"prompt": "x", "first_frame": "asset_1", "audio": "asset_2"})
    assert p["assets"]["first_image"] == "asset_1"


def test_graph_wiring():
    _, g = _compile({"prompt": "cinematic", "first_image": "a1", "audio": "a2"},
                    {"duration": 4, "fps": 24, "strength": 0.6, "seed": 7})
    assert g["21"]["inputs"]["image"] == "asset:first_image"
    assert g["50"]["inputs"]["audio"] == "asset:audio"
    # image chain feeds concat as the video half; locked audio as the other
    assert g["10"]["inputs"]["video_latent"] == ["23", 0]
    assert g["10"]["inputs"]["audio_latent"] == ["54", 0]
    assert g["23"]["inputs"]["latent"] == ["8", 0]
    assert g["23"]["inputs"]["strength"] == 0.6
    assert g["8"]["inputs"]["length"] == 97          # 1 + floor(4*24/8)*8
    assert g["51"]["inputs"]["duration"] == 4        # trim follows duration
    assert g["53"]["inputs"]["value"] == 0.0         # audio latent stays locked
    assert g["19"]["inputs"]["audio"] == ["51", 0]   # mux the recording, not generated audio
    assert g["11"]["inputs"]["noise_seed"] == 7


def test_fast_only_falls_back_to_base_template():
    p, _ = _compile({"prompt": "x", "first_image": "a1", "audio": "a2"},
                    {"mode": "high"})
    assert p["mode"] == "high"   # accepted like a2v (no upscale variant yet), runs fast path
