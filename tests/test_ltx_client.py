"""Tests for ltx_client.py — the fast motion engine.

LTX's stock template is shaped differently from Hunyuan/Wan in two ways that
silently break the shared path, and both are locked in here: it sizes clips in
SECONDS rather than frames, and its all-in-one node emits a VIDEO with no
VAEDecode to hang a SaveImage off.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import ltx_client as lx


def _tpl(tmp_path, placeholder=True):
    g = {
        "1": {"class_type": "LTXImageToVideo",
              "inputs": {"prompt": "RUFUS_PROMPT" if placeholder else "static",
                         "width": 1280, "height": 720, "duration": 5, "fps": 25,
                         "seed": 1, "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
    }
    p = tmp_path / "ltx_i2v_api.json"
    p.write_text(json.dumps(g))
    return p


def test_enabled_default_on(monkeypatch):
    monkeypatch.delenv("RUFUS_LTX", raising=False)
    monkeypatch.delenv("RUFUS_STILLS_ONLY", raising=False)
    assert lx.enabled() is True


def test_stills_only_disables_it(monkeypatch):
    monkeypatch.setenv("RUFUS_STILLS_ONLY", "1")
    monkeypatch.setenv("RUFUS_LTX", "1")
    assert lx.enabled() is False


def test_ready_fails_without_export(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_LTX_TEMPLATE", str(tmp_path / "nope.json"))
    ok, why = lx.ready()
    assert not ok and "Export (API)" in why


def test_ready_fails_without_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_LTX_TEMPLATE", str(_tpl(tmp_path, placeholder=False)))
    ok, why = lx.ready()
    assert not ok and "RUFUS_PROMPT" in why


def test_prepare_keeps_the_video_writer(monkeypatch, tmp_path):
    """The critical shape difference: this graph has no VAEDecode, so the
    shared strip-writers path would delete its ONLY output and render nothing."""
    import comfy_template as ct
    tpl = ct.load_template(_tpl(tmp_path))
    g = lx._prepare(tpl, "a coin falls", "up.png", 42, (832, 1472, 121))
    classes = {n["class_type"] for n in g.values()}
    assert "SaveVideo" in classes          # kept, not stripped
    assert "rufus_save" not in g           # no frame-saver bolted on


def test_prepare_converts_frames_to_portrait_seconds(monkeypatch, tmp_path):
    """Without duration support the export's 1280x720 LANDSCAPE would survive
    into a 1080x1920 portrait pipeline."""
    import comfy_template as ct
    tpl = ct.load_template(_tpl(tmp_path))
    g = lx._prepare(tpl, "p", "up.png", 42, (832, 1472, 121))
    ins = g["1"]["inputs"]
    assert (ins["width"], ins["height"]) == (832, 1472)   # portrait now
    assert ins["duration"] == 5                            # 121 frames @ 25fps
    assert ins["prompt"] == "p"
    assert g["2"]["inputs"]["image"] == "up.png"


def test_animate_returns_false_without_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_LTX_TEMPLATE", str(tmp_path / "nope.json"))
    assert lx.animate_image(tmp_path / "a.png", tmp_path / "o.mp4") is False


def test_motion_prompt_demands_sustained_motion():
    p = lx._motion_prompt("A vault door swinging open")
    assert "vault door" in p
    assert "never static" in p
    # freeze-extended downstream, so a COMPLETED action would visibly stall
    assert "never completes" in p


def test_settings_reports_the_template_model(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_LTX_TEMPLATE", str(_tpl(tmp_path)))
    s = lx.settings()
    assert s["ckpt_name"] == "ltx-2.3-22b-dev-fp8.safetensors"
    assert s["width"] == 832 and s["height"] == 1472


def test_settings_survives_missing_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_LTX_TEMPLATE", str(tmp_path / "nope.json"))
    assert lx.settings()["width"] == 832      # env half still resolves, no raise
