"""Tests for hunyuan_client.py — the template-driven face-capable motion engine."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import hunyuan_client as hy


def _write_template(tmp_path, placeholder=True):
    g = {
        "1": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT" if placeholder else "static"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "3": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
    }
    p = tmp_path / "hunyuan_i2v_api.json"
    p.write_text(json.dumps(g))
    return p


def test_enabled_default_on(monkeypatch):
    monkeypatch.delenv("RUFUS_HUNYUAN", raising=False)
    assert hy.enabled() is True


def test_enabled_env_off(monkeypatch):
    monkeypatch.setenv("RUFUS_HUNYUAN", "0")
    assert hy.enabled() is False


def test_ready_fails_without_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_HUNYUAN_TEMPLATE", str(tmp_path / "missing.json"))
    ok, why = hy.ready()
    assert not ok
    assert "Export (API)" in why or "export" in why.lower()


def test_ready_fails_without_placeholder(monkeypatch, tmp_path):
    p = _write_template(tmp_path, placeholder=False)
    monkeypatch.setenv("RUFUS_HUNYUAN_TEMPLATE", str(p))
    ok, why = hy.ready()
    assert not ok
    assert "RUFUS_PROMPT" in why


def test_ready_fails_when_nodes_missing(monkeypatch, tmp_path):
    p = _write_template(tmp_path)
    monkeypatch.setenv("RUFUS_HUNYUAN_TEMPLATE", str(p))
    with patch.object(hy.comfy_template, "missing_nodes",
                      return_value=["HunyuanVideo15ImageToVideo"]):
        ok, why = hy.ready()
    assert not ok
    assert "HunyuanVideo15ImageToVideo" in why


def test_ready_ok_with_template_and_nodes(monkeypatch, tmp_path):
    p = _write_template(tmp_path)
    monkeypatch.setenv("RUFUS_HUNYUAN_TEMPLATE", str(p))
    with patch.object(hy.comfy_template, "missing_nodes", return_value=[]):
        ok, why = hy.ready()
    assert ok
    assert "face" in why.lower()


def test_animate_returns_false_without_template(monkeypatch, tmp_path):
    """No template = instant False (chain falls through) — no network calls."""
    monkeypatch.setenv("RUFUS_HUNYUAN_TEMPLATE", str(tmp_path / "missing.json"))
    assert hy.animate_image(tmp_path / "a.png", tmp_path / "out.mp4") is False


def test_motion_prompt_face_stability_language():
    p = hy._motion_prompt("A medium portrait of a Roman official in a courtyard")
    assert "Roman official" in p
    assert "no morphing" in p
    assert "blinks naturally" in p
    # Same one-way-action constraint as Wan
    assert "one-way actions" in p


def test_motion_prompt_truncates_long_subjects():
    p = hy._motion_prompt("word " * 200)
    assert len(p) < 700
