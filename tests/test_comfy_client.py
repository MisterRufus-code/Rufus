"""Tests for the ComfyUI + FLUX.1-dev video source. Pure-function + mocked-HTTP
level — no ComfyUI server needed to run these."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client as c


# ── FLUX workflow graph ──────────────────────────────────────────────────────────

def test_flux_graph_is_json_serializable_and_wired():
    g = c._build_flux_graph("a roman denarius coin, museum lighting", 999, "flux1-dev-fp8.safetensors", 20)
    json.dumps(g)  # must not raise
    # KSampler reads the model, positive (via FluxGuidance), negative, and latent
    ks = g["3"]["inputs"]
    assert ks["seed"] == 999 and ks["steps"] == 20
    assert ks["model"] == ["4", 0]
    assert ks["positive"] == ["15", 0]   # FluxGuidance node
    assert ks["negative"] == ["7", 0]
    assert ks["latent_image"] == ["10", 0]


def test_flux_graph_uses_portrait_latent():
    g = c._build_flux_graph("x", 1, "m.safetensors", 20)
    assert g["10"]["inputs"]["width"] == c.GEN_W
    assert g["10"]["inputs"]["height"] == c.GEN_H
    assert c.GEN_H > c.GEN_W           # portrait
    assert c.GEN_W % 16 == 0 and c.GEN_H % 16 == 0   # FLUX needs ÷16


def test_flux_graph_passes_prompt_and_model_through():
    g = c._build_flux_graph("hyperinflation banknotes 1923", 7, "custom.safetensors", 12)
    assert g["6"]["inputs"]["text"] == "hyperinflation banknotes 1923"
    assert g["4"]["inputs"]["ckpt_name"] == "custom.safetensors"
    assert g["3"]["inputs"]["steps"] == 12


# ── Graceful degradation when the server is down ─────────────────────────────────

def test_generate_clips_returns_empty_when_unavailable():
    with patch.object(c, "is_available", return_value=False):
        assert c.generate_clips(["test prompt"], n=1) == []


def test_is_available_false_on_connection_error():
    with patch("comfy_client.requests.get", side_effect=OSError("refused")):
        assert c.is_available() is False


# ── Perceptual dedup helpers ─────────────────────────────────────────────────────

def test_hamming_counts_differing_bits():
    assert c._hamming(0b1010, 0b1010) == 0
    assert c._hamming(0b1111, 0b0000) == 4
    assert c._hamming(0b1100, 0b1010) == 2
