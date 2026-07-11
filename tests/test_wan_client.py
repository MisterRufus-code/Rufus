"""Tests for the Wan 2.2 image-to-video engine. Pure-function + mocked-HTTP —
no ComfyUI server needed. The graph is blind-wired (no API export captured),
so these lock in structure and the fail-safe contract, not server behavior."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import wan_client as w


# ── Env knob ─────────────────────────────────────────────────────────────────────

def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_WAN", raising=False)
    assert w.enabled() is True


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RUFUS_WAN", "0")
    assert w.enabled() is False


# ── Graph structure ──────────────────────────────────────────────────────────────

def _graph():
    return w._build_wan_graph("init.png", "gentle push-in", 999,
                              frames=81, steps=4, shift=5.0)


def test_wan_graph_is_json_serializable_and_two_stage():
    g = _graph()
    json.dumps(g)   # must not raise

    hi, lo = g["13"]["inputs"], g["14"]["inputs"]
    # High-noise expert starts the schedule and hands off leftover noise…
    assert hi["add_noise"] == "enable"
    assert hi["start_at_step"] == 0 and hi["end_at_step"] == 2
    assert hi["return_with_leftover_noise"] == "enable"
    # …low-noise expert finishes from the SAME latent, no fresh noise.
    assert lo["add_noise"] == "disable"
    assert lo["start_at_step"] == 2
    assert lo["latent_image"] == ["13", 0]
    # Each sampler runs its own expert (loras chained onto separate UNETs)
    assert hi["model"] == ["5", 0] and lo["model"] == ["6", 0]
    assert g["3"]["inputs"]["model"] == ["1", 0]
    assert g["4"]["inputs"]["model"] == ["2", 0]
    # lightx2v recipe: cfg 1.0 both stages
    assert hi["cfg"] == 1.0 and lo["cfg"] == 1.0


def test_wan_graph_conditioning_wiring():
    g = _graph()
    cond = g["12"]["inputs"]
    assert cond["start_image"] == ["11", 0]           # LoadImage
    assert cond["vae"] == ["10", 0]                   # VAELoader
    assert cond["length"] == 81
    assert cond["width"] == w.SVD_W and cond["height"] == w.SVD_H
    assert g["8"]["inputs"]["text"] == "gentle push-in"
    assert g["9"]["inputs"]["text"] == w.NEGATIVE_PROMPT
    assert g["7"]["inputs"]["type"] == "wan"          # umt5 loaded in wan mode
    # Both samplers read the Wan conditioning triplet
    for node in ("13", "14"):
        assert g[node]["inputs"]["positive"] == ["12", 0]
        assert g[node]["inputs"]["negative"] == ["12", 1]
    # Decode reads the low-noise output with the Wan VAE
    assert g["15"]["inputs"]["samples"] == ["14", 0]
    assert g["15"]["inputs"]["vae"] == ["10", 0]


def test_wan_graph_passes_model_filenames():
    g = _graph()
    assert g["1"]["inputs"]["unet_name"] == w.HIGH_MODEL
    assert g["2"]["inputs"]["unet_name"] == w.LOW_MODEL
    assert g["3"]["inputs"]["lora_name"] == w.HIGH_LORA
    assert g["4"]["inputs"]["lora_name"] == w.LOW_LORA
    assert g["7"]["inputs"]["clip_name"] == w.CLIP_NAME
    assert g["10"]["inputs"]["vae_name"] == w.VAE_NAME


# ── Readiness preflight (fail-closed) ────────────────────────────────────────────

def test_ready_false_when_node_missing():
    class R:
        status_code = 404
        def json(self):
            return {}
    with patch.object(w.requests, "get", return_value=R()):
        ok, why = w.ready()
    assert ok is False
    assert "WanImageToVideo" in why


def test_ready_false_when_models_missing():
    def fake_get(url, timeout=10):
        class R:
            status_code = 200
            def json(self):
                if "WanImageToVideo" in url:
                    return {"WanImageToVideo": {"input": {}}}
                return {"UNETLoader": {"input": {"required":
                        {"unet_name": [["some_other_model.safetensors"]]}}}}
        return R()
    with patch.object(w.requests, "get", side_effect=fake_get):
        ok, why = w.ready()
    assert ok is False
    assert "diffusion_models" in why


def test_ready_true_when_node_and_models_present():
    def fake_get(url, timeout=10):
        class R:
            status_code = 200
            def json(self):
                if "WanImageToVideo" in url:
                    return {"WanImageToVideo": {"input": {}}}
                return {"UNETLoader": {"input": {"required":
                        {"unet_name": [[w.HIGH_MODEL, w.LOW_MODEL, "x.safetensors"]]}}}}
        return R()
    with patch.object(w.requests, "get", side_effect=fake_get):
        ok, why = w.ready()
    assert ok is True


def test_ready_false_when_server_unreachable():
    with patch.object(w.requests, "get", side_effect=OSError("refused")):
        ok, why = w.ready()
    assert ok is False


# ── Fail-safe animate ────────────────────────────────────────────────────────────

def test_animate_image_false_when_source_missing(tmp_path):
    assert w.animate_image(tmp_path / "nope.png", tmp_path / "out.mp4") is False


def test_animate_image_false_when_submit_rejected(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    monkeypatch.setattr(w, "_prep_init_image", lambda a, b: True)
    monkeypatch.setattr(w, "_upload_image", lambda p: "init.png")
    monkeypatch.setattr(w, "_submit_verbose", lambda g, c: None)   # graph rejected
    assert w.animate_image(src, tmp_path / "out.mp4", prompt="x") is False


def test_motion_prompt_includes_subject_and_restraint():
    mp = w._motion_prompt("A 1907 bank run, crowds outside a marble bank")
    assert "1907 bank run" in mp
    assert "slowly" in mp.lower() or "subtle" in mp.lower()
