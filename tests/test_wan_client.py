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
# Verified against an actual API export of the channel owner's proven-good
# test run: the template's default toggle ("Enable 4steps LoRA?") is FALSE,
# meaning real cfg (3.5) with NO LoRA in the loop at all — not just a guess.

def _graph(steps=12, cfg=3.5, use_lora=False):
    return w._build_wan_graph("init.png", "gentle push-in", 999, frames=81,
                              steps=steps, cfg=cfg, shift=5.0, use_lora=use_lora)


def test_wan_graph_default_matches_proven_export_no_lora():
    """The proven-good run had the model feed ModelSamplingSD3 DIRECTLY from
    UNETLoader — no LoraLoaderModelOnly node in the graph at all when the
    template's toggle is off. Regression guard: use_lora=False must not just
    set lora strength to 0, it must omit the node entirely, matching the
    export exactly."""
    g = _graph(use_lora=False)
    json.dumps(g)
    assert "3" not in g and "4" not in g   # no LoraLoaderModelOnly nodes at all
    assert g["5"]["inputs"]["model"] == ["1", 0]   # ModelSamplingSD3 <- UNETLoader directly
    assert g["6"]["inputs"]["model"] == ["2", 0]
    hi, lo = g["13"]["inputs"], g["14"]["inputs"]
    assert hi["cfg"] == 3.5 and lo["cfg"] == 3.5    # real CFG, the proven setting


def test_wan_graph_lora_mode_forces_cfg1_and_8_steps_by_default(monkeypatch):
    """use_lora=True is the fast opt-in — it forces cfg=1.0 (the distill's
    only supported guidance) and now defaults to 8 steps (4+4), NOT the old
    hardcoded 4: 4-step lightx2v produces documented slow/reduced motion,
    8 steps is the community fix. Chains LoraLoaderModelOnly onto each UNET."""
    monkeypatch.delenv("RUFUS_WAN_LORA_STEPS", raising=False)
    g = _graph(steps=12, cfg=3.5, use_lora=True)   # deliberately "wrong" inputs
    json.dumps(g)
    assert g["3"]["inputs"]["model"] == ["1", 0]
    assert g["3"]["inputs"]["lora_name"] == w.HIGH_LORA
    assert g["4"]["inputs"]["model"] == ["2", 0]
    assert g["4"]["inputs"]["lora_name"] == w.LOW_LORA
    assert g["5"]["inputs"]["model"] == ["3", 0]    # ModelSamplingSD3 <- LoRA output
    assert g["6"]["inputs"]["model"] == ["4", 0]
    hi, lo = g["13"]["inputs"], g["14"]["inputs"]
    assert hi["cfg"] == 1.0 and lo["cfg"] == 1.0    # cfg forced, ignoring the 3.5 passed in
    assert hi["steps"] == 8 and lo["steps"] == 8    # 8, not the passed-in 12 or the old 4
    assert hi["end_at_step"] == 4                   # split 4+4


def test_wan_lora_steps_env_override(monkeypatch):
    monkeypatch.setenv("RUFUS_WAN_LORA_STEPS", "6")
    g = _graph(use_lora=True)
    assert g["13"]["inputs"]["steps"] == 6
    assert g["13"]["inputs"]["end_at_step"] == 3    # split 3+3


def test_wan_lora_high_noise_strength_reduced_by_default(monkeypatch):
    """The high-noise LoRA is applied at reduced strength (default 0.8) to
    restore motion range the 4-step distill loses; low-noise stays at 1.0."""
    monkeypatch.delenv("RUFUS_WAN_LORA_HIGH_STRENGTH", raising=False)
    g = _graph(use_lora=True)
    assert g["3"]["inputs"]["strength_model"] == 0.8   # high-noise, reduced
    assert g["4"]["inputs"]["strength_model"] == 1.0   # low-noise, full


def test_wan_lora_high_noise_strength_env_override(monkeypatch):
    monkeypatch.setenv("RUFUS_WAN_LORA_HIGH_STRENGTH", "0.6")
    g = _graph(use_lora=True)
    assert g["3"]["inputs"]["strength_model"] == 0.6


def test_wan_graph_is_json_serializable_and_two_stage():
    g = _graph()
    json.dumps(g)   # must not raise

    hi, lo = g["13"]["inputs"], g["14"]["inputs"]
    # High-noise expert starts the schedule and hands off leftover noise…
    assert hi["add_noise"] == "enable"
    assert hi["start_at_step"] == 0 and hi["end_at_step"] == 6   # 12 steps / 2
    assert hi["return_with_leftover_noise"] == "enable"
    # …low-noise expert finishes from the SAME latent, no fresh noise.
    assert lo["add_noise"] == "disable"
    assert lo["start_at_step"] == 6
    assert lo["latent_image"] == ["13", 0]
    assert hi["model"] == ["5", 0] and lo["model"] == ["6", 0]


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
    g = _graph(use_lora=True)   # LoRA-mode is the one where lora filenames matter
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


def test_motion_prompt_bans_one_way_completing_actions():
    """Live glitch report: a man reading a newspaper had the page-turn visibly
    undo itself every cycle. Every clip is assembled forward-then-reversed for
    a seamless loop (see svd_client._assemble) — fine for ambient drift, but a
    one-way action (page turn, gesture, walking) played backward looks exactly
    like rewinding a tape. The prompt must explicitly ban these."""
    mp = w._motion_prompt("a man reading a newspaper").lower()
    assert "page turning" in mp or "page-turning" in mp
    assert "one-way" in mp
    assert "walking" in mp


# ── Face-skip (reinstated after a live glitch report) ────────────────────────

def test_skip_face_motion_default_is_true(monkeypatch):
    monkeypatch.delenv("RUFUS_WAN_FACE_MOTION", raising=False)
    assert w._skip_face_motion() is True


def test_skip_face_motion_opt_out(monkeypatch):
    monkeypatch.setenv("RUFUS_WAN_FACE_MOTION", "1")
    assert w._skip_face_motion() is False


def test_animate_image_skips_wan_for_face_prompt(tmp_path, monkeypatch):
    """Faces looked clean in the FLUX still but blurred/glitched once Wan
    animated them — regression test for the fix, mirroring SVD's equivalent."""
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    monkeypatch.delenv("RUFUS_WAN_FACE_MOTION", raising=False)
    monkeypatch.setattr(w, "_prep_init_image",
                        lambda a, b: (_ for _ in ()).throw(AssertionError("Wan pipeline touched")))
    assert w.animate_image(src, tmp_path / "out.mp4",
                           prompt="a medium portrait of a worried banker") is False


def test_animate_image_allows_faces_when_opted_in(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    calls = []
    monkeypatch.setenv("RUFUS_WAN_FACE_MOTION", "1")
    monkeypatch.setattr(w, "_prep_init_image", lambda a, b: calls.append(1) or True)
    monkeypatch.setattr(w, "_upload_image", lambda p: "init.png")
    monkeypatch.setattr(w, "_submit_verbose", lambda g, c: None)   # stop early, harmlessly

    w.animate_image(src, tmp_path / "out.mp4",
                    prompt="a medium portrait of a worried banker")

    # The face-skip must NOT have short-circuited — the pipeline was actually
    # entered (and only failed later, at the stubbed submit step).
    assert calls == [1]
