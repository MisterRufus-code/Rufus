"""Tests for the ComfyUI stills video source. Pure-function + mocked-HTTP
level — no ComfyUI server needed to run these.

There is deliberately no built-in fallback image model (see comfy_client.py's
module docstring) — the old FLUX.1-dev graph was removed because FLUX.1-dev
is non-commercial-licensed and this pipeline is monetized. Every test below
that needs generate_clips()/_render_image() to actually proceed supplies a
stills template via _dummy_tpl(), standing in for a real
config/stills_api.json export (Z-Image-Turbo in production)."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client as c


def _dummy_tpl():
    """A minimal valid stills template — stands in for a real ComfyUI export
    (Z-Image-Turbo in production) wherever a test needs _stills_template() to
    return something so generate_clips()/_render_image() proceed past the
    'no template configured' guard, without touching the filesystem."""
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "2": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "s", "images": ["1", 0]}},
    }


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


# ── Composition-preserving fit ───────────────────────────────────────────────────

_STILLS_W, _STILLS_H = 832, 1472   # the documented stills-model generation bucket


def test_fit_to_frame_outputs_the_formats_exact_size(tmp_path):
    import io
    import random
    from PIL import Image

    # A generation-bucket-sized frame like the stills model produces. Noise,
    # not a flat color — the function's size sanity gate (>20KB) is calibrated
    # for real photos, and a solid color compresses below it.
    rng = random.Random(42)
    src = Image.new("RGB", (_STILLS_W, _STILLS_H))
    src.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(_STILLS_W * _STILLS_H)])
    buf = io.BytesIO()
    src.save(buf, format="PNG")

    out = tmp_path / "fit.png"
    assert c._fit_to_frame(buf.getvalue(), out) is True
    assert Image.open(out).size == (1080, 1920)


def test_fit_to_frame_trims_only_a_sliver():
    # 832/1472 vs 1080/1920: cover-scale then crop must discard <2% per axis —
    # the whole point vs. a 2×-upscale-then-crop that would discard 35%.
    scale = max(1080 / _STILLS_W, 1920 / _STILLS_H)
    new_w, new_h = round(_STILLS_W * scale), round(_STILLS_H * scale)
    assert (new_w - 1080) / new_w < 0.02
    assert (new_h - 1920) / new_h < 0.02


# ── Checkpoint preflight ─────────────────────────────────────────────────────────

def _obj_info_fixture(names):
    return {"CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [names, {"tooltip": "checkpoint to load"}]}}}}


def test_parse_checkpoint_list_extracts_names():
    got = c._parse_checkpoint_list(_obj_info_fixture(["flux1-dev-fp8.safetensors", "sd15.ckpt"]))
    assert got == ["flux1-dev-fp8.safetensors", "sd15.ckpt"]


def test_parse_checkpoint_list_garbage_shapes_return_empty():
    assert c._parse_checkpoint_list({}) == []
    assert c._parse_checkpoint_list({"CheckpointLoaderSimple": {}}) == []
    assert c._parse_checkpoint_list(
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": "oops"}}}}) == []
    assert c._parse_checkpoint_list(
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [None]}}}}) == []


def test_generate_clips_returns_empty_when_no_stills_template():
    # No config/stills_api.json exported → nothing to render. No built-in
    # fallback model (see module docstring) — return [] and let main.py fall
    # through to sd/diffusers/pexels, same as ComfyUI being offline.
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=None):
        assert c.generate_clips(["a prompt"], n=1) == []


def test_generate_clips_empty_when_submit_fails():
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_submit", return_value=None), \
         patch.object(c, "MAX_DUP_RETRIES", 0), \
         patch.object(c, "GEN_ERROR_BACKOFF", 0):
        assert c.generate_clips(["a prompt"], n=1) == []


def test_generate_clips_backs_off_between_hard_failures(monkeypatch):
    # A ComfyUI-side generation error (vs. a plain duplicate) is often a
    # transient GPU/model-loading hiccup — regression test for hammering the
    # same broken state 3x back-to-back with no pause.
    sleeps = []
    monkeypatch.setattr(c.time, "sleep", lambda s: sleeps.append(s))
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_submit", return_value=None), \
         patch.object(c, "MAX_DUP_RETRIES", 2):
        assert c.generate_clips(["a prompt"], n=1) == []
    assert sleeps == [c.GEN_ERROR_BACKOFF] * 3


# ── _render_image: stills template only, no built-in fallback model ──────────
# Face restoration and the hardcoded FLUX.1-dev fallback graph were removed
# together (FLUX.1-dev is non-commercial-licensed; the restore node only ever
# attached to that now-removed graph). A future face-restore pass would need
# to attach to the stills-template graph instead — not implemented here.

def test_render_image_returns_bytes_on_success(monkeypatch):
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"IMG"):
        assert c._render_image("x", 1, "cid") == b"IMG"


def test_render_image_none_when_no_template_configured(monkeypatch):
    monkeypatch.setattr(c, "_stills_template", lambda: None)
    with patch.object(c, "_submit") as sub:
        assert c._render_image("x", 1, "cid") is None
    sub.assert_not_called()   # nothing submitted with no model configured


def test_render_image_none_when_submit_fails(monkeypatch):
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value=None):
        assert c._render_image("x", 1, "cid") is None


def test_render_image_none_when_await_fails(monkeypatch):
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=None):
        assert c._render_image("x", 1, "cid") is None


# ── Scheduled runner hygiene ─────────────────────────────────────────────────────

def test_run_scheduled_bat_is_task_scheduler_safe():
    bat = (Path(__file__).parent.parent / "run_scheduled.bat").read_text()
    commands = [l.strip().lower() for l in bat.splitlines()
                if l.strip() and not l.strip().upper().startswith("REM")]
    assert "pause" not in commands             # a pause command hangs the scheduled task
    # the scheduled run IS the product run — no --skip-upload on any command line
    assert not any("--skip-upload" in l for l in commands)
    assert any("main.py" in l for l in commands)


def test_run_scheduled_bat_wires_in_the_feedback_loop():
    # analytics_fetcher.py + feedback_analyzer.py were built but never invoked
    # by the schedule — dormant until wired in here. Must run BEFORE main.py
    # so today's script can actually use freshly updated learnings.json.
    bat = (Path(__file__).parent.parent / "run_scheduled.bat").read_text()
    lines = bat.splitlines()
    idx = {name: next(i for i, l in enumerate(lines) if name in l)
           for name in ("analytics_fetcher.py", "feedback_analyzer.py", "main.py")}
    assert idx["analytics_fetcher.py"] < idx["main.py"]
    assert idx["feedback_analyzer.py"] < idx["main.py"]


def test_run_scheduled_bat_propagates_exit_code():
    # A silently-swallowed failure defeats the whole point of alerting on one —
    # the script must both capture and ultimately exit with main.py's real code.
    bat = (Path(__file__).parent.parent / "run_scheduled.bat").read_text()
    assert "set RUFUS_EXIT=%ERRORLEVEL%" in bat
    assert "exit /b %RUFUS_EXIT%" in bat


def test_run_scheduled_bat_date_var_set_outside_any_conditional_block():
    # Regression guard for a real batch pitfall caught before shipping: setting
    # a variable and reading it back with %var% (not !var!) inside the SAME
    # parenthesized if-block silently sees a stale/empty value, because the
    # whole block is %-expanded once at parse time before it runs line by line.
    bat = (Path(__file__).parent.parent / "run_scheduled.bat").read_text()
    lines = bat.splitlines()
    set_today_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("for /f") and "set TODAY=" in l)
    # Nothing before this line may be an unclosed "if ... (" — i.e. no line
    # above it may open a parenthesized block that hasn't already closed.
    depth = 0
    for l in lines[:set_today_idx]:
        depth += l.count("(") - l.count(")")
    assert depth == 0


def test_run_scheduled_bat_rotates_and_reports():
    """--scheduled makes a multi-niche schedule actually rotate (plain main.py
    silently ignored the schedule); report.py after main.py surfaces KPIs in
    the daily log instead of requiring a manual invocation nobody does."""
    bat = (Path(__file__).parent.parent / "run_scheduled.bat").read_text()
    lines = bat.splitlines()
    main_idx   = next(i for i, l in enumerate(lines) if "main.py" in l)
    assert "--scheduled" in lines[main_idx]
    report_idx = next(i for i, l in enumerate(lines) if "report.py" in l)
    assert report_idx > main_idx


def test_motion_chain_prefers_wan_then_svd(monkeypatch, tmp_path):
    """Engine order is wan → svd → Ken Burns: when Wan succeeds, SVD must not
    run; when Wan fails for an image, SVD gets it; Ken Burns takes the rest."""
    import types
    import wan_client
    import svd_client

    calls = []

    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))

    def wan_animate(png, clip, duration=8.0, idx=0, prompt=""):
        calls.append(("wan", idx))
        clip.write_bytes(b"x" * 60_000)
        return idx == 0                      # succeeds only for clip 0

    def svd_animate(png, clip, duration=8.0, idx=0, engine="comfy", prompt=""):
        calls.append(("svd", idx))
        clip.write_bytes(b"x" * 60_000)
        return True                          # picks up what wan dropped

    monkeypatch.setattr(wan_client, "animate_image", wan_animate)
    monkeypatch.setattr(svd_client, "img2vid_enabled", lambda: True)
    monkeypatch.setattr(svd_client, "resolve_engine", lambda: ("comfy", "test"))
    monkeypatch.setattr(svd_client, "animate_image", svd_animate)

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None):
        clips = c.generate_clips(["prompt one", "prompt two"], n=2)

    assert len(clips) == 2
    assert ("wan", 0) in calls                       # clip 0 → wan succeeded
    assert ("svd", 0) not in calls                   # …so svd never touched it
    assert ("wan", 1) in calls and ("svd", 1) in calls   # clip 1 walked the chain


# ── Stills template FILENAME resolution (stills_api.json / legacy flux2_api.json name) ──
# The template MECHANISM itself (comfy_template.py) is model-agnostic — these
# test which FILENAME comfy_client looks for, unrelated to which actual model
# (Z-Image-Turbo in production) the exported graph contains.

def _flux2_tpl(tmp_path):
    g = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "2": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "flux2", "images": ["1", 0]}},
    }
    p = tmp_path / "flux2_api.json"
    p.write_text(json.dumps(g))
    return p


def test_flux2_template_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "STILLS_TEMPLATE", tmp_path / "missing1.json")
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", tmp_path / "missing.json")
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)
    monkeypatch.delenv("RUFUS_STILLS_TEMPLATE", raising=False)
    assert c._flux2_template() is None


def _stills_tpl(tmp_path, name="stills_api.json"):
    g = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "2": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "s", "images": ["1", 0]}},
    }
    p = tmp_path / name
    p.write_text(json.dumps(g))
    return p


def test_generic_stills_template_loads_any_model_export(monkeypatch, tmp_path):
    """A stills_api.json export (Z-Image/Qwen/anything) is picked up model-
    agnostically, not just flux2_api.json."""
    monkeypatch.setattr(c, "STILLS_TEMPLATE", _stills_tpl(tmp_path))
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", tmp_path / "no_flux2.json")
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)
    monkeypatch.delenv("RUFUS_STILLS_TEMPLATE", raising=False)
    assert c._stills_template() is not None


def test_stills_template_env_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "STILLS_TEMPLATE", _stills_tpl(tmp_path))
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", tmp_path / "no_flux2.json")
    monkeypatch.setenv("RUFUS_STILLS_TEMPLATE", "0")
    assert c._stills_template() is None


def test_stills_template_falls_back_to_flux2_name(monkeypatch, tmp_path):
    """Back-compat: an existing flux2_api.json still works when there's no
    stills_api.json."""
    monkeypatch.setattr(c, "STILLS_TEMPLATE", tmp_path / "no_stills.json")
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", _flux2_tpl(tmp_path))
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)
    monkeypatch.delenv("RUFUS_STILLS_TEMPLATE", raising=False)
    assert c._stills_template() is not None


def test_flux2_template_env_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", _flux2_tpl(tmp_path))
    monkeypatch.setenv("RUFUS_FLUX2", "0")
    assert c._flux2_template() is None


def test_flux2_template_loads_with_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", _flux2_tpl(tmp_path))
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)
    assert c._flux2_template() is not None


def test_render_image_uses_legacy_flux2_filename_template(monkeypatch, tmp_path):
    """A template exported under the legacy flux2_api.json filename is used
    exactly like stills_api.json — the filename is historical, the model
    inside it is whatever the channel owner exported (Z-Image-Turbo, etc.)."""
    monkeypatch.setattr(c, "STILLS_TEMPLATE", tmp_path / "no_stills.json")
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", _flux2_tpl(tmp_path))
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)
    monkeypatch.delenv("RUFUS_STILLS_TEMPLATE", raising=False)

    submitted = []

    def fake_submit(graph, client_id):
        submitted.append(graph)
        return "pid1"

    with patch.object(c, "_submit", side_effect=fake_submit), \
         patch.object(c, "_await_image", return_value=b"PNGBYTES"):
        img = c._render_image("a coin", 123, "cid")

    assert img == b"PNGBYTES"
    assert len(submitted) == 1
    assert submitted[0]["1"]["inputs"]["text"] == "a coin"


def test_render_image_no_fallback_when_template_render_fails(monkeypatch):
    """The old behavior fell back to a hardcoded FLUX.1-dev graph on any
    template failure. That fallback is gone (non-commercial license) — a
    failed render now returns None, full stop, relying on generate_clips'
    own retry/reuse-previous-still safety net instead of a second model."""
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value="pid1"), \
         patch.object(c, "_await_image", return_value=None):
        assert c._render_image("a coin", 123, "cid") is None


def test_motion_chain_hunyuan_catches_wan_face_skip(monkeypatch, tmp_path):
    """Face shots: Wan skips → Hunyuan (the face engine) animates them,
    instead of falling to static Ken Burns."""
    import wan_client
    import hunyuan_client
    import svd_client

    calls = []

    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))
    monkeypatch.setattr(hunyuan_client, "enabled", lambda: True)
    monkeypatch.setattr(hunyuan_client, "ready", lambda: (True, "test"))
    monkeypatch.setattr(svd_client, "img2vid_enabled", lambda: False)

    def wan_animate(png, clip, duration=8.0, idx=0, prompt=""):
        calls.append(("wan", idx))
        return False                          # face shot — wan skips

    def hy_animate(png, clip, duration=8.0, idx=0, prompt=""):
        calls.append(("hunyuan", idx))
        clip.write_bytes(b"x" * 60_000)
        return True

    monkeypatch.setattr(wan_client, "animate_image", wan_animate)
    monkeypatch.setattr(hunyuan_client, "animate_image", hy_animate)

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None):
        clips = c.generate_clips(["a portrait of a banker"], n=1)

    assert len(clips) == 1
    assert ("wan", 0) in calls and ("hunyuan", 0) in calls


# ── Two-phase generation: stills first, then motion (24GB thrash fix) ─────────

def test_two_phase_all_renders_before_any_motion(monkeypatch, tmp_path):
    """Interleaving image→animate per clip forced ComfyUI to swap the stills
    model in and out for EVERY clip on a card that can't hold both models. All
    stills must now complete before the first motion call."""
    import wan_client

    order = []
    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))

    def wan_animate(png, clip, duration=8.0, idx=0, prompt=""):
        order.append(("animate", idx))
        clip.write_bytes(b"x" * 60_000)
        return True

    def fake_render(prompt, seed, client_id, niche=None):
        order.append(("render", None))
        return b"PNG"

    monkeypatch.setattr(wan_client, "animate_image", wan_animate)
    import svd_client
    monkeypatch.setattr(svd_client, "img2vid_enabled", lambda: False)

    freed = []
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_free_comfy_memory", side_effect=lambda: freed.append(1)):
        clips = c.generate_clips(["p1", "p2", "p3"], n=3)

    assert len(clips) == 3
    renders  = [i for i, ev in enumerate(order) if ev[0] == "render"]
    animates = [i for i, ev in enumerate(order) if ev[0] == "animate"]
    assert max(renders) < min(animates)          # every render precedes every animate
    assert freed == [1]                          # /free called exactly once, at the boundary


def test_failed_image_reuses_previous_still_for_beat_alignment(monkeypatch, tmp_path):
    """A failed image used to be SKIPPED, shifting every later clip one beat
    ahead of its narration. It must now reuse the previous still so clip[i]
    keeps matching beat[i]."""
    calls = {"n": 0}

    def fake_render(prompt, seed, client_id, niche=None):
        calls["n"] += 1
        # Fail every render attempt for the SECOND prompt only
        if "SECOND" in prompt:
            return None
        return b"PNG"

    def fake_kenburns(png, clip, duration=8.0, idx=0):
        clip.write_bytes(b"x" * 60_000)
        return True

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip", side_effect=fake_kenburns), \
         patch.object(c, "GEN_ERROR_BACKOFF", 0):
        clips = c.generate_clips(["FIRST prompt", "SECOND prompt", "THIRD prompt"], n=3)

    # All 3 beats still get a clip — beat 2 reuses beat 1's image
    assert len(clips) == 3


def test_free_not_called_in_stills_only_mode(monkeypatch, tmp_path):
    """No motion engines (RUFUS_WAN=0 etc.) → no model switch → no /free."""
    freed = []
    monkeypatch.setenv("RUFUS_WAN", "0")
    monkeypatch.setenv("RUFUS_HUNYUAN", "0")
    monkeypatch.setenv("RUFUS_IMG2VID", "0")
    def fake_kenburns(png, clip, duration=8.0, idx=0):
        clip.write_bytes(b"x" * 60_000)
        return True

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip", side_effect=fake_kenburns), \
         patch.object(c, "_free_comfy_memory", side_effect=lambda: freed.append(1)):
        clips = c.generate_clips(["p1", "p2"], n=2)
    assert len(clips) == 2
    assert freed == []


# ── Detail/realism direction on stills prompts ────────────────────────────────
# The stills model's encoder is an LLM (Qwen3-4B for Z-Image), so it reads
# descriptive prose — NOT sd_client's booru-style "8k, masterpiece" tag stack,
# which is the SD1.5 idiom and is out-of-distribution here.

def test_with_detail_appends_illustration_direction(monkeypatch):
    """DEFAULT_DETAIL_SUFFIX is flat 2D illustration, not photorealism — see
    its docstring and main.py's _FLUX_INSTRUCTION, changed together."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    out = c._with_detail("A macro shot of a tarnished coin")
    assert "flat 2d vector illustration" in out.lower()
    assert "A macro shot of a tarnished coin" in out
    # natural language, not the SD1.5 tag idiom
    assert "masterpiece" not in out.lower() and "8k" not in out.lower()
    # the old photorealistic direction must actually be gone, not just added
    # to — "depth of field" itself still legitimately appears as part of the
    # new text's own "no lens blur or depth of field" prohibition, so check
    # for the specific old positive instruction instead of the bare phrase.
    assert "85mm" not in out
    assert "shallow depth of field: the subject is tack-sharp" not in out


def test_with_detail_is_env_overridable(monkeypatch):
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "my own direction")
    assert c._with_detail("subject").endswith("my own direction")


def test_with_detail_disabled_when_env_empty(monkeypatch):
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "")
    assert c._with_detail("subject") == "subject"


def test_with_detail_skips_prompt_that_already_has_direction(monkeypatch):
    """A prompt already carrying photographic direction must not get a second,
    contradictory one — but only when the STYLE is itself photographic, where
    the prompt's spec is a more specific version of the same intent."""
    monkeypatch.setenv("RUFUS_STILLS_DETAIL",
                       "photorealistic, shot on a real camera")
    p = "a coin, 85mm f/1.4, moody light"
    assert c._with_detail(p) == p


def test_with_detail_strips_camera_spec_under_a_flat_style(monkeypatch):
    """Under the flat-2D default the prompt's camera spec is a CONTRADICTION,
    not a second opinion. Skipping the suffix (the old behaviour) rendered that
    one beat photoreal among nine flat-vector ones — a mixed look inside a
    single Short, which reads worse than either look on its own."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    out = c._with_detail("a coin, 85mm f/1.4, moody light")
    assert "Flat 2D vector illustration" in out
    assert "85mm" not in out and "f/1.4" not in out
    assert "moody light" in out


def test_generate_clips_sends_the_detailed_prompt(monkeypatch):
    """The suffix must be applied before render AND before the debug/log save,
    so what's reviewed afterwards is what was actually sent."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    seen = []

    def fake_render(prompt, seed, client_id, niche=None):
        seen.append(prompt)
        return b"PNG"

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0: clip.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a vintage ledger"], n=1)

    assert seen and "flat 2d vector illustration" in seen[0].lower()
    assert "a vintage ledger" in seen[0]


# ── Recurring-character path (character_engine.py integration) ───────────────
# _render_image tries the character path first when the niche has one
# enabled, and must fall back to the plain stills template on ANY miss along
# that path (no template exported, reference bootstrap failed, render
# failed) — character mode must never turn a working pipeline into a broken
# one. generate_clips(niche=...) threads niche through unchanged for every
# existing caller that doesn't pass it (defaults to None → character path
# never even attempted, identical to pre-character-mode behavior).

def test_render_image_skips_character_path_without_niche(monkeypatch):
    """No niche passed (every pre-existing caller) → character_engine must
    not even be consulted, let alone change behavior."""
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_render_character_image") as char_render, \
         patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"IMG"):
        assert c._render_image("x", 1, "cid") == b"IMG"
    char_render.assert_not_called()


def test_render_image_skips_character_path_when_niche_not_enabled(monkeypatch):
    import character_engine
    monkeypatch.setattr(character_engine, "enabled", lambda niche: False)
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_render_character_image") as char_render, \
         patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"IMG"):
        assert c._render_image("x", 1, "cid", niche="money_history") == b"IMG"
    char_render.assert_not_called()


def test_render_image_uses_character_path_when_enabled(monkeypatch):
    import character_engine
    monkeypatch.setattr(character_engine, "enabled", lambda niche: True)
    with patch.object(c, "_render_character_image", return_value=b"CHAR_IMG") as char_render, \
         patch.object(c, "_submit") as sub:
        out = c._render_image("x", 1, "cid", niche="money_history")
    assert out == b"CHAR_IMG"
    char_render.assert_called_once_with("x", 1, "cid", "money_history")
    sub.assert_not_called()   # plain path never reached once character path succeeds


def test_render_image_falls_back_to_plain_when_character_path_misses(monkeypatch):
    """Character path returning None (no template exported yet, reference
    bootstrap failed, etc.) must fall through to the ordinary stills
    template — never leave the beat with no image at all."""
    import character_engine
    monkeypatch.setattr(character_engine, "enabled", lambda niche: True)
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_render_character_image", return_value=None), \
         patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"PLAIN_IMG"):
        out = c._render_image("x", 1, "cid", niche="money_history")
    assert out == b"PLAIN_IMG"


def test_render_image_falls_back_when_character_engine_raises(monkeypatch):
    """A broken character_engine import/call must never take down a render
    that would otherwise succeed via the plain path — fail-open."""
    import character_engine
    def _boom(niche):
        raise RuntimeError("broken config")
    monkeypatch.setattr(character_engine, "enabled", _boom)
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"PLAIN_IMG"):
        out = c._render_image("x", 1, "cid", niche="money_history")
    assert out == b"PLAIN_IMG"


def test_character_template_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "CHARACTER_TEMPLATE", tmp_path / "missing.json")
    monkeypatch.delenv("RUFUS_CHARACTER_TEMPLATE", raising=False)
    assert c._character_template() is None


def test_character_template_loads_when_exported(monkeypatch, tmp_path):
    p = tmp_path / "character_stills_api.json"
    p.write_text(json.dumps({
        "1": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "s", "images": ["2", 0]}},
    }))
    monkeypatch.setattr(c, "CHARACTER_TEMPLATE", p)
    monkeypatch.delenv("RUFUS_CHARACTER_TEMPLATE", raising=False)
    assert c._character_template() is not None


def test_character_template_env_kill_switch(monkeypatch, tmp_path):
    p = tmp_path / "character_stills_api.json"
    p.write_text(json.dumps({
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
    }))
    monkeypatch.setattr(c, "CHARACTER_TEMPLATE", p)
    monkeypatch.setenv("RUFUS_CHARACTER_TEMPLATE", "0")
    assert c._character_template() is None


def test_ensure_character_reference_returns_existing_file(monkeypatch, tmp_path):
    import character_engine
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"already rendered")
    monkeypatch.setattr(character_engine, "reference_image_path", lambda niche: ref)
    with patch.object(c, "_submit") as sub:
        got = c._ensure_character_reference("money_history", "cid")
    assert got == ref
    sub.assert_not_called()   # existing reference is reused, never re-rendered


def test_ensure_character_reference_bootstraps_when_missing(monkeypatch, tmp_path):
    import character_engine
    ref = tmp_path / "ref.png"
    monkeypatch.setattr(character_engine, "reference_image_path", lambda niche: ref)
    monkeypatch.setattr(character_engine, "character_sheet_prompt",
                        lambda niche: "a character reference sheet")
    monkeypatch.setattr(c, "_stills_template", lambda: _dummy_tpl())
    with patch.object(c, "_submit", return_value="pid-1"), \
         patch.object(c, "_await_image", return_value=b"REF_PNG"):
        got = c._ensure_character_reference("money_history", "cid")
    assert got == ref
    assert ref.read_bytes() == b"REF_PNG"


def test_ensure_character_reference_none_without_plain_template(monkeypatch, tmp_path):
    import character_engine
    ref = tmp_path / "ref.png"
    monkeypatch.setattr(character_engine, "reference_image_path", lambda niche: ref)
    monkeypatch.setattr(character_engine, "character_sheet_prompt",
                        lambda niche: "a character reference sheet")
    monkeypatch.setattr(c, "_stills_template", lambda: None)
    assert c._ensure_character_reference("money_history", "cid") is None
    assert not ref.exists()


def _dummy_character_tpl():
    """A minimal valid character-consistency template — a LoadImage feeding
    the reference portrait alongside RUFUS_PROMPT, standing in for a real
    IPAdapter/PuLID export."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "3": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "s", "images": ["2", 0]}},
    }


def test_render_character_image_none_without_character_template(monkeypatch):
    monkeypatch.setattr(c, "_character_template", lambda: None)
    with patch.object(c, "_submit") as sub:
        assert c._render_character_image("x", 1, "cid", "money_history") is None
    sub.assert_not_called()


def test_render_character_image_none_when_reference_unavailable(monkeypatch):
    monkeypatch.setattr(c, "_character_template", lambda: _dummy_tpl())
    monkeypatch.setattr(c, "_ensure_character_reference", lambda niche, cid: None)
    with patch.object(c, "_submit") as sub:
        assert c._render_character_image("x", 1, "cid", "money_history") is None
    sub.assert_not_called()


def test_render_character_image_uses_uploaded_reference(monkeypatch, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    monkeypatch.setattr(c, "_character_template", lambda: _dummy_character_tpl())
    monkeypatch.setattr(c, "_ensure_character_reference", lambda niche, cid: ref)
    import svd_client
    monkeypatch.setattr(svd_client, "_upload_image", lambda p: "uploaded_ref.png")
    with patch.object(c, "_submit", return_value="pid-1") as sub, \
         patch.object(c, "_await_image", return_value=b"CHAR_PNG"):
        out = c._render_character_image("prompt text", 1, "cid", "money_history")
    assert out == b"CHAR_PNG"
    submitted_graph = sub.call_args[0][0]
    assert submitted_graph["1"]["inputs"]["image"] == "uploaded_ref.png"


def test_render_character_image_none_when_upload_fails(monkeypatch, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    monkeypatch.setattr(c, "_character_template", lambda: _dummy_character_tpl())
    monkeypatch.setattr(c, "_ensure_character_reference", lambda niche, cid: ref)
    import svd_client
    monkeypatch.setattr(svd_client, "_upload_image", lambda p: None)
    with patch.object(c, "_submit") as sub:
        assert c._render_character_image("x", 1, "cid", "money_history") is None
    sub.assert_not_called()


def test_generate_clips_threads_niche_into_render(monkeypatch):
    """generate_clips(niche=...) must reach _render_image so character mode
    can activate — a regression guard for the plumbing, not the feature
    logic itself (covered above)."""
    seen_niches = []

    def fake_render(prompt, seed, client_id, niche=None):
        seen_niches.append(niche)
        return b"PNG"

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0: clip.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a prompt"], n=1, niche="money_history")

    assert seen_niches == ["money_history"]


# ── Frames-per-beat: animate by cutting between stills, not a motion model ────
# A motion model costs ~10 min/video; Z-Image-Turbo renders a still in seconds,
# so several stills per beat + hard cuts buys an animated feel far cheaper.

def test_progression_modifiers_single_frame_is_a_noop():
    assert c._progression_modifiers(1) == [""]


def test_progression_modifiers_include_the_unmodified_peak():
    """One modifier must be empty — that slot is the prompt exactly as the
    prompt-builder composed it, and the base render fills it."""
    for n in (2, 3, 4):
        mods = c._progression_modifiers(n)
        assert len(mods) == n
        assert "" in mods


def test_progression_modifiers_are_ordered_earlier_then_later():
    mods = c._progression_modifiers(3)
    assert "earlier" in mods[0].lower()
    assert mods[1] == ""
    assert "later" in mods[2].lower()


def test_progression_modifiers_clamped_to_available_steps():
    assert len(c._progression_modifiers(99)) == len(c._PROGRESSION_STEPS)


def test_frames_per_beat_defaults_to_one(monkeypatch):
    monkeypatch.delenv("RUFUS_FRAMES_PER_BEAT", raising=False)
    assert c._frames_per_beat() == 1


def test_frames_per_beat_reads_env_and_floors_at_one(monkeypatch):
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "3")
    assert c._frames_per_beat() == 3
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "0")
    assert c._frames_per_beat() == 1
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "nonsense")
    assert c._frames_per_beat() == 1


def test_concat_clips_empty_is_false(tmp_path):
    assert c._concat_clips([], tmp_path / "out.mp4") is False


def test_concat_clips_single_part_is_a_rename_not_a_reencode(tmp_path):
    part = tmp_path / "p0.mp4"
    part.write_bytes(b"x" * 100)
    out = tmp_path / "out.mp4"
    assert c._concat_clips([part], out) is True
    assert out.read_bytes() == b"x" * 100
    assert not part.exists()


def _multiframe_env(monkeypatch, n):
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", str(n))
    for var in ("RUFUS_WAN", "RUFUS_HUNYUAN", "RUFUS_LTX", "RUFUS_IMG2VID"):
        monkeypatch.setenv(var, "0")


def test_multiframe_renders_n_frames_per_beat_at_one_seed(monkeypatch):
    """Same seed across the sub-frames is what holds the composition steady
    while the progression modifier advances only the action."""
    _multiframe_env(monkeypatch, 3)
    calls = []

    def fake_render(prompt, seed, client_id, niche=None):
        calls.append((prompt, seed))
        return b"PNG"

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=0:
                          clip.write_bytes(b"x" * 60_000) or True), \
         patch.object(c, "_concat_clips",
                      lambda parts, out: out.write_bytes(b"x" * 60_000) or True):
        clips = c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert len(clips) == 1
    assert len(calls) == 3, "one render per frame"
    assert len({seed for _, seed in calls}) == 1, "all sub-frames share the seed"
    joined = " | ".join(p for p, _ in calls)
    assert "earlier" in joined and "later" in joined


def test_multiframe_orders_frames_earlier_peak_later(monkeypatch):
    """Regression: the base render IS the peak, so it must be slotted at the
    peak's position in the arc. Placing it first instead re-rendered the peak
    and dropped the 'moment earlier' frame entirely."""
    _multiframe_env(monkeypatch, 3)
    frame_order = []

    # Identify each frame by CONTENT, not filename: the temp stamp itself
    # contains an underscore ("<epoch>_<pid>"), so the base frame's name also
    # ends in "_<beat>.png" and cannot be told apart from a sub-frame by suffix.
    def fake_render(prompt, seed, client_id, niche=None):
        low = prompt.lower()
        return b"EARLIER" if "earlier" in low else (
            b"LATER" if "later" in low else b"PEAK")

    def fake_animate(png, clip, duration=8.0, idx=0, min_bytes=0):
        frame_order.append(Path(png).read_bytes().split(b"|")[0].decode())
        clip.write_bytes(b"x" * 60_000)
        return True

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b + b"|" + b"x" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip", side_effect=fake_animate), \
         patch.object(c, "_concat_clips",
                      lambda parts, out: out.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert frame_order == ["EARLIER", "PEAK", "LATER"]


def test_multiframe_bypasses_motion_engines(monkeypatch):
    """The two are different answers to 'how does this beat move' — running
    both would animate each sub-frame separately."""
    _multiframe_env(monkeypatch, 3)
    monkeypatch.delenv("RUFUS_WAN", raising=False)   # would otherwise be ON
    import wan_client
    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))
    animated = []
    monkeypatch.setattr(wan_client, "animate_image",
                        lambda *a, **k: animated.append(1) or True)

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=0:
                          clip.write_bytes(b"x" * 60_000) or True), \
         patch.object(c, "_concat_clips",
                      lambda parts, out: out.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert animated == [], "motion engine must not run in frames-per-beat mode"


def test_multiframe_sub_frames_are_not_hash_checked(monkeypatch):
    """Sub-frames are MEANT to resemble the base frame — running them through
    the dup gate would reject them, and adding their hashes to the accepted
    pool would make every later beat look like a duplicate."""
    _multiframe_env(monkeypatch, 3)
    # No prior-hash history: otherwise the BASE frame legitimately hits the dup
    # path and re-hashes on each retry, which would mask what this is measuring.
    monkeypatch.setenv("RUFUS_FRESH_IMAGES", "0")
    hashed = []

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", side_effect=lambda p: hashed.append(p) or 123), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=0:
                          clip.write_bytes(b"x" * 60_000) or True), \
         patch.object(c, "_concat_clips",
                      lambda parts, out: out.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert len(hashed) == 1, "only the base frame is hashed, not the sub-frames"


def test_multiframe_survives_a_failed_sub_frame(monkeypatch):
    """A sub-frame failing shortens the beat; it must not lose the whole clip."""
    _multiframe_env(monkeypatch, 3)
    calls = {"n": 0}

    def flaky(prompt, seed, client_id, niche=None):
        calls["n"] += 1
        return None if calls["n"] == 2 else b"PNG"

    animated = []

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=flaky), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=0:
                          animated.append(png) or clip.write_bytes(b"x" * 60_000) or True), \
         patch.object(c, "_concat_clips",
                      lambda parts, out: out.write_bytes(b"x" * 60_000) or True):
        clips = c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert len(clips) == 1
    assert len(animated) == 2, "beat keeps the two frames that rendered"


def test_frames_per_beat_one_keeps_the_original_single_still_path(monkeypatch):
    """Default must be byte-for-byte the old behaviour: one render, one
    Ken Burns clip, no concat."""
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "1")
    for var in ("RUFUS_WAN", "RUFUS_HUNYUAN", "RUFUS_LTX", "RUFUS_IMG2VID"):
        monkeypatch.setenv(var, "0")
    renders, concats = [], []

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=lambda *a, **k: renders.append(1) or b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=50_000:
                          clip.write_bytes(b"x" * 60_000) or True), \
         patch.object(c, "_concat_clips", side_effect=lambda parts, out: concats.append(1) or True):
        clips = c.generate_clips(["a gold florin"], n=1, clip_duration=6.0)

    assert len(clips) == 1
    assert len(renders) == 1
    assert concats == [], "single-frame mode must not go through concat"


# ── i2i: chain each frame from the previous, then interpolate to real motion ──

def test_beat_motion_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("RUFUS_BEAT_MOTION", raising=False)
    assert c._beat_motion() == ""


def test_beat_motion_accepts_known_modes_only(monkeypatch):
    for mode in c.BEAT_MOTION_MODES:
        monkeypatch.setenv("RUFUS_BEAT_MOTION", mode.upper())
        assert c._beat_motion() == mode
    monkeypatch.setenv("RUFUS_BEAT_MOTION", "nonsense")
    assert c._beat_motion() == "", "unknown mode must fall back to legacy, not crash"


def test_i2i_template_none_when_not_exported(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "I2I_TEMPLATE", tmp_path / "missing.json")
    monkeypatch.delenv("RUFUS_I2I_TEMPLATE", raising=False)
    assert c._i2i_template() is None


def test_i2i_template_loads_and_has_kill_switch(monkeypatch, tmp_path):
    p = tmp_path / "stills_i2i_api.json"
    p.write_text(json.dumps({
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
    }))
    monkeypatch.setattr(c, "I2I_TEMPLATE", p)
    monkeypatch.delenv("RUFUS_I2I_TEMPLATE", raising=False)
    assert c._i2i_template() is not None
    monkeypatch.setenv("RUFUS_I2I_TEMPLATE", "0")
    assert c._i2i_template() is None


def test_i2i_step_prompt_moves_forward_only():
    """Unlike the `cut` arc, every i2i step is relative to the frame it starts
    FROM, so the sequence only ever advances."""
    base = "A gold florin changing hands"
    assert "later" in c._i2i_step_prompt(base, 1).lower()
    assert base in c._i2i_step_prompt(base, 1)
    # Past the end of the step list it clamps rather than raising.
    assert c._i2i_step_prompt(base, 99)


def test_build_i2i_chain_feeds_each_frame_into_the_next(tmp_path):
    """The whole point: frame k is generated FROM frame k-1, which is what
    makes the frames continuous enough to interpolate between."""
    inits = []

    def fake_i2i(prompt, seed, client_id, init_png):
        inits.append(Path(init_png).read_bytes())
        return b"RAW" + bytes([len(inits)])

    with patch.object(c, "_render_image_i2i", side_effect=fake_i2i), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True):
        frames = c._build_i2i_chain(
            base_png=tmp_path / "base.png", base_raw=b"RAW0", prompt="a florin",
            seed=1, client_id="cid", n=4, tmp_dir=tmp_path, stamp="s", beat=0)

    assert len(frames) == 4
    # frame 1 starts from the base render, frame 2 from frame 1's output, etc.
    assert inits[0] == b"RAW0"
    assert inits[1] == b"RAW" + bytes([1])
    assert inits[2] == b"RAW" + bytes([2])


def test_build_i2i_chain_uses_a_different_seed_each_link(tmp_path):
    """One seed reused across an img2img chain pulls every step back toward the
    same result — the previous image already supplies the continuity."""
    seeds = []

    with patch.object(c, "_render_image_i2i",
                      side_effect=lambda p, s, cid, init: seeds.append(s) or b"RAW"), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True):
        c._build_i2i_chain(base_png=tmp_path / "b.png", base_raw=b"R", prompt="x",
                           seed=5, client_id="cid", n=4, tmp_dir=tmp_path,
                           stamp="s", beat=0)

    assert len(seeds) == 3 and len(set(seeds)) == 3


def test_build_i2i_chain_stops_early_without_losing_the_beat(tmp_path):
    calls = {"n": 0}

    def flaky(prompt, seed, client_id, init_png):
        calls["n"] += 1
        return None if calls["n"] == 3 else b"RAW"

    with patch.object(c, "_render_image_i2i", side_effect=flaky), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True):
        frames = c._build_i2i_chain(base_png=tmp_path / "b.png", base_raw=b"R",
                                    prompt="x", seed=1, client_id="cid", n=5,
                                    tmp_dir=tmp_path, stamp="s", beat=0)

    assert len(frames) == 3, "keeps base + the two links that worked"


def test_build_i2i_chain_cleans_up_its_raw_intermediates(tmp_path):
    """The raw model outputs exist only to be fed to the next link."""
    with patch.object(c, "_render_image_i2i", return_value=b"RAW"), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True):
        c._build_i2i_chain(base_png=tmp_path / "b.png", base_raw=b"R", prompt="x",
                           seed=1, client_id="cid", n=4, tmp_dir=tmp_path,
                           stamp="s", beat=0)
    assert list(tmp_path.glob("*_raw*.png")) == []


def test_assemble_smooth_beat_empty_is_false(tmp_path):
    assert c._assemble_smooth_beat([], tmp_path / "o.mp4", 4.0) is False


def test_assemble_smooth_beat_single_frame_uses_ken_burns(tmp_path):
    """Nothing to interpolate between — must not build a one-frame sequence."""
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    with patch.object(c, "_animate_to_clip", return_value=True) as kb:
        assert c._assemble_smooth_beat([frame], tmp_path / "o.mp4", 4.0) is True
    kb.assert_called_once()


def test_assemble_smooth_beat_stages_the_last_frame_twice(tmp_path, monkeypatch):
    """Measured: minterpolate emits only (N-2) intervals because it needs a
    frame to interpolate TOWARD, so without a duplicate the beat came out
    3.63s instead of 4.80s and the final keyframe was never shown."""
    frames = []
    for k in range(3):
        f = tmp_path / f"f{k}.png"
        f.write_bytes(bytes([k]) * 10)
        frames.append(f)

    staged = {}

    def fake_run(cmd, **kw):
        seq = tmp_path / "o_seq"
        staged["files"] = sorted(p.name for p in seq.glob("*.png"))
        staged["contents"] = [(seq / n).read_bytes()[:1] for n in staged["files"]]
        out = tmp_path / "o.mp4"
        out.write_bytes(b"x" * 20_000)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(c.subprocess, "run", fake_run)
    assert c._assemble_smooth_beat(frames, tmp_path / "o.mp4", 4.0) is True
    assert len(staged["files"]) == 4, "3 frames staged plus a duplicate tail"
    assert staged["contents"][-1] == staged["contents"][-2], "tail is the last frame again"


def test_i2i_mode_falls_back_when_no_template_exported(monkeypatch):
    """Must degrade to plain stills, not produce a broken run."""
    monkeypatch.setenv("RUFUS_BEAT_MOTION", "i2i")
    monkeypatch.setenv("RUFUS_FRESH_IMAGES", "0")
    for var in ("RUFUS_WAN", "RUFUS_HUNYUAN", "RUFUS_LTX", "RUFUS_IMG2VID"):
        monkeypatch.setenv(var, "0")

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_i2i_template", return_value=None), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_build_i2i_chain") as chain, \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0, min_bytes=50_000:
                          clip.write_bytes(b"x" * 60_000) or True):
        clips = c.generate_clips(["a florin"], n=1, clip_duration=4.8)

    assert len(clips) == 1
    chain.assert_not_called()


def test_i2i_mode_interpolates_instead_of_cutting(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_BEAT_MOTION", "i2i")
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "4")
    monkeypatch.setenv("RUFUS_FRESH_IMAGES", "0")
    for var in ("RUFUS_WAN", "RUFUS_HUNYUAN", "RUFUS_LTX", "RUFUS_IMG2VID"):
        monkeypatch.setenv(var, "0")

    fake_frames = []
    for k in range(4):
        f = tmp_path / f"chain{k}.png"
        f.write_bytes(b"x" * 25_000)
        fake_frames.append(f)

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_i2i_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_build_i2i_chain", return_value=fake_frames), \
         patch.object(c, "_assemble_smooth_beat",
                      side_effect=lambda frames, out, dur:
                          out.write_bytes(b"x" * 60_000) or True) as smooth, \
         patch.object(c, "_concat_clips") as concat:
        clips = c.generate_clips(["a florin"], n=1, clip_duration=4.8)

    assert len(clips) == 1
    smooth.assert_called_once()
    assert smooth.call_args[0][0] == fake_frames, "all chain frames interpolated"
    assert not concat.called, "i2i must interpolate, not hard-cut"


def test_i2v_mode_forces_the_motion_chain_despite_frames_per_beat(monkeypatch):
    """A stale RUFUS_FRAMES_PER_BEAT must not silently bypass an explicit
    request for the motion model."""
    monkeypatch.setenv("RUFUS_BEAT_MOTION", "i2v")
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "3")
    monkeypatch.setenv("RUFUS_FRESH_IMAGES", "0")
    monkeypatch.delenv("RUFUS_WAN", raising=False)
    monkeypatch.setenv("RUFUS_HUNYUAN", "0")
    monkeypatch.setenv("RUFUS_LTX", "0")
    monkeypatch.setenv("RUFUS_IMG2VID", "0")

    import wan_client
    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))
    animated = []
    monkeypatch.setattr(wan_client, "animate_image",
                        lambda png, clip, **k: animated.append(1) or
                        (clip.write_bytes(b"x" * 60_000) or True))

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", return_value=b"PNG"), \
         patch.object(c, "_fit_to_frame", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_free_comfy_memory", lambda: None):
        c.generate_clips(["a florin"], n=1, clip_duration=4.8)

    assert animated == [1], "i2v mode must actually run the motion engine"


def test_i2v_mode_overrides_a_standing_stills_only_flag(monkeypatch):
    """run.bat hardcodes RUFUS_STILLS_ONLY=1, so without this an explicit
    request for the motion model is silently ignored and every beat comes out
    a Ken Burns zoom with nothing in the log explaining why."""
    import svd_client
    monkeypatch.setenv("RUFUS_STILLS_ONLY", "1")

    monkeypatch.delenv("RUFUS_BEAT_MOTION", raising=False)
    assert svd_client._stills_only() is True

    monkeypatch.setenv("RUFUS_BEAT_MOTION", "i2v")
    assert svd_client._stills_only() is False, \
        "an explicit i2v request must beat the blanket stills-only default"

    # Other modes must NOT punch through it — only i2v asks for motion models.
    for mode in ("i2i", "cut", "kenburns"):
        monkeypatch.setenv("RUFUS_BEAT_MOTION", mode)
        assert svd_client._stills_only() is True, mode


def test_i2v_mode_actually_enables_hunyuan_under_stills_only(monkeypatch):
    """End-to-end version of the above, through hunyuan_client.enabled()."""
    import hunyuan_client
    monkeypatch.setenv("RUFUS_STILLS_ONLY", "1")
    monkeypatch.delenv("RUFUS_HUNYUAN", raising=False)

    monkeypatch.delenv("RUFUS_BEAT_MOTION", raising=False)
    assert hunyuan_client.enabled() is False

    monkeypatch.setenv("RUFUS_BEAT_MOTION", "i2v")
    assert hunyuan_client.enabled() is True


# ── Cross-run freshness must be judged more loosely than within-run ──────────
# Proof this was wrong, from a live single-image test run:
#
#   [comfy] freshness: 120 image hash(es) from recent runs loaded
#   [comfy] 1/1: a worn gold coin on a wooden table...
#   [comfy] dup on clip 1 -> regen (retry 1)
#   [comfy] dup on clip 1 -> regen (retry 2)
#
# ONE image, no in-run predecessors, flagged twice. The only thing it could
# collide with was the 120-hash history. aHash reduces an image to an 8x8
# grayscale grid, and this channel's flat-2D style occupies a tiny corner of
# that space, so with enough accumulated history almost any new flat image
# lands within 6 bits of something. The pool became a ratchet: the longer the
# channel ran, the more good images were rejected and re-rendered for nothing.

def test_within_run_duplicate_still_caught_at_the_normal_threshold():
    """Two beats of the SAME video are 4 seconds apart — a loose resemblance
    is a real defect."""
    h = 0
    near = 0b1111                      # 4 bits away, inside DUP_THRESHOLD (6)
    assert c._is_duplicate(h, [near], n_prior=0)


def test_a_cross_run_lookalike_is_allowed_through():
    """The live failure. 4 bits from something rendered last week is not a
    reason to burn two more renders."""
    h = 0
    near = 0b1111                      # 4 bits: dup in-run, fine across runs
    assert not c._is_duplicate(h, [near], n_prior=1)


def test_a_cross_run_near_identical_is_still_rejected():
    """Loosening must not mean 'anything goes' — a frame 2 bits from a
    published one really is the same picture."""
    h = 0
    identical = 0b11                   # 2 bits, inside FRESH_DUP_THRESHOLD (3)
    assert c._is_duplicate(h, [identical], n_prior=1)


def test_the_two_pools_are_judged_separately():
    """Prior hashes must not make an in-run comparison stricter, or vice
    versa — they answer different questions."""
    h = 0
    prior_far = 0b111111111            # far from everything
    current_near = 0b1111              # 4 bits — in-run duplicate
    assert c._is_duplicate(h, [prior_far, current_near], n_prior=1)
    # Same distances, but now the near one came from history, not this run.
    assert not c._is_duplicate(h, [current_near, prior_far], n_prior=1)


def test_no_history_and_no_peers_is_never_a_duplicate():
    """The first image of the first run has nothing to be a duplicate OF."""
    assert not c._is_duplicate(12345, [], n_prior=0)


def test_cross_run_threshold_is_stricter_than_within_run():
    assert c.FRESH_DUP_THRESHOLD < c.DUP_THRESHOLD


def test_cli_defaults_to_stills_only():
    """RUFUS_STILLS_ONLY=1 is set by run.bat, not by this module, so a
    one-prompt check from the CLI fell through to the motion chain. On a
    16GB-RAM box that turned "does this prompt look right?" into a four-minute
    sample plus a VAE decode that ran for over an hour."""
    src = Path(c.__file__).read_text()
    main_block = src.split('if __name__ == "__main__":')[1]
    assert 'os.environ.setdefault("RUFUS_STILLS_ONLY", "1")' in main_block
    assert main_block.index('setdefault("RUFUS_STILLS_ONLY"') < \
           main_block.index("generate_clips("), "must be set BEFORE generating"


# ── the template the format switch cannot reach ──────────────────────────────

def test_a_matched_workflow_says_nothing(capsys):
    """832×1472 into 1080×1920 discards ~0.5%. A warning here would fire on
    every single frame of every normal run, which is the noise this repo has
    twice had to walk back."""
    c._crop_warned = False
    loss = c._warn_if_mostly_cropped(832, 1472)
    assert loss < 0.02
    assert capsys.readouterr().out == ""


def test_a_portrait_workflow_on_a_landscape_frame_is_loud(monkeypatch, capsys):
    """The failure nothing downstream can see: the render succeeds, QC passes,
    the file is exactly 1920×1080, and every picture in it is the middle
    third of a portrait image with the heads cropped off."""
    monkeypatch.setattr(c, "OUT_W", 1920)
    monkeypatch.setattr(c, "OUT_H", 1080)
    c._crop_warned = False
    loss = c._warn_if_mostly_cropped(832, 1472)
    assert loss > 0.6
    out = capsys.readouterr().out
    assert "832×1472" in out and "1920×1080" in out
    assert "stills_api.json" in out, "a warning that does not name the fix"


def test_the_crop_warning_is_said_once_not_per_frame(monkeypatch, capsys):
    """A hundred and fifty identical lines is not a louder warning."""
    monkeypatch.setattr(c, "OUT_W", 1920)
    monkeypatch.setattr(c, "OUT_H", 1080)
    c._crop_warned = False
    for _ in range(5):
        c._warn_if_mostly_cropped(832, 1472)
    assert capsys.readouterr().out.count("cropped away") == 1


def test_a_zero_sized_image_does_not_divide_by_zero():
    c._crop_warned = False
    assert c._warn_if_mostly_cropped(0, 0) == 0.0


# ── the re-roll a person does by hand ────────────────────────────────────────
#
# The loop was always there — MAX_DUP_RETRIES attempts, a fresh seed each time
# — and the only thing that could reject a frame was a perceptual-duplicate
# check. A six-panel contact sheet is not a duplicate of anything, so it was
# accepted on the first try and shipped.

def _gate_run(monkeypatch, tmp_path, verdicts, prompts=("the table goes over",),
              duplicate=False):
    """Run generate_clips with the gate answering `verdicts` in order."""
    import frame_gate
    seen_prompts = []
    calls = {"n": 0}

    def fake_check(path, prompt="", client=None):
        i = min(calls["n"], len(verdicts) - 1)
        calls["n"] += 1
        return verdicts[i]

    def fake_render(prompt, seed, client_id, niche=None, px=None):
        seen_prompts.append(prompt)
        return b"PNG"

    monkeypatch.setattr(frame_gate, "enabled", lambda: True)
    monkeypatch.setattr(frame_gate, "check", fake_check)
    monkeypatch.setenv("RUFUS_DEBUG_RUN_ID", f"gate_{len(verdicts)}")
    monkeypatch.setattr(c.paths, "debug_root", lambda: tmp_path)

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=1 if duplicate else None), \
         patch.object(c, "_is_duplicate", return_value=duplicate), \
         patch.object(c, "_animate_to_clip",
                      lambda *a, **k: a[1].write_bytes(b"x" * 60_000) or True):
        clips = c.generate_clips(list(prompts), n=len(prompts))
    return clips, seen_prompts


def test_a_rejected_frame_is_rendered_again(monkeypatch, tmp_path):
    ok = (True, "", "")
    bad = (False, "contact_sheet", "3x2 interior gutters")
    _, prompts = _gate_run(monkeypatch, tmp_path, [bad, ok])
    assert len(prompts) == 2, "the rejection did not cost a re-render"


def test_the_re_roll_is_told_what_was_wrong(monkeypatch, tmp_path):
    """A re-roll with the same prompt and a new seed IS the same prompt. The
    hint is the only thing that makes the second attempt different in the way
    that matters."""
    bad = (False, "contact_sheet", "3x2 interior gutters")
    _, prompts = _gate_run(monkeypatch, tmp_path, [bad, (True, "", "")])
    assert "never a grid" in prompts[1]
    assert "never a grid" not in prompts[0], "the first attempt is the storyboard's"


def test_a_frame_that_never_passes_is_kept_and_said_out_loud(monkeypatch, tmp_path, capsys):
    """Skipping would shift every later clip one beat earlier than its
    narration, which is far worse than one weak picture — the same argument
    the duplicate path already makes."""
    bad = (False, "lettering", "the word Proof")
    clips, prompts = _gate_run(monkeypatch, tmp_path, [bad])
    out = capsys.readouterr().out
    assert len(clips) == 1, "the beat still got a picture"
    assert "still lettering after" in out
    assert len(prompts) == 1 + c.GATE_RETRIES, "the whole budget was spent"


def test_the_gate_does_not_starve_the_duplicate_check(monkeypatch, tmp_path):
    """FOUND BY THIS TEST. The duplicate check re-rolled while
    `retry < MAX_DUP_RETRIES`, and `retry` is the loop index — which the gate's
    own rejections have already spent. Two gate re-rolls left the duplicate
    check with no attempts, so a run with the gate on quietly stopped
    de-duplicating. It compares against the attempts remaining now, and with
    the gate off that is the same number it always was."""
    from sd_client import MAX_DUP_RETRIES
    bad = (False, "blank_frame", "94% of the frame is bare paper")
    _, prompts = _gate_run(monkeypatch, tmp_path, [bad], duplicate=True)
    assert len(prompts) == 1 + c.GATE_RETRIES + MAX_DUP_RETRIES


def test_the_gate_writes_what_it_rejected(monkeypatch, tmp_path):
    """The gate has to be judgeable by a number, or the only way to know
    whether it helped is to open the folder again."""
    import json as _json
    bad = (False, "contact_sheet", "3x2 interior gutters")
    _gate_run(monkeypatch, tmp_path, [bad, (True, "", "")])
    written = _json.loads(
        (tmp_path / "gate_2" / "gate.json").read_text(encoding="utf-8"))
    assert written["frames"] == 1
    assert written["rejects"][0]["reason"] == "contact_sheet"


def test_with_the_gate_off_nothing_changes(monkeypatch, tmp_path):
    """The shipping channel renders exactly as it did — one attempt per frame,
    the storyboard's own prompt, no gate.json."""
    import frame_gate
    monkeypatch.setattr(frame_gate, "enabled", lambda: False)
    seen = []

    def fake_render(prompt, seed, client_id, niche=None, px=None):
        seen.append(prompt)
        return b"PNG"

    monkeypatch.setenv("RUFUS_DEBUG_RUN_ID", "gate_off")
    monkeypatch.setattr(c.paths, "debug_root", lambda: tmp_path)
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_frame",
                      lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda *a, **k: a[1].write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a quiet street"], n=1)

    assert len(seen) == 1, "one attempt per frame, exactly as before"
    assert seen[0].startswith("a quiet street")
    assert not (tmp_path / "gate_off" / "gate.json").exists()
