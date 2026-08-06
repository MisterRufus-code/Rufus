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


def test_fit_to_portrait_outputs_exact_1080x1920(tmp_path):
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
    assert c._fit_to_portrait(buf.getvalue(), out) is True
    assert Image.open(out).size == (1080, 1920)


def test_fit_to_portrait_trims_only_a_sliver():
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
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
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
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
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

    def fake_render(prompt, seed, client_id):
        order.append(("render", None))
        return b"PNG"

    monkeypatch.setattr(wan_client, "animate_image", wan_animate)
    import svd_client
    monkeypatch.setattr(svd_client, "img2vid_enabled", lambda: False)

    freed = []
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
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

    def fake_render(prompt, seed, client_id):
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
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
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
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
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
    """A niche style_suffix or hand-written --topic prompt already carrying
    photographic direction must not get a second, contradictory one."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    p = "a coin, 85mm f/1.4, moody light"
    assert c._with_detail(p) == p


def test_generate_clips_sends_the_detailed_prompt(monkeypatch):
    """The suffix must be applied before render AND before the debug/log save,
    so what's reviewed afterwards is what was actually sent."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    seen = []

    def fake_render(prompt, seed, client_id):
        seen.append(prompt)
        return b"PNG"

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "_stills_template", return_value=_dummy_tpl()), \
         patch.object(c, "_render_image", side_effect=fake_render), \
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip",
                      lambda png, clip, duration=8.0, idx=0: clip.write_bytes(b"x" * 60_000) or True):
        c.generate_clips(["a vintage ledger"], n=1)

    assert seen and "flat 2d vector illustration" in seen[0].lower()
    assert "a vintage ledger" in seen[0]
