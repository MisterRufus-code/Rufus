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


# ── Composition-preserving fit ───────────────────────────────────────────────────

def test_fit_to_portrait_outputs_exact_1080x1920(tmp_path):
    import io
    import random
    from PIL import Image

    # A GEN_W×GEN_H frame like FLUX produces. Noise, not a flat color — the
    # function's size sanity gate (>20KB) is calibrated for real photos, and a
    # solid color compresses below it.
    rng = random.Random(42)
    src = Image.new("RGB", (c.GEN_W, c.GEN_H))
    src.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(c.GEN_W * c.GEN_H)])
    buf = io.BytesIO()
    src.save(buf, format="PNG")

    out = tmp_path / "fit.png"
    assert c._fit_to_portrait(buf.getvalue(), out) is True
    assert Image.open(out).size == (1080, 1920)


def test_fit_to_portrait_trims_only_a_sliver():
    # 832/1472 vs 1080/1920: cover-scale then crop must discard <2% per axis —
    # the whole point vs. the old 2×-upscale-then-crop that discarded 35%.
    scale = max(1080 / c.GEN_W, 1920 / c.GEN_H)
    new_w, new_h = round(c.GEN_W * scale), round(c.GEN_H * scale)
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


def test_generate_clips_refuses_missing_checkpoint():
    # Server up, but the configured FLUX model isn't in ComfyUI's list →
    # return [] (fall back down the chain) instead of submitting doomed jobs.
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "list_checkpoints", return_value=["something_else.safetensors"]):
        assert c.generate_clips(["a prompt"], n=1) == []


def test_generate_clips_proceeds_when_list_unavailable():
    # Empty list = endpoint couldn't be read → can't verify → fail-open past the
    # preflight (then stop at submit, which we stub to fail fast here).
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
         patch.object(c, "_submit", return_value=None), \
         patch.object(c, "MAX_DUP_RETRIES", 0), \
         patch.object(c, "GEN_ERROR_BACKOFF", 0):
        assert c.generate_clips(["a prompt"], n=1) == []   # fails at submit, not preflight


def test_generate_clips_backs_off_between_hard_failures(monkeypatch):
    # A ComfyUI-side generation error (vs. a plain duplicate) is often a
    # transient GPU/model-loading hiccup — regression test for hammering the
    # same broken state 3x back-to-back with no pause.
    sleeps = []
    monkeypatch.setattr(c.time, "sleep", lambda s: sleeps.append(s))
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
         patch.object(c, "_submit", return_value=None), \
         patch.object(c, "MAX_DUP_RETRIES", 2):
        assert c.generate_clips(["a prompt"], n=1) == []
    assert sleeps == [c.GEN_ERROR_BACKOFF] * 3


# ── Optional face restoration ─────────────────────────────────────────────────

def test_apply_face_restore_cf_repoints_saveimage():
    g = c._build_flux_graph("a portrait of a person", 1, "flux.safetensors", 28)
    c._apply_face_restore(g, {"kind": "facerestore_cf", "model": "GFPGANv1.4.pth", "fidelity": 0.5})
    json.dumps(g)                                    # must stay serializable
    assert g["21"]["class_type"] == "FaceRestoreCFWithModel"
    assert g["21"]["inputs"]["image"] == ["8", 0]    # reads the VAEDecode output
    assert g["21"]["inputs"]["facerestore_model"] == ["20", 0]
    assert g["9"]["inputs"]["images"] == ["21", 0]   # SaveImage now reads restored


def test_apply_face_restore_reactor_repoints_saveimage():
    g = c._build_flux_graph("a portrait", 1, "flux.safetensors", 28)
    c._apply_face_restore(g, {"kind": "reactor", "model": "GFPGANv1.4.pth", "fidelity": 0.5})
    json.dumps(g)
    assert g["21"]["class_type"] == "ReActorRestoreFace"
    assert g["21"]["inputs"]["image"] == ["8", 0]
    assert g["9"]["inputs"]["images"] == ["21", 0]


def test_resolve_face_restore_off_when_disabled(monkeypatch):
    monkeypatch.setenv("RUFUS_FACE_RESTORE", "0")
    assert c.resolve_face_restore() is None


def test_resolve_face_restore_none_when_no_node(monkeypatch):
    monkeypatch.delenv("RUFUS_FACE_RESTORE", raising=False)
    with patch.object(c, "_has_node", return_value=False):
        assert c.resolve_face_restore() is None


def test_resolve_face_restore_prefers_cf(monkeypatch):
    monkeypatch.delenv("RUFUS_FACE_RESTORE", raising=False)
    with patch.object(c, "_has_node", lambda cls: cls in ("FaceRestoreCFWithModel", "FaceRestoreModelLoader")):
        r = c.resolve_face_restore()
    assert r["kind"] == "facerestore_cf"


def test_resolve_face_restore_uses_reactor_when_only_it_present(monkeypatch):
    monkeypatch.delenv("RUFUS_FACE_RESTORE", raising=False)
    with patch.object(c, "_has_node", lambda cls: cls == "ReActorRestoreFace"):
        r = c.resolve_face_restore()
    assert r["kind"] == "reactor"


def test_render_image_falls_back_to_plain_when_restore_fails():
    # Face-restore submit is rejected (bad node/param) → must retry the SAME seed
    # on the plain graph so a misconfigured restore node never costs a clip.
    restore = {"kind": "reactor", "model": "GFPGANv1.4.pth", "fidelity": 0.5}
    calls = []

    def fake_submit(graph, client_id):
        has_restore = "21" in graph
        calls.append(has_restore)
        return None if has_restore else "pid-123"

    with patch.object(c, "_submit", side_effect=fake_submit), \
         patch.object(c, "_await_image", return_value=b"PNGDATA"):
        img, used = c._render_image("a face", 42, "flux.safetensors", 28, "cid", restore)

    assert img == b"PNGDATA" and used is False       # plain graph produced it
    assert calls == [True, False]                    # tried restore, then plain


def test_render_image_plain_when_no_restore():
    with patch.object(c, "_submit", return_value="pid-1") as sub, \
         patch.object(c, "_await_image", return_value=b"IMG"):
        img, used = c._render_image("x", 1, "flux.safetensors", 28, "cid", None)
    assert img == b"IMG" and used is False
    assert "21" not in sub.call_args[0][0]           # plain graph only


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
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
         patch.object(c, "_render_image", return_value=(b"PNG", False)), \
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None):
        clips = c.generate_clips(["prompt one", "prompt two"], n=2)

    assert len(clips) == 2
    assert ("wan", 0) in calls                       # clip 0 → wan succeeded
    assert ("svd", 0) not in calls                   # …so svd never touched it
    assert ("wan", 1) in calls and ("svd", 1) in calls   # clip 1 walked the chain


# ── FLUX.2 template stills (config/flux2_api.json) ────────────────────────────

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


def test_render_image_prefers_flux2_then_falls_back(monkeypatch, tmp_path):
    """FLUX.2 template render fails → the SAME seed goes to the FLUX.1 graph;
    the upgrade can never cost a clip."""
    monkeypatch.setattr(c, "FLUX2_TEMPLATE", _flux2_tpl(tmp_path))
    monkeypatch.delenv("RUFUS_FLUX2", raising=False)

    submitted = []

    def fake_submit(graph, client_id):
        submitted.append(graph)
        return f"pid{len(submitted)}"

    # First await (FLUX.2) fails, second (FLUX.1) succeeds
    awaits = iter([None, b"PNGBYTES"])
    with patch.object(c, "_submit", side_effect=fake_submit), \
         patch.object(c, "_await_image", side_effect=lambda pid: next(awaits)):
        img, used_restore = c._render_image("a coin", 123, "m.safetensors", 20,
                                            "cid", None)

    assert img == b"PNGBYTES"
    assert len(submitted) == 2
    # First submit was the template (has our substituted prompt)
    assert submitted[0]["1"]["inputs"]["text"] == "a coin"
    # Second submit was the FLUX.1 graph with the same seed
    assert submitted[1]["3"]["inputs"]["seed"] == 123


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
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
         patch.object(c, "_render_image", return_value=(b"PNG", False)), \
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None):
        clips = c.generate_clips(["a portrait of a banker"], n=1)

    assert len(clips) == 1
    assert ("wan", 0) in calls and ("hunyuan", 0) in calls


# ── Two-phase generation: stills first, then motion (24GB thrash fix) ─────────

def test_two_phase_all_renders_before_any_motion(monkeypatch, tmp_path):
    """Interleaving image→animate per clip forced ComfyUI to swap FLUX in and
    out for EVERY clip on a card that can't hold both models. All stills must
    now complete before the first motion call."""
    import wan_client

    order = []
    monkeypatch.setattr(wan_client, "enabled", lambda: True)
    monkeypatch.setattr(wan_client, "ready", lambda: (True, "test"))

    def wan_animate(png, clip, duration=8.0, idx=0, prompt=""):
        order.append(("animate", idx))
        clip.write_bytes(b"x" * 60_000)
        return True

    def fake_render(prompt, seed, model, steps, client_id, restore):
        order.append(("render", None))
        return (b"PNG", False)

    monkeypatch.setattr(wan_client, "animate_image", wan_animate)
    import svd_client
    monkeypatch.setattr(svd_client, "img2vid_enabled", lambda: False)

    freed = []
    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
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

    def fake_render(prompt, seed, model, steps, client_id, restore):
        calls["n"] += 1
        # Fail every render attempt for the SECOND prompt only
        if "SECOND" in prompt:
            return (None, False)
        return (b"PNG", False)

    def fake_kenburns(png, clip, duration=8.0, idx=0):
        clip.write_bytes(b"x" * 60_000)
        return True

    with patch.object(c, "is_available", return_value=True), \
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
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
         patch.object(c, "list_checkpoints", return_value=[]), \
         patch.object(c, "resolve_face_restore", return_value=None), \
         patch.object(c, "_render_image", return_value=(b"PNG", False)), \
         patch.object(c, "_fit_to_portrait", lambda b, p: p.write_bytes(b"i" * 25_000) or True), \
         patch.object(c, "_avg_hash", return_value=None), \
         patch.object(c, "_animate_to_clip", side_effect=fake_kenburns), \
         patch.object(c, "_free_comfy_memory", side_effect=lambda: freed.append(1)):
        clips = c.generate_clips(["p1", "p2"], n=2)
    assert len(clips) == 2
    assert freed == []
