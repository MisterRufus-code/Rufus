"""Tests for the SVD image-to-video source. Pure-function + mocked-HTTP level —
no ComfyUI server needed to run these."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import svd_client as s


# ── Env knob ─────────────────────────────────────────────────────────────────────

def test_img2vid_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    assert s.img2vid_enabled() is True


def test_img2vid_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RUFUS_IMG2VID", "0")
    assert s.img2vid_enabled() is False


# ── RUFUS_STILLS_ONLY: single switch overriding all three motion engines ──────

def test_stills_only_off_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_STILLS_ONLY", raising=False)
    assert s._stills_only() is False


def test_stills_only_on_forces_img2vid_off_even_with_img2vid_enabled(monkeypatch):
    """RUFUS_STILLS_ONLY=1 must win even if RUFUS_IMG2VID=1 is explicitly set —
    it's the master override, not just another default."""
    monkeypatch.setenv("RUFUS_STILLS_ONLY", "1")
    monkeypatch.setenv("RUFUS_IMG2VID", "1")
    assert s.img2vid_enabled() is False


# ── SVD workflow graph ───────────────────────────────────────────────────────────

def test_svd_graph_is_json_serializable_and_wired():
    g = s._build_svd_graph("init.png", 999, "svd_xt.safetensors",
                           frames=25, fps=8, motion=100, steps=20)
    json.dumps(g)  # must not raise
    # KSampler samples the CFG-guided model with the SVD conditioning triplet
    ks = g["5"]["inputs"]
    assert ks["seed"] == 999 and ks["steps"] == 20
    assert ks["model"] == ["4", 0]          # VideoLinearCFGGuidance
    assert ks["positive"] == ["3", 0]       # SVD_img2vid_Conditioning outputs
    assert ks["negative"] == ["3", 1]
    assert ks["latent_image"] == ["3", 2]
    # Conditioning reads clip_vision + vae from the SVD loader, image from LoadImage
    cond = g["3"]["inputs"]
    assert cond["clip_vision"] == ["1", 1]
    assert cond["vae"] == ["1", 2]
    assert cond["init_image"] == ["2", 0]


def test_svd_graph_is_portrait_and_passes_params_through():
    g = s._build_svd_graph("x.png", 1, "custom.safetensors",
                           frames=14, fps=6, motion=42, steps=15)
    cond = g["3"]["inputs"]
    assert cond["width"] == s.SVD_W and cond["height"] == s.SVD_H
    assert s.SVD_H > s.SVD_W                    # portrait
    assert cond["video_frames"] == 14
    assert cond["fps"] == 6
    assert cond["motion_bucket_id"] == 42
    assert g["1"]["inputs"]["ckpt_name"] == "custom.safetensors"
    assert g["5"]["inputs"]["steps"] == 15


# ── Checkpoint preflight ─────────────────────────────────────────────────────────

def _obj_info_fixture(names):
    return {"ImageOnlyCheckpointLoader": {"input": {"required": {
        "ckpt_name": [names, {"tooltip": "checkpoint to load"}]}}}}


def test_parse_image_checkpoint_list_extracts_names():
    got = s._parse_image_checkpoint_list(_obj_info_fixture(["svd_xt.safetensors", "svd.ckpt"]))
    assert got == ["svd_xt.safetensors", "svd.ckpt"]


def test_parse_image_checkpoint_list_garbage_shapes_return_empty():
    assert s._parse_image_checkpoint_list({}) == []
    assert s._parse_image_checkpoint_list({"ImageOnlyCheckpointLoader": {}}) == []
    assert s._parse_image_checkpoint_list(
        {"ImageOnlyCheckpointLoader": {"input": {"required": {"ckpt_name": "oops"}}}}) == []


def test_svd_ready_true_when_model_present(monkeypatch):
    monkeypatch.delenv("COMFY_SVD_MODEL", raising=False)
    with patch.object(s, "list_svd_checkpoints", return_value=["svd_xt.safetensors"]):
        ok, why = s.svd_ready()
    assert ok is True


def test_svd_ready_false_when_model_missing():
    with patch.object(s, "list_svd_checkpoints", return_value=["something_else.safetensors"]):
        ok, why = s.svd_ready()
    assert ok is False
    assert "svd_xt.safetensors" in why      # the fix is named in the message


def test_svd_ready_fails_closed_when_list_unreadable():
    # Opposite of the FLUX preflight (fail-open): SVD is optional with a
    # guaranteed Ken Burns fallback, so "can't verify" must not spend submits.
    with patch.object(s, "list_svd_checkpoints", return_value=[]):
        ok, why = s.svd_ready()
    assert ok is False


# ── Engine resolution (comfy → diffusers → off) ──────────────────────────────────

def test_resolve_engine_disabled(monkeypatch):
    monkeypatch.setenv("RUFUS_IMG2VID", "0")
    engine, why = s.resolve_engine()
    assert engine is None
    assert "disabled" in why


def test_resolve_engine_prefers_comfy(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.delenv("RUFUS_IMG2VID_ENGINE", raising=False)
    with patch.object(s, "svd_ready", return_value=(True, "svd_xt loadable")):
        engine, _ = s.resolve_engine()
    assert engine == "comfy"


def test_resolve_engine_falls_back_to_diffusers(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.delenv("RUFUS_IMG2VID_ENGINE", raising=False)
    with patch.object(s, "svd_ready", return_value=(False, "no checkpoint")), \
         patch.object(s, "diffusers_ready", return_value=(True, "diffusers ok")):
        engine, _ = s.resolve_engine()
    assert engine == "diffusers"


def test_resolve_engine_none_when_neither_available(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.delenv("RUFUS_IMG2VID_ENGINE", raising=False)
    with patch.object(s, "svd_ready", return_value=(False, "no checkpoint")), \
         patch.object(s, "diffusers_ready", return_value=(False, "not installed")):
        engine, why = s.resolve_engine()
    assert engine is None
    assert "no checkpoint" in why and "not installed" in why   # both reasons surfaced


def test_resolve_engine_forced_comfy_does_not_fall_back(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.setenv("RUFUS_IMG2VID_ENGINE", "comfy")
    with patch.object(s, "svd_ready", return_value=(False, "no checkpoint")), \
         patch.object(s, "diffusers_ready", return_value=(True, "diffusers ok")):
        engine, why = s.resolve_engine()
    assert engine is None                       # forced engine missing → off, not swap
    assert "no checkpoint" in why


def test_resolve_engine_forced_diffusers_skips_comfy(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.setenv("RUFUS_IMG2VID_ENGINE", "diffusers")
    with patch.object(s, "svd_ready", side_effect=AssertionError("must not be called")), \
         patch.object(s, "diffusers_ready", return_value=(True, "diffusers ok")):
        engine, _ = s.resolve_engine()
    assert engine == "diffusers"


def test_resolve_engine_rejects_unknown_engine(monkeypatch):
    monkeypatch.delenv("RUFUS_IMG2VID", raising=False)
    monkeypatch.setenv("RUFUS_IMG2VID_ENGINE", "banana")
    engine, why = s.resolve_engine()
    assert engine is None
    assert "banana" in why


# ── Fail-safe animate ────────────────────────────────────────────────────────────

def test_animate_image_false_when_source_missing(tmp_path):
    # Nonexistent source PNG → init-image prep fails → False, no raise.
    assert s.animate_image(tmp_path / "nope.png", tmp_path / "out.mp4") is False


def test_animate_image_false_when_upload_fails(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    monkeypatch.setattr(s, "_prep_init_image", lambda a, b: True)
    monkeypatch.setattr(s, "_upload_image", lambda p: None)
    assert s.animate_image(src, tmp_path / "out.mp4") is False


def test_animate_image_diffusers_engine_false_when_generation_empty(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    monkeypatch.setattr(s, "_prep_init_image", lambda a, b: True)
    monkeypatch.setattr(s, "_run_diffusers", lambda *a, **k: [])
    # The comfy path (upload/submit) must never be touched on the diffusers engine
    monkeypatch.setattr(s, "_upload_image",
                        lambda p: (_ for _ in ()).throw(AssertionError("comfy path used")))
    assert s.animate_image(src, tmp_path / "out.mp4", engine="diffusers") is False


# ── Face-hint heuristic (SVD isn't face-aware, warps facial features) ────────────

def test_prompt_likely_shows_a_face_matches_portrait_and_face():
    assert s._prompt_likely_shows_a_face("A medium portrait of Richard Nixon speaking")
    assert s._prompt_likely_shows_a_face("An extreme close-up of a distressed face")
    assert s._prompt_likely_shows_a_face("Two portraits side by side")


def test_prompt_likely_shows_a_face_false_for_object_scene_prompts():
    assert not s._prompt_likely_shows_a_face(
        "A macro shot of ancient electrum Lydian stater coins on a stone counter")
    assert not s._prompt_likely_shows_a_face(
        "A wide establishing shot of a 1970s trading floor")
    assert not s._prompt_likely_shows_a_face("")
    assert not s._prompt_likely_shows_a_face(None)


def test_animate_image_skips_svd_for_face_prompt(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    # If the face short-circuit didn't fire, this would raise and fail the test.
    monkeypatch.setattr(s, "_prep_init_image",
                        lambda a, b: (_ for _ in ()).throw(AssertionError("SVD pipeline touched")))
    assert s.animate_image(src, tmp_path / "out.mp4",
                           prompt="A medium portrait of a worried banker") is False


def test_image_shows_a_face_false_for_missing_file(tmp_path):
    assert s._image_shows_a_face(tmp_path / "nope.png") is False


def test_image_shows_a_face_false_for_solid_color_object_image(tmp_path):
    # No face detector should ever fire on a plain object/scene image.
    from PIL import Image
    obj = tmp_path / "obj.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(obj))
    assert s._image_shows_a_face(obj) is False


def test_image_shows_a_face_fails_open_on_detector_error(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "img.png"
    Image.new("RGB", (100, 100), (1, 2, 3)).save(str(img))
    monkeypatch.setattr(s, "_load_face_cascade",
                        lambda: (_ for _ in ()).throw(RuntimeError("no cascade")))
    assert s._image_shows_a_face(img) is False


def test_animate_image_skips_svd_when_image_detector_finds_a_face(tmp_path, monkeypatch):
    # Prompt text gives no hint ("a banknote depicting a historical figure"),
    # but the pixel-level check catches the engraved portrait anyway —
    # regression test for the exact live bug (Mao's portrait warping on a
    # banknote clip the text heuristic missed).
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))

    monkeypatch.setattr(s, "_image_shows_a_face", lambda p: True)
    monkeypatch.setattr(s, "_prep_init_image",
                        lambda a, b: (_ for _ in ()).throw(AssertionError("SVD pipeline touched")))
    assert s.animate_image(src, tmp_path / "out.mp4",
                           prompt="a banknote depicting a historical figure") is False


def test_animate_image_no_prompt_runs_normal_path(tmp_path, monkeypatch):
    # Backward compatible: no prompt passed -> behaves exactly as before (the
    # existing missing-source-file test already covers this without prompt=).
    monkeypatch.setattr(s, "_prep_init_image", lambda a, b: True)
    monkeypatch.setattr(s, "_upload_image", lambda p: None)   # fails further down, not a face-skip
    from PIL import Image
    src = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920), (120, 90, 60)).save(str(src))
    assert s.animate_image(src, tmp_path / "out.mp4",
                           prompt="a macro shot of ancient coins") is False


def test_prep_init_image_outputs_svd_portrait(tmp_path):
    import random
    from PIL import Image

    rng = random.Random(7)
    src_img = Image.new("RGB", (1080, 1920))
    src_img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                     for _ in range(1080 * 1920)])
    src = tmp_path / "still.png"
    src_img.save(str(src))

    dst = tmp_path / "init.png"
    assert s._prep_init_image(src, dst) is True
    assert Image.open(dst).size == (s.SVD_W, s.SVD_H)


# ── Interpolation mode (mci warps fine detail — regression guard) ────────────

def test_assemble_uses_blend_interpolation_not_motion_compensated(tmp_path):
    """Live evidence: a newspaper-clipping shot (no face anywhere in frame)
    came out visibly warped/melted. mi_mode=mci estimates per-pixel motion
    vectors and warps along them — it mis-tracks fine, high-contrast detail
    (text, documents, ornate patterns), not just faces. mi_mode=blend
    cross-fades instead of estimating motion, so it can't produce that
    warping artifact. Must never regress back to mci/aobmc."""
    captured = {}

    def fake_run(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        result = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return result

    frames_dir = tmp_path
    (frames_dir / "pingpong.mp4").write_bytes(b"x" * 60_000)   # pass the size check
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)

    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(frames_dir, fps=8, out_path=out_path, duration=8.0)

    assert "mi_mode=blend" in captured["filter"]
    assert "mci" not in captured["filter"]
    assert "aobmc" not in captured["filter"]


# ── Encode-quality chain (intermediates near-lossless, warp-safe SVD params) ─────

def test_assemble_intermediates_are_near_lossless(tmp_path):
    """Both _assemble passes feed a THIRD encode (the final render) — lossy
    generations compound, which read on screen as 'the still is sharp but the
    video is mushy'. Intermediates must stay near-lossless (crf <= 16)."""
    crfs = []

    def fake_run(cmd, **kwargs):
        if "-crf" in cmd:
            crfs.append(int(cmd[cmd.index("-crf") + 1]))
        (tmp_path / "pingpong.mp4").write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)
    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=8, out_path=out_path, duration=8.0)

    assert len(crfs) == 2
    assert all(c <= 16 for c in crfs), f"intermediate crf too lossy: {crfs}"


def test_assemble_sharpens_after_upscale(tmp_path):
    # SVD's 576x1024 output is upscaled 1.9x — a mild luma unsharp recovers
    # the detail Lanczos smears. Chroma must stay untouched (halo safety).
    captured = {}

    def fake_run(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        (tmp_path / "pingpong.mp4").write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)
    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=8, out_path=out_path, duration=8.0)

    assert "unsharp" in captured["filter"]
    assert captured["filter"].index("lanczos") < captured["filter"].index("unsharp")


def test_svd_defaults_favor_low_warp_motion(monkeypatch):
    """motion_bucket_id is SVD's biggest warping lever — the default must stay
    conservative (<=80 base) and steps generous (>=30), per the channel owner's
    explicit quality-over-speed preference after repeated warp complaints."""
    import inspect
    src = inspect.getsource(s.animate_image)
    assert '"RUFUS_SVD_MOTION", "70"' in src
    assert '"RUFUS_SVD_STEPS", "30"' in src


def test_assemble_timeout_scales_with_frame_count(tmp_path):
    """Live bug: Wan's 81 frames (vs SVD's 25) need far longer to interpolate/
    upscale/sharpen than a fixed 480s allows — a real run showed Wan's
    server-side generation succeed, then this timeout threw the result away
    and fell back to SVD anyway. Timeout must scale with actual frame count,
    with 480s as the floor (still fine for SVD's smaller batches). _assemble
    makes TWO subprocess.run calls (interpolate pass, then a fixed-180s loop
    pass) — only the FIRST call's timeout is the one under test here."""
    timeouts = []

    def fake_run(cmd, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        (tmp_path / "pingpong.mp4").write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    # Simulate a Wan-sized batch: 81 frames on disk.
    for i in range(81):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")

    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)
    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=16, out_path=out_path, duration=8.0)

    assemble_timeout = timeouts[0]
    assert assemble_timeout >= 81 * 12
    assert assemble_timeout > 480   # must exceed the old fixed value


def test_assemble_timeout_has_a_floor_for_small_batches(tmp_path):
    # SVD's 25-frame batches must keep the same generous headroom as before.
    timeouts = []

    def fake_run(cmd, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        (tmp_path / "pingpong.mp4").write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    for i in range(25):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")

    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)
    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=8, out_path=out_path, duration=8.0)

    assert timeouts[0] == 480


# ── One-way assembly (ping_pong=False — Wan's directional-motion fix) ────────────

def test_assemble_one_way_uses_single_ffmpeg_pass_no_reverse(tmp_path):
    """Live glitch report: a one-way action (page turn) visibly undid itself
    every ping-pong cycle. ping_pong=False must not reverse/concat at all —
    single pass, no 'reverse' or 'concat' in the filter graph."""
    captured = {}
    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        out_path.write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    for i in range(81):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with patch.object(s.subprocess, "run", side_effect=fake_run):
        ok = s._assemble(tmp_path, fps=16, out_path=out_path, duration=8.0, ping_pong=False)

    assert ok is True
    assert call_count[0] == 1              # single pass — no separate loop pass
    assert "reverse" not in captured["filter"]
    assert "concat" not in captured["filter"]
    assert "tpad" in captured["filter"]     # freeze-extend instead of looping


def test_assemble_one_way_freezes_last_frame_for_remaining_duration(tmp_path):
    """81 frames @16fps = ~5.06s native; an 8s beat needs ~2.94s more —
    tpad's stop_duration must cover exactly that gap, not loop or cut."""
    captured = {}

    def fake_run(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        out_path.write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    for i in range(81):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=16, out_path=out_path, duration=8.0, ping_pong=False)

    # native_duration = 81/16 = 5.0625; stop_duration = 8.0 - 5.0625 = 2.9375
    assert "stop_duration=2.938" in captured["filter"] or "stop_duration=2.937" in captured["filter"]


def test_assemble_one_way_clamps_stop_duration_when_native_already_longer(tmp_path):
    """If the native clip is already longer than the requested duration,
    stop_duration must clamp to 0 (no negative tpad), and the output -t flag
    trims the excess instead."""
    captured = {}

    def fake_run(cmd, **kwargs):
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        out_path.write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    for i in range(81):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=16, out_path=out_path, duration=2.0, ping_pong=False)

    assert "stop_duration=0.000" in captured["filter"]


def test_assemble_ping_pong_default_unchanged_for_svd(tmp_path):
    """SVD's existing behavior (default ping_pong=True) must be untouched —
    this fix targets Wan's directional-motion problem specifically."""
    captured = {}
    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        if "-filter_complex" in cmd:
            captured["filter"] = cmd[cmd.index("-filter_complex") + 1]
        (tmp_path / "pingpong.mp4").write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    for i in range(25):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"x" * 60_000)

    with patch.object(s.subprocess, "run", side_effect=fake_run):
        s._assemble(tmp_path, fps=8, out_path=out_path, duration=8.0)   # ping_pong default

    assert call_count[0] == 2               # interpolate pass + loop pass, as before
    assert "reverse" in captured["filter"]
    assert "concat" in captured["filter"]
