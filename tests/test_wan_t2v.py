"""Text-to-video, and the three things holding it together.

An image-to-video engine receives the world in its init frame. A text-to-video
engine builds every beat from noise having seen nothing, so ten beats are ten
unrelated worlds unless something forces them together. These tests cover the
forcing: an identical world lock, a derived seed lineage, and — for real object
continuity — chaining the next beat off the last frame of the previous one.

The engine itself stays inert until its template is exported, like every other
comfy-backed engine here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import wan_t2v_client as t2v


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("RUFUS_T2V", "RUFUS_T2V_CHAIN", "RUFUS_T2V_SEED",
                "RUFUS_T2V_W", "RUFUS_T2V_H", "RUFUS_T2V_FRAMES",
                "RUFUS_STILLS_ONLY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(t2v, "_run_seed", None)


# ------------------------------------------------------------- opt-in only

def test_off_by_default():
    """Text-to-video has real trade-offs — a run must not fall into it because
    a file happened to exist."""
    assert t2v.enabled() is False


def test_enabled_when_asked(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V", "1")
    assert t2v.enabled() is True


def test_stills_only_still_wins(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V", "1")
    monkeypatch.setenv("RUFUS_STILLS_ONLY", "1")
    assert t2v.enabled() is False


def test_inert_without_a_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_T2V_TEMPLATE", str(tmp_path / "absent.json"))
    ok, why = t2v.ready()
    assert ok is False
    assert "wan_t2v_api.json" in why


def test_the_missing_template_message_says_t2v_needs_its_own_models(monkeypatch, tmp_path):
    """Installing the I2V template does not give you T2V — different files.
    Someone who has just downloaded 35GB will assume otherwise."""
    monkeypatch.setenv("RUFUS_T2V_TEMPLATE", str(tmp_path / "absent.json"))
    _, why = t2v.ready()
    assert "DIFFERENT model files" in why


def test_generate_returns_false_without_a_template(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_T2V_TEMPLATE", str(tmp_path / "absent.json"))
    assert t2v.generate_clip("a beat", tmp_path / "out.mp4") is False


def test_it_is_not_named_animate_image():
    """The motion chain contracts every entry to receive a still. A t2v engine
    with that name could be dropped in and would break on the first call."""
    assert not hasattr(t2v, "animate_image")
    assert hasattr(t2v, "generate_clip")


# --------------------------------------------------------- 1. the world lock

def test_the_world_lock_is_byte_identical_across_beats():
    """The model keys off the exact token sequence — two beats describing the
    same world in different words are two different worlds to it."""
    world = t2v.build_world([], character="a hooded figure in a sepia cloak",
                            style="Flat 2D vector illustration")
    a = t2v._motion_prompt("shot one", world)
    b = t2v._motion_prompt("shot two", world)
    assert world in a and world in b
    assert a[len(a) - len(world):] == b[len(b) - len(world):]


def test_the_world_lock_carries_character_and_style():
    world = t2v.build_world([], character="a hooded Chronicler",
                            style="Flat 2D vector illustration")
    assert "Chronicler" in world
    assert "Flat 2D vector" in world
    assert "identical palette" in world


def test_an_empty_world_still_produces_a_usable_prompt():
    assert t2v.build_world([]) .strip()
    assert t2v._motion_prompt("a beat", "").strip()


def test_the_beat_text_leads_and_the_lock_follows():
    """Order matters: the subject has to arrive before the constraints."""
    world = t2v.build_world([], character="a hooded Chronicler")
    p = t2v._motion_prompt("a gold coin on a table", world)
    assert p.index("gold coin") < p.index("Chronicler")


def test_motion_direction_forbids_a_completing_action():
    """Clips are freeze-extended to fill their slot, so a finished gesture
    visibly stalls — the same constraint every other engine here carries."""
    p = t2v._motion_prompt("a beat")
    assert "never completes" in p
    assert "no cuts" in p


# ------------------------------------------------------ 2. the seed lineage

def test_the_run_seed_is_stable_within_a_process():
    assert t2v.run_seed() == t2v.run_seed()


def test_an_explicit_seed_makes_a_run_reproducible(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_SEED", "12345")
    monkeypatch.setattr(t2v, "_run_seed", None)
    assert t2v.run_seed() == 12345
    first = [t2v.beat_seed(i) for i in range(5)]

    monkeypatch.setattr(t2v, "_run_seed", None)
    assert [t2v.beat_seed(i) for i in range(5)] == first


def test_a_junk_seed_does_not_crash_the_run(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_SEED", "not a number")
    monkeypatch.setattr(t2v, "_run_seed", None)
    assert isinstance(t2v.run_seed(), int)


def test_beats_differ_from_each_other():
    """Derived, not identical — one seed across differing prompts produces
    near-duplicate compositions. Steadiness, not repetition."""
    seeds = [t2v.beat_seed(i) for i in range(10)]
    assert len(set(seeds)) == 10


def test_seeds_stay_in_comfyui_range():
    assert all(1 <= t2v.beat_seed(i) < 2**31 for i in range(50))


def test_different_runs_diverge(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_SEED", "111")
    monkeypatch.setattr(t2v, "_run_seed", None)
    a = [t2v.beat_seed(i) for i in range(4)]
    monkeypatch.setenv("RUFUS_T2V_SEED", "222")
    monkeypatch.setattr(t2v, "_run_seed", None)
    assert [t2v.beat_seed(i) for i in range(4)] != a


# --------------------------------------------------------- 3. frame chaining

def test_chaining_is_opt_in():
    assert t2v.chaining() is False


def test_chaining_switches_on(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_CHAIN", "1")
    assert t2v.chaining() is True


def test_the_last_frame_of_a_real_clip_is_extractable(tmp_path):
    """The only mechanism here that carries actual objects forward — if this
    silently produces nothing, chaining degrades to plain t2v and the video
    quietly loses its continuity."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x568:rate=30:duration=2",
         "-pix_fmt", "yuv420p", "-y", str(clip)],
        check=True, capture_output=True)

    png = tmp_path / "last.png"
    assert t2v.last_frame(clip, png) is True
    assert png.stat().st_size > 1_000


def test_a_missing_clip_fails_without_raising(tmp_path):
    assert t2v.last_frame(tmp_path / "nope.mp4", tmp_path / "out.png") is False


# ------------------------------------------------------------------ settings

def test_settings_report_what_the_run_will_actually_do(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_W", "720")
    monkeypatch.setenv("RUFUS_T2V_FRAMES", "61")
    cfg = t2v.settings()
    assert cfg["width"] == 720
    assert cfg["frames"] == 61
    assert "run_seed" in cfg and "chaining" in cfg


def test_portrait_by_default():
    cfg = t2v.settings()
    assert cfg["height"] > cfg["width"], "Shorts are vertical"


def test_no_ffmpeg_on_path_degrades_instead_of_raising(monkeypatch, tmp_path):
    """Caught by CI, which has no ffmpeg: subprocess raises FileNotFoundError
    rather than returning non-zero, so a bare `except TimeoutExpired` let it
    propagate. Chaining must degrade to plain t2v, never take the run down."""
    import subprocess as sp

    def boom(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(sp, "run", boom)
    monkeypatch.setattr(t2v.subprocess, "run", boom)

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    assert t2v.last_frame(clip, tmp_path / "out.png") is False
    assert t2v._finish(clip, tmp_path / "out.mp4", 5.0) is False


def test_a_seconds_based_node_gets_wans_own_framerate():
    """ComfyUI's packaged "Text to Video (Wan2.2)" node sizes its clip in
    width/height/DURATION and exposes no fps input. The generic fallback is 25
    (LTX's rate), so a 49-frame request silently became a 2-second clip instead
    of a 3-second one — the substitution "succeeded", so nothing said the
    length had changed."""
    import comfy_template
    import wan_t2v_client

    g = {"1": {"class_type": "WanT2V",
               "inputs": {"width": 1, "height": 1, "duration": 5.0}}}
    out = comfy_template.prepare(g, dims=(480, 832, 49),
                                 fps=wan_t2v_client.WAN_FPS)
    assert out["1"]["inputs"]["duration"] == 3     # 49/16, not 49/25


def test_a_node_that_states_its_own_fps_still_wins():
    """The hint is a fallback, not an override — a template that knows its own
    rate knows better than a per-engine constant."""
    import comfy_template
    g = {"1": {"class_type": "X",
               "inputs": {"width": 1, "height": 1, "duration": 5.0, "fps": 8}}}
    out = comfy_template.prepare(g, dims=(480, 832, 48), fps=16)
    assert out["1"]["inputs"]["duration"] == 6     # 48/8


def test_the_client_passes_its_framerate_through():
    from pathlib import Path
    src = Path("scripts/wan_t2v_client.py").read_text(encoding="utf-8")
    assert "fps=WAN_FPS" in src
