"""Redrawing one beat, and rebuilding the video around it.

WHY THIS EXISTS. A gallery is usually right about most of its frames and wrong
about one: a contact sheet where a scene should be, a figure with no arms, the
wrong object on the table. Fixing that by re-running the pipeline costs the
script, the voice and thirteen minutes of GPU to change one picture — so it
does not get done, and the video ships with the bad frame in it.

The property every test here is really defending: a re-cut may change WHICH
picture is on screen and must never change WHEN. Cuts are placed from Whisper's
word timings, so re-synthesizing the voice would produce audio a few
milliseconds different, and every cut in the video would move. "I redrew beat
7" would silently reshuffle the other nine.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client  # noqa: E402
import db_manager  # noqa: E402
import recut  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A finished run's debug folder: stills, prompts and a voiceover."""
    import paths
    root = tmp_path / "debug"
    d = root / "20260820-abc"
    d.mkdir(parents=True)
    for n in ("01", "02", "03"):
        (d / f"{n}.png").write_bytes(b"\x89PNG" + n.encode())
        (d / f"{n}.txt").write_text(f"FLUX PROMPT:\nbeat {n} scene\n",
                                    encoding="utf-8")
    (d / "voiceover.mp3").write_bytes(b"ID3voice")
    monkeypatch.setattr(paths, "debug_root", lambda: root)
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    vid = db_manager.save_video(niche="money_history", script_hook="h",
                                scene_desc="s", video_file=str(tmp_path / "old.mp4"),
                                score=8, run_id="20260820-abc",
                                script_full="A. B. C.")
    return {"dir": d, "video_id": vid, "root": root}


# ── which frames, in which order ─────────────────────────────────────────────

def test_the_frames_come_back_in_narration_order(run):
    assert [p.name for p in recut.stills_for(run["dir"])] == \
        ["01.png", "02.png", "03.png"]


def test_a_regenerated_frame_does_not_jump_to_the_end(run):
    """Sorting by mtime would put the newest file last, and the newest file is
    precisely the one that was just redrawn. The video would end on beat 2."""
    import os, time
    p = run["dir"] / "01.png"
    os.utime(p, (time.time() + 500, time.time() + 500))
    assert recut.stills_for(run["dir"])[0].name == "01.png"


def test_sub_frames_sort_inside_their_own_beat(run):
    """A beat rendered as several stills writes 01.png, 01a.png, 01b.png —
    lexical order IS narration order, and it has to stay that way."""
    for suffix in ("a", "b"):
        (run["dir"] / f"01{suffix}.png").write_bytes(b"\x89PNG")
    assert [p.name for p in recut.stills_for(run["dir"])][:3] == \
        ["01.png", "01a.png", "01b.png"]


def test_a_stray_png_is_not_mistaken_for_a_beat(run):
    (run["dir"] / "contact_sheet.jpg").write_bytes(b"x")
    (run["dir"] / "gate.png").write_bytes(b"x")
    assert all(p.stem[:2].isdigit() for p in recut.stills_for(run["dir"]))


# ── what a re-cut would use ──────────────────────────────────────────────────

def test_a_usable_run_reports_its_frames_and_voice(run):
    p = recut.plan(run["video_id"])
    assert p["ok"] is True
    assert len(p["frames"]) == 3
    assert p["voiceover"].endswith("voiceover.mp3")
    assert "warning" not in p


def test_a_missing_voiceover_is_a_warning_not_a_refusal(run):
    """It still works — the voice is regenerated. But that moves the cuts, and
    discovering it in the finished file is worse than being told."""
    (run["dir"] / "voiceover.mp3").unlink()
    p = recut.plan(run["video_id"])
    assert p["ok"] is True
    assert "cuts may land differently" in p["warning"]


def test_a_run_with_no_frames_says_so(run):
    for png in run["dir"].glob("*.png"):
        png.unlink()
    p = recut.plan(run["video_id"])
    assert p["ok"] is False and "no numbered stills" in p["why"]


def test_a_video_with_no_run_id_cannot_be_recut(run, tmp_path):
    vid = db_manager.save_video(niche="x", script_hook="h", scene_desc="s",
                                video_file="a.mp4", score=8,
                                script_full="A.")
    p = recut.plan(vid)
    assert p["ok"] is False and "no run_id" in p["why"]


def test_a_video_with_no_script_cannot_be_recut(run):
    """The render builds its captions from the script. A re-cut without one
    would produce a silent-looking video rather than fail."""
    with db_manager._conn() as c:
        c.execute("UPDATE videos SET script_full='' WHERE id=?",
                  (run["video_id"],))
    p = recut.plan(run["video_id"])
    assert p["ok"] is False and "no saved script" in p["why"]


def test_an_unknown_video_id_is_a_reason_not_a_traceback(run):
    # Takes the fixture for its isolated database. Without it this reads the
    # repo's own rufus.db, which predates several columns and raises — which
    # is how recut.py was found to be the one entry point not calling
    # init_db() before its first read.
    p = recut.plan(999999)
    assert p["ok"] is False and "no video" in p["why"]


# ── the prompt sidecar ───────────────────────────────────────────────────────

def test_the_prompt_is_read_back_without_its_header(run):
    assert comfy_client.read_beat_prompt(run["dir"] / "01.txt") == "beat 01 scene"


def test_an_edited_prompt_is_written_in_the_runs_own_format(run):
    """The sidecar is the run's record of what produced the picture. A stale
    one makes the debug folder lie about its own contents."""
    txt = run["dir"] / "01.txt"
    assert comfy_client.write_beat_prompt(txt, "a completely different shot")
    assert comfy_client.read_beat_prompt(txt) == "a completely different shot"
    assert txt.read_text(encoding="utf-8").startswith("FLUX PROMPT:")


def test_a_missing_sidecar_is_empty_not_an_exception(run):
    assert comfy_client.read_beat_prompt(run["dir"] / "99.txt") == ""


# ── the render is never allowed to destroy a working video ───────────────────

def test_a_failed_render_leaves_the_row_pointing_at_the_old_file(run, monkeypatch):
    import remotion_renderer, audio_gen
    monkeypatch.setattr(remotion_renderer, "render",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no node")))
    monkeypatch.setattr(audio_gen, "render",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    before = db_manager.video_by_id(run["video_id"])["video_file"]
    assert recut.recut(run["video_id"]) is None
    assert db_manager.video_by_id(run["video_id"])["video_file"] == before


def test_a_successful_render_repoints_the_row(run, monkeypatch):
    import remotion_renderer
    new = run["root"].parent / "short_new.mp4"
    new.write_bytes(b"mp4")
    monkeypatch.setattr(remotion_renderer, "render", lambda *a, **k: new)
    assert recut.recut(run["video_id"]) == new
    assert db_manager.video_by_id(run["video_id"])["video_file"] == str(new)


def test_the_existing_voiceover_is_handed_to_the_renderer(run, monkeypatch):
    """THE SAFETY PROPERTY. Same bytes in, same word timings out, same cuts."""
    import remotion_renderer
    seen = {}

    def _fake(script, frames, out_dir, **kw):
        seen.update(kw)
        out = run["root"].parent / "o.mp4"
        out.write_bytes(b"mp4")
        return out

    monkeypatch.setattr(remotion_renderer, "render", _fake)
    recut.recut(run["video_id"])
    assert seen["voice_path"] is not None
    assert Path(seen["voice_path"]).name == "voiceover.mp3"


# ── redrawing one frame ──────────────────────────────────────────────────────

def test_a_failed_regen_leaves_the_frame_untouched(run, monkeypatch):
    monkeypatch.setattr(comfy_client, "is_available", lambda: True)
    monkeypatch.setattr(comfy_client, "_render_image", lambda *a, **k: None)
    png = run["dir"] / "01.png"
    before = png.read_bytes()
    assert comfy_client.render_one_beat("a shot", png) is False
    assert png.read_bytes() == before


def test_an_empty_prompt_renders_nothing(run):
    assert comfy_client.render_one_beat("   ", run["dir"] / "01.png") is False


def test_comfyui_being_down_is_a_reason_not_a_crash(run, monkeypatch, capsys):
    monkeypatch.setattr(comfy_client, "is_available", lambda: False)
    assert comfy_client.render_one_beat("a shot", run["dir"] / "01.png") is False
    assert "not reachable" in capsys.readouterr().out


def test_a_half_written_frame_never_replaces_a_good_one(run, monkeypatch):
    """_fit_to_frame failing partway through would otherwise leave a truncated
    png where a working still used to be."""
    monkeypatch.setattr(comfy_client, "is_available", lambda: True)
    monkeypatch.setattr(comfy_client, "_render_image", lambda *a, **k: b"raw")
    monkeypatch.setattr(comfy_client, "_fit_to_frame", lambda raw, out: False)
    png = run["dir"] / "01.png"
    before = png.read_bytes()
    assert comfy_client.render_one_beat("a shot", png) is False
    assert png.read_bytes() == before
    assert not (run["dir"] / "01.regen.png").exists(), "staging file left behind"
