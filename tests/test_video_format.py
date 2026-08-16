"""One format was written into seven places, and none of them mentioned the
others.

1080×1920 lived in audio_gen, in Short.tsx and in sd_client. A 115-word cap
lived in script_standards.json. A 10–30 beat range lived in main. A 1.6-second
minimum shot lived in audio_gen. A 180-second "broken render" ceiling lived in
qc_check. Nothing was wrong with any of them — they simply all encoded ONE
format, so asking for a second was not asking for a setting. It was asking
seven numbers to move together, which they can only do if they live in one
place first.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import video_format  # noqa: E402


def test_short_is_the_default_and_is_unchanged():
    """The existing channel must not move because a second format exists."""
    p = video_format.profile("short")
    assert (p["width"], p["height"]) == (1080, 1920)
    assert (p["words_min"], p["words_max"]) == (80, 115)
    assert (p["beats_min"], p["beats_max"]) == (10, 30)
    assert p["min_seg_s"] == 1.6
    assert p["qc_max_s"] == 180.0
    assert video_format.name() == "short"


def test_long_is_landscape_and_calmer():
    p = video_format.profile("long")
    assert (p["width"], p["height"]) == (1920, 1080)
    assert p["min_seg_s"] > video_format.profile("short")["min_seg_s"]
    assert p["qc_max_s"] > 900


def test_the_beat_rule_is_one_rule_with_two_sets_of_constants():
    """A 105-word Short and a 1,350-word explainer are the same arithmetic."""
    assert video_format.target_beats(105, "short") == 21
    assert video_format.target_beats(1350, "long") == 150


def test_the_floors_and_ceilings_hold():
    assert video_format.target_beats(3, "short") == 10
    assert video_format.target_beats(100_000, "short") == 30
    assert video_format.target_beats(10, "long") == 40
    assert video_format.target_beats(100_000, "long") == 220


def test_an_unknown_format_is_loud_and_falls_back(monkeypatch, capsys):
    """A typo that silently changed the aspect ratio of a nine-minute render
    would be an expensive way to learn about it."""
    monkeypatch.setenv("RUFUS_FORMAT", "vertical-ish")
    assert video_format.name() == "short"
    assert "is not a known format" in capsys.readouterr().out


def test_an_empty_setting_is_the_default_and_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_FORMAT", "  ")
    assert video_format.name() == "short"
    assert capsys.readouterr().out == ""


# ── the readers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt,dims,min_seg,qc_max", [
    ("short", (1080, 1920), 1.6, 180.0),
    ("long", (1920, 1080), 2.5, 1500.0),
])
def test_every_reader_follows_the_profile(monkeypatch, fmt, dims, min_seg, qc_max):
    """The point of the exercise: the seven numbers move together, or the
    long-form render comes out cropped to portrait with a 'broken render'
    warning on a video that is exactly as long as it was asked to be."""
    monkeypatch.setenv("RUFUS_FORMAT", fmt)
    import audio_gen, qc_check, sd_client
    for mod in (audio_gen, qc_check, sd_client):
        importlib.reload(mod)
    assert (audio_gen.W, audio_gen.H) == dims
    assert (sd_client.OUT_W, sd_client.OUT_H) == dims
    assert audio_gen.MIN_SEG == min_seg
    assert qc_check.MAX_DUR == qc_max


def test_the_frame_fit_is_no_longer_named_for_one_shape():
    """`_fit_to_portrait` was never portrait-specific except in its name, and
    a function called that is one nobody would think to call for a landscape
    render."""
    import comfy_client
    assert hasattr(comfy_client, "_fit_to_frame")
    assert not hasattr(comfy_client, "_fit_to_portrait")


def test_the_run_can_say_which_shape_it_is_making():
    assert "1080×1920" in video_format.describe()


# ── the frame is not the only thing that changes shape ──────────────────────

@pytest.mark.parametrize("fmt,size,margin", [
    ("short", 140, 600),
    ("long", 58, 70),
])
def test_captions_follow_the_format(monkeypatch, fmt, size, margin):
    """140px and MarginV 600 are right for a phone at arm's length with the
    Shorts UI covering the bottom fifth. On a 1080-tall landscape frame the
    same numbers are 13% of the height with the words halfway up the
    picture."""
    monkeypatch.setenv("RUFUS_FORMAT", fmt)
    import audio_gen
    importlib.reload(audio_gen)
    assert audio_gen.FONTSIZE == size
    assert audio_gen.MARGIN_V == margin


def test_the_caption_is_a_sane_share_of_the_frame_in_both():
    """The check that would have caught this by arithmetic instead of by
    watching a video: a caption over a tenth of the frame height is a caption
    that IS the video."""
    for fmt in ("short", "long"):
        p = video_format.profile(fmt)
        share = p["caption_size"] / p["height"]
        assert 0.04 <= share <= 0.09, (fmt, round(share, 3))


def test_the_caption_sits_inside_the_frame_in_both():
    for fmt in ("short", "long"):
        p = video_format.profile(fmt)
        assert p["caption_margin_v"] + p["caption_size"] < p["height"], fmt


def test_an_insert_never_swallows_the_frame():
    """460 of 1080 is 43% of the width — fine on a phone, and on a landscape
    frame it would cover most of the shot it is meant to annotate."""
    for fmt in ("short", "long"):
        p = video_format.profile(fmt)
        assert p["insert_w"] / p["width"] < 0.5, fmt
