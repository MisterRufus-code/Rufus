"""The subtitle look, chosen before a render rather than compiled into it.

The owner asked for "a few subtitles styles to choose from before the
rendering". The captions are half of what a Short IS — one word at a time in
capitals with a colour pop is a genre — and the numbers behind them lived in
video_format.PROFILES next to the frame size and the QC bounds, where changing
them changes every future video at once.

Two things have to stay true for this to be a choice rather than a trap: the
default must still be exactly what the channel already ships, and a style must
mean the same thing in a nine-minute landscape video as in a sixty-second
vertical one.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import caption_styles as cs  # noqa: E402

SHORT = {"caption_size": 140, "caption_margin_v": 600,
         "caption_words": 1, "caption_upper": True}
LONG = {"caption_size": 58, "caption_margin_v": 70,
        "caption_words": 4, "caption_upper": False}


def test_the_default_changes_nothing(monkeypatch):
    """THE ONE THAT MATTERS. Every unattended cron render goes through here
    with nothing set, and a new picker that quietly restyles them all would be
    a channel redesigned by a feature nobody chose."""
    monkeypatch.delenv(cs.ENV_VAR, raising=False)
    for profile, height in ((SHORT, 1920), (LONG, 1080)):
        r = cs.resolve(profile, frame_height=height)
        assert r["words"] == profile["caption_words"]
        assert r["upper"] == profile["caption_upper"]
        assert r["size"] == profile["caption_size"]
        assert r["margin_v"] == profile["caption_margin_v"]
        assert r["enabled"] is True


def test_the_default_is_the_format_and_not_one_of_the_looks():
    """A preset that hard-codes one word in capitals would be right for a
    Short and wrong for nine minutes on a television — which is the whole
    reason video_format carries per-format caption numbers."""
    assert cs.DEFAULT == "format"
    assert "words" not in cs.PRESETS["format"]
    assert "height_pct" not in cs.PRESETS["format"]


def test_a_style_means_the_same_thing_in_both_formats():
    """140px is 7% of a 1920-tall frame and 13% of a 1080-tall one. Sizes are
    shares of the frame for exactly that reason — the alternative is a preset
    that is a different design depending on which shape it lands in."""
    a = cs.resolve(SHORT, "word_pop", frame_height=1920)
    b = cs.resolve(LONG, "word_pop", frame_height=1080)
    assert a["size"] / 1920 == pytest.approx(b["size"] / 1080, abs=0.002)
    assert a["margin_v"] / 1920 == pytest.approx(b["margin_v"] / 1080, abs=0.002)
    assert a["words"] == b["words"] == 1


def test_scaling_the_formats_own_number_would_have_been_wrong():
    """Long-form ALREADY ships the broadcast look. A preset multiplying the
    format's 58px by 0.45 to "make it broadcast" lands at 26px — unreadable,
    and produced by asking for the thing it was already doing."""
    r = cs.resolve(LONG, "broadcast", frame_height=1080)
    assert r["size"] >= 50


def test_the_channel_accent_survives_a_style_that_is_not_about_colour():
    """accent_color is this channel's identity, set per niche. A subtitle
    preset has no business overwriting it, and only the two whose whole point
    is colour say anything about it."""
    for key in ("format", "word_pop", "phrase", "boxed"):
        assert cs.resolve(SHORT, key, frame_height=1920)["accent"] is None
    assert cs.resolve(SHORT, "gold", frame_height=1920)["accent"] is not None


def test_an_unknown_style_falls_back_loudly(monkeypatch, capsys):
    """Silently is how a video gets rendered in a look nobody picked and
    nobody notices until upload."""
    monkeypatch.setenv(cs.ENV_VAR, "hormozi")
    assert cs.name() == cs.DEFAULT
    assert "hormozi" in capsys.readouterr().out


def test_every_preset_says_what_it_costs_you():
    """A dropdown of five words is a guess every time. What separates these is
    what they do to a viewer, so each has to carry the sentence that says so."""
    for key, pre in cs.PRESETS.items():
        assert pre.get("label"), key
        assert len(pre.get("blurb", "")) > 60, key


def test_turning_the_words_off_writes_a_file_with_no_words(tmp_path,
                                                           monkeypatch):
    """"none" has to reach the burned frames, not just the config. An option
    that is honoured everywhere except where it matters is this repo's oldest
    bug."""
    monkeypatch.setenv(cs.ENV_VAR, "none")
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    import audio_gen
    importlib.reload(audio_gen)
    try:
        word = type("W", (), {"word": "money", "start": 0.0, "end": 0.6})()
        seg = type("S", (), {"words": [word]})()
        ass = tmp_path / "out.ass"
        audio_gen.build_ass([seg], ass, 1.0)
        text = ass.read_text(encoding="utf-8")
        assert "[Events]" in text, "the file is still a valid ASS file"
        assert "Dialogue:" not in text
        assert "money" not in text.lower()
    finally:
        monkeypatch.delenv(cs.ENV_VAR, raising=False)
        importlib.reload(audio_gen)


def test_a_chosen_style_reaches_the_burned_style_line(tmp_path, monkeypatch):
    """The look half of a preset is written into the ASS style line and burned
    by FFmpeg. Asserted on the bytes rather than on the config dict, because
    the config dict was never the thing that was going wrong."""
    monkeypatch.setenv(cs.ENV_VAR, "boxed")
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    import audio_gen
    importlib.reload(audio_gen)
    try:
        words = [type("W", (), {"word": w, "start": i * 0.5,
                                "end": i * 0.5 + 0.4})()
                 for i, w in enumerate(["one", "two", "three", "four"])]
        seg = type("S", (), {"words": words})()
        ass = tmp_path / "boxed.ass"
        audio_gen.build_ass([seg], ass, 3.0)
        style = next(ln for ln in ass.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("Style: Default"))
        # BorderStyle 3 is the opaque panel; 1 is an outline. It is the field
        # after the four zeros that follow ScaleX/ScaleY/Spacing/Angle.
        assert ",3," in style, style
        assert audio_gen.CLUSTER_SIZE == 2
        assert "ONE TWO" in ass.read_text(encoding="utf-8")
        assert "\\t(" not in ass.read_text(encoding="utf-8"), (
            "a style with no pop should not pay to animate every frame")
    finally:
        monkeypatch.delenv(cs.ENV_VAR, raising=False)
        importlib.reload(audio_gen)
