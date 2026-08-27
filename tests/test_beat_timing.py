"""How long each picture is on screen, known before it is drawn.

THE ORDER THIS EXISTS TO MAKE POSSIBLE — the owner's call, and the right one:
record the voice first, measure it, show it to be chosen last. Timings come
from Whisper reading real audio, so until a take exists there is nothing to
measure, and every shot length before this was a guess from the script's word
count.

WHY ONE MEASUREMENT SERVES ALL THREE TAKES. The takes differ in pace — Kokoro's
speed runs 0.92 to 1.03 — so a forty-five second script lands between about
forty-four and forty-nine seconds. What does not change is how many pictures
the video wants: that is the script's beat count, and _max_shots
(audio_dur // 1.6) only starts binding below about twenty-six seconds.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import beat_timing  # noqa: E402


class _Info:
    def __init__(self, duration):
        self.duration = duration


def _fake_audio(monkeypatch, duration=45.0, ends=None):
    import audio_gen
    monkeypatch.setattr(audio_gen, "_transcribe",
                        lambda mp3: ([], _Info(duration)))
    monkeypatch.setattr(audio_gen, "_sentence_ends",
                        lambda segs: ends if ends is not None
                        else [i * 3.0 for i in range(1, 15)])


def test_the_spans_cover_the_whole_audio(tmp_path, monkeypatch):
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=45.0)
    spans = beat_timing.measure(mp3, "a script", 8)
    assert len(spans) == 8
    assert spans[0]["start"] == 0.0
    assert spans[-1]["end"] == pytest.approx(45.0, abs=0.01)


def test_the_spans_do_not_overlap_or_leave_gaps(tmp_path, monkeypatch):
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=45.0)
    spans = beat_timing.measure(mp3, "a script", 10)
    for a, b in zip(spans, spans[1:]):
        assert a["end"] == b["start"], (a, b)


def test_the_number_of_pictures_is_the_same_at_every_take_speed(tmp_path,
                                                                monkeypatch):
    """Kokoro's tone speeds span 0.92–1.03, so the same script lands roughly
    44–49s. Sixteen pictures is nowhere near the floor at either end, which is
    what lets one image set serve all three takes."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    counts = set()
    for duration in (44.0, 45.0, 49.0):
        _fake_audio(monkeypatch, duration=duration)
        counts.add(len(beat_timing.measure(mp3, "s", 16)))
    assert counts == {16}


def test_a_shot_on_the_floor_is_reported_before_anything_is_rendered(
        tmp_path, monkeypatch):
    """Asking for more pictures than the narration can carry is what produced
    the machine-gun run. Better to see it here than after forty minutes of
    drawing.

    The floor is passed explicitly. MIN_SEG is read from the active video
    format, so a test that relied on the ambient value asserted whichever
    format the previous test happened to leave set — it passed alone and failed
    in the suite. What is under test here is the flagging, not the constant."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=12.0)
    spans = beat_timing.measure(mp3, "s", 12)
    assert spans, "something must be measured to be flagged"
    longest = max(s["seconds"] for s in spans)
    assert beat_timing.too_short(spans, floor=longest), (
        "every shot at or under the floor must be named")
    assert beat_timing.too_short(spans, floor=0.01) == [], (
        "and none of them when the floor is below all of them")


def test_the_default_floor_is_the_renderer_s_own_minimum():
    """Not a number of this module's own: a second opinion about what counts as
    too short is a warning that disagrees with the thing it warns about."""
    import audio_gen
    spans = [{"index": 0, "start": 0, "end": audio_gen.MIN_SEG,
              "seconds": audio_gen.MIN_SEG},
             {"index": 1, "start": 0, "end": 99, "seconds": 99.0}]
    assert beat_timing.too_short(spans) == [0]


def test_a_comfortable_video_flags_nothing(tmp_path, monkeypatch):
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=45.0)
    assert beat_timing.too_short(beat_timing.measure(mp3, "s", 8)) == []


def test_a_missing_file_costs_the_numbers_not_the_stage(tmp_path, capsys):
    assert beat_timing.measure(tmp_path / "nope.mp3", "s", 4) == []
    assert "is not there" in capsys.readouterr().out


def test_whisper_falling_over_costs_the_numbers_not_the_stage(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """This is a nicety on a page. A model that will not load must cost the
    shot lengths, not the ability to choose pictures."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    import audio_gen
    monkeypatch.setattr(audio_gen, "_transcribe",
                        lambda mp3: (_ for _ in ()).throw(RuntimeError("no cuda")))
    assert beat_timing.measure(mp3, "s", 4) == []
    assert "could not measure" in capsys.readouterr().out


def test_silence_is_reported_rather_than_divided_up(tmp_path, monkeypatch,
                                                    capsys):
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=0.0)
    assert beat_timing.measure(mp3, "s", 4) == []
    assert "transcribed to nothing" in capsys.readouterr().out


def test_it_measures_the_same_way_the_render_cuts(tmp_path, monkeypatch):
    """Two functions that both decide where a cut goes are two functions that
    will disagree, and the one that disagrees silently is the one on the page —
    showing durations the render then does not use."""
    src = (Path(__file__).parent.parent / "scripts" / "beat_timing.py"
           ).read_text(encoding="utf-8")
    assert "audio_gen._plan_cuts" in src
    assert "audio_gen._sentence_ends" in src
