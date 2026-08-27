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


# ── the words under each shot ───────────────────────────────────────────────
#
# THE GAP THE OWNER HAS BEEN DESCRIBING SINCE THE FIRST GALLERY. Image prompts
# were planned from _split_beats — a split of the SCRIPT TEXT. The renderer
# cuts on sentence boundaries found in the AUDIO. Nothing ever made those two
# agree, so shot 7's picture was drawn for the seventh chunk of the text while
# shot 7 on screen covered whatever was said between the sixth and seventh cut.
# _build_sd_prompts' own docstring promised "the on-screen image tracks the
# voice-over"; it was true only by luck.

class _W:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Seg:
    def __init__(self, words):
        self.words = words


def _spoken(monkeypatch, duration, words, ends=None):
    import audio_gen
    segs = [_Seg([_W(w, s, e) for w, s, e in words])]
    monkeypatch.setattr(audio_gen, "_transcribe",
                        lambda mp3: (segs, _Info(duration)))
    monkeypatch.setattr(audio_gen, "_sentence_ends",
                        lambda s: ends if ends is not None else [10.0, 20.0])


def test_each_shot_carries_the_words_spoken_over_it(tmp_path, monkeypatch):
    """Say "cucumber" at 12.4s and the picture covering 12.4s is drawn from a
    sentence containing cucumber."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _spoken(monkeypatch, 30.0, [
        ("a", 1.0, 1.4), ("computer", 2.0, 2.6),
        ("cucumber", 12.0, 12.8),
        ("later", 25.0, 25.5)])
    shots = beat_timing.spoken_shots(mp3, 3)
    assert len(shots) == 3
    assert "computer" in shots[0]["text"]
    assert "cucumber" in shots[1]["text"]
    assert "later" in shots[2]["text"]


def test_a_word_straddling_a_cut_goes_where_it_mostly_is(tmp_path, monkeypatch):
    """Assigning by start would put a word that is almost entirely under the
    next picture with the previous one."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _spoken(monkeypatch, 30.0, [("straddler", 9.6, 10.9)], ends=[10.0, 20.0])
    shots = beat_timing.spoken_shots(mp3, 3)
    joined = [s["text"] for s in shots]
    assert joined.count("straddler") == 1, "it belongs to exactly one shot"
    assert "straddler" in joined[1], "its midpoint is past the cut"


def test_a_shot_sitting_in_a_pause_is_named(tmp_path, monkeypatch, capsys):
    """It has nothing to depict, and a prompt written from nothing is a
    picture that cannot match the narration because there is none."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _spoken(monkeypatch, 30.0, [("only", 1.0, 1.5)])
    shots = beat_timing.spoken_shots(mp3, 3)
    assert shots[2]["text"] == ""
    assert "no words under them" in capsys.readouterr().out


def test_no_word_timings_still_returns_the_spans(tmp_path, monkeypatch, capsys):
    """Fail-open: without words you lose the precision, not the pictures."""
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"x")
    _fake_audio(monkeypatch, duration=30.0)      # segments with no .words
    shots = beat_timing.spoken_shots(mp3, 3)
    assert len(shots) == 3
    assert all("text" not in s or s["text"] == "" for s in shots)
