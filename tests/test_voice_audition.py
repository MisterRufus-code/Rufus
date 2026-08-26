"""The same line in every voice, so the narrator is chosen once.

/voice varies the TONE of one video's opening. This varies the narrator — and a
channel whose narrator changes every video has no narrator, because the voice is
the one thing a returning viewer recognises before they have read a word. So it
is the audio twin of /styles: a rare identity decision, made against one fixed
line, then left alone.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import voice_audition as va  # noqa: E402


def test_every_voice_reads_the_same_line():
    """A voice compared against a different sentence than the one before it is
    not being compared — half of what you hear is the writing."""
    assert va.SAMPLE.strip()
    assert len(va.SAMPLE.split()) < 25, "long enough to hear, short enough to compare"


def test_every_voice_carries_a_label_a_person_can_choose_between():
    """"am_adam" tells you nothing, and a sheet of eight of those is a page
    nobody opens twice."""
    for backend, voices in va.BACKENDS.items():
        for voice, label in voices:
            assert label and label != voice, f"{backend}/{voice}"
            assert "—" in label, f"{backend}/{voice} has no description"


def test_the_current_default_is_in_the_catalogue():
    """A sheet that cannot show you the voice you are already using cannot tell
    you whether changing is an improvement."""
    import tts_engine
    assert tts_engine.KOKORO_VOICE in {v for v, _l in va.KOKORO_VOICES}
    assert tts_engine.EDGE_VOICE in {v for v, _l in va.EDGE_VOICES}


def test_each_backend_knows_where_its_voice_is_read_from():
    for backend in va.BACKENDS:
        assert backend in va.VOICE_VAR


def test_the_pin_puts_the_callers_backend_back():
    """This runs in the dashboard process, which also renders videos. A helper
    that leaves RUFUS_TTS pointing somewhere else is a bug that shows up on the
    next render rather than here."""
    os.environ["RUFUS_TTS"] = "kokoro"
    os.environ["RUFUS_KOKORO_VOICE"] = "am_adam"
    try:
        with va._pinned_voice("edge", "en-GB-RyanNeural"):
            assert os.environ["RUFUS_TTS"] == "edge"
            assert os.environ["RUFUS_EDGE_VOICE"] == "en-GB-RyanNeural"
        assert os.environ["RUFUS_TTS"] == "kokoro"
        assert os.environ["RUFUS_KOKORO_VOICE"] == "am_adam"
    finally:
        os.environ.pop("RUFUS_TTS", None)
        os.environ.pop("RUFUS_KOKORO_VOICE", None)


def test_the_pin_moves_the_module_constant_too():
    """THE ENVIRONMENT IS NOT ENOUGH. tts_engine reads its voice into a module
    constant at import time, so a process auditioning eight voices would record
    eight files of whichever voice was set when it started."""
    import tts_engine
    before = tts_engine.KOKORO_VOICE
    with va._pinned_voice("kokoro", "bf_emma"):
        assert tts_engine.KOKORO_VOICE == "bf_emma"
    assert tts_engine.KOKORO_VOICE == before


def test_one_voice_that_will_not_speak_leaves_the_sheet_intact(tmp_path,
                                                               monkeypatch):
    """A Kokoro voice file that was never downloaded, an Edge name that has
    been retired — a missing row is a voice you cannot pick, which is the truth
    about it."""
    import tts_engine
    calls = {"n": 0}

    def _fake(text, out, tones=None, beats=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("no such voice")
        Path(out).write_bytes(b"x" * 2000)
    monkeypatch.setattr(tts_engine, "synthesize", _fake)
    monkeypatch.setattr(va, "audition_dir", lambda: tmp_path)
    done = va.build("kokoro", only=["am_adam", "am_michael", "af_heart"])
    assert len(done) == 2
    assert "am_michael" not in {d["voice"] for d in done}


def test_an_empty_file_is_not_offered_as_a_voice(tmp_path, monkeypatch):
    """A player that plays nothing is a choice a person cannot make, and a
    backend that fails quietly writes a zero-byte mp3 rather than raising."""
    import tts_engine
    monkeypatch.setattr(tts_engine, "synthesize",
                        lambda text, out, tones=None, beats=None:
                        Path(out).write_bytes(b""))
    monkeypatch.setattr(va, "audition_dir", lambda: tmp_path)
    assert va.build("kokoro", only=["am_adam"]) == []


def test_an_unknown_backend_says_so_rather_than_recording_nothing(capsys):
    assert va.build("elevenlabs") == []
    assert "no catalogue" in capsys.readouterr().out


def test_the_sample_is_recorded_once_per_voice(tmp_path, monkeypatch):
    import tts_engine
    seen = []
    monkeypatch.setattr(tts_engine, "synthesize",
                        lambda text, out, tones=None, beats=None:
                        (seen.append((text, Path(out).name)),
                         Path(out).write_bytes(b"x" * 2000)))
    monkeypatch.setattr(va, "audition_dir", lambda: tmp_path)
    va.build("kokoro", only=["am_adam", "bf_emma"])
    assert [n for _t, n in seen] == ["am_adam.mp3", "bf_emma.mp3"]
    assert {t for t, _n in seen} == {va.SAMPLE}, "same words, every voice"
