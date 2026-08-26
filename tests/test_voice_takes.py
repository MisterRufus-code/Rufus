"""Three reads of the hook, and only the hook.

Audio is the one thing on this pipeline's choose-from-several list that cannot
be skimmed — it plays at one times speed. Three full forty-five-second takes is
two and a half minutes of attention; three eight-second hooks is twenty-four
seconds, and if the opening read lands the rest follows it.

The VOICE does not vary and should not: a channel whose narrator changes every
video has no narrator. What varies is the tone beat 0 is read in — the same
lever that already sizes that beat's pauses and grades its picture.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import voice_takes as vt  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


def test_the_take_is_the_opening_line():
    script = "\n\nYou checked your portfolio today.\nThat is the problem.\n"
    assert vt.hook_of(script) == "You checked your portfolio today."


def test_a_script_with_nothing_in_it_has_no_hook():
    assert vt.hook_of("   \n\n  ") == ""


def test_the_directors_own_choice_is_offered_first(monkeypatch):
    """A set that never offers what the pipeline would have done by itself
    turns a choice into a forced change — and the director read the actual
    beat, which a fixed list cannot."""
    import edit_director
    import emotional_map
    import main as rufus_main
    monkeypatch.setattr(rufus_main, "_split_beats", lambda *a, **k: ["a", "b"])
    monkeypatch.setattr(edit_director, "direct", lambda beats: {"beats": [
        {"tone": "weight"}, {"tone": "neutral"}]})
    monkeypatch.setattr(emotional_map, "tones_from_plan",
                        lambda plan, n: ["weight", "neutral"])
    assert vt.tones_for("a script", 3)[0] == "weight"


def test_the_set_spans_the_range_without_repeating(monkeypatch):
    import edit_director
    monkeypatch.setattr(edit_director, "direct",
                        lambda beats: (_ for _ in ()).throw(RuntimeError("x")))
    tones = vt.tones_for("a script", 3)
    assert len(tones) == len(set(tones)) == 3


def test_a_silent_director_still_produces_a_choice(monkeypatch, capsys):
    """Fail-open: three reads is still a choice, it just no longer leads with
    the one the pipeline would have picked."""
    import edit_director
    monkeypatch.setattr(edit_director, "direct",
                        lambda beats: (_ for _ in ()).throw(RuntimeError("down")))
    assert len(vt.tones_for("s", 3)) == 3
    assert "did not answer" in capsys.readouterr().out


def _script(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("You checked your portfolio today.\nThat is the problem.",
                 encoding="utf-8")
    return p


def test_each_read_is_synthesized_at_its_own_tone(db, tmp_path, monkeypatch):
    """Through synthesize's per-beat path rather than around it, so the take a
    person hears is produced the way the render will produce it."""
    import tts_engine
    seen = []

    def _fake(text, out, tones=None, beats=None):
        seen.append((text, tones))
        Path(out).write_bytes(b"x" * 2000)
    monkeypatch.setattr(tts_engine, "synthesize", _fake)
    monkeypatch.setattr(vt, "takes_dir", lambda sid: tmp_path / "takes")
    monkeypatch.setattr(vt, "tones_for", lambda s, n: ["tension", "curiosity"])

    saved = vt.build(str(_script(tmp_path)), set_id=1, n=2)
    assert len(saved) == 2
    assert [t for _text, t in seen] == [["tension"], ["curiosity"]]
    assert all(text == "You checked your portfolio today." for text, _ in seen)


def test_one_tone_that_will_not_speak_does_not_cost_the_others(
        db, tmp_path, monkeypatch):
    import tts_engine
    calls = {"n": 0}

    def _fake(text, out, tones=None, beats=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend choked")
        Path(out).write_bytes(b"x" * 2000)
    monkeypatch.setattr(tts_engine, "synthesize", _fake)
    monkeypatch.setattr(vt, "takes_dir", lambda sid: tmp_path / "takes")
    monkeypatch.setattr(vt, "tones_for", lambda s, n: ["a", "b", "c"])
    assert len(vt.build(str(_script(tmp_path)), set_id=1, n=3)) == 2


def test_an_empty_audio_file_is_not_offered_as_a_read(db, tmp_path, monkeypatch):
    """A player that plays nothing is a choice a person cannot make, and a
    backend that fails quietly writes a zero-byte mp3 rather than raising."""
    import tts_engine
    monkeypatch.setattr(tts_engine, "synthesize",
                        lambda text, out, tones=None, beats=None:
                        Path(out).write_bytes(b""))
    monkeypatch.setattr(vt, "takes_dir", lambda sid: tmp_path / "takes")
    monkeypatch.setattr(vt, "tones_for", lambda s, n: ["a", "b"])
    assert vt.build(str(_script(tmp_path)), set_id=1, n=2) == []


# ── the choice ───────────────────────────────────────────────────────────────

def _three(db, set_id=1):
    return [db.save_voice_take(set_id=set_id, channel="c", topic="T",
                               tone=tone, text="hook", path=f"/{tone}.mp3")
            for tone in ("curiosity", "tension", "revelation")]


def test_choosing_one_read_rejects_its_siblings(db):
    ids = _three(db)
    got = db.choose_voice_take(ids[2])
    assert got["tone"] == "revelation"
    rows = {t["id"]: t["status"] for t in db.voice_takes(set_id=1)}
    assert rows[ids[2]] == "chosen"
    assert rows[ids[0]] == rows[ids[1]] == "rejected"


def test_a_second_click_cannot_re_decide_the_read(db):
    ids = _three(db)
    assert db.choose_voice_take(ids[0])
    assert db.choose_voice_take(ids[1]) is None


def test_another_sets_reads_are_not_siblings(db):
    a = _three(db, set_id=1)
    b = _three(db, set_id=2)
    db.choose_voice_take(a[0])
    rows = {t["id"]: t["status"] for t in db.voice_takes(set_id=2)}
    assert set(rows.values()) == {"pending"}
    assert b  # the other set is untouched


def test_how_many_defaults_to_three(monkeypatch):
    monkeypatch.delenv("RUFUS_VOICE_TAKES", raising=False)
    assert vt.how_many() == 3


def test_a_uniform_tone_reaches_the_kokoro_speed():
    """The bug this stage shipped with. A take is one chunk, the tone only
    added silence BETWEEN chunks, and three reads came back identical."""
    import emotional_map
    import tts_engine
    assert tts_engine._kokoro_speed_for(["weight"], 1.0) == \
        emotional_map.kokoro_speed("weight", 1.0)


def test_beats_that_disagree_keep_the_base_speed():
    """One call carries one speed. A script whose beats differ would otherwise
    have whichever tone won the tie decide the pace of all of them — that case
    keeps the inter-beat pauses it always had."""
    import tts_engine
    assert tts_engine._kokoro_speed_for(["weight", "curiosity"], 1.0) == 1.0


def test_no_tones_at_all_is_the_voice_that_always_shipped():
    import tts_engine
    assert tts_engine._kokoro_speed_for(None, 1.1) == 1.1
    assert tts_engine._kokoro_speed_for([], 1.1) == 1.1
