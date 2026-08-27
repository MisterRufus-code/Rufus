"""Two complete galleries for one script, and the swap that makes two enough.

A gallery of sixteen pictures is sixteen independent draws, not one artefact.
Variant A comes back best on shot 3 and worst on shot 9; variant B the other
way round. Choosing a whole bundle throws away the good half of the other one,
so the unit of choice is the SHOT — a base in one click, then corrections.

Two rather than three because of what a third buys: at a defect rate around one
in five the chance both variants fail the same shot is about four per cent, so
a sixteen-shot set expects well under one unfixable shot. A third takes that
under a sixth of a shot for another thirteen minutes of the 3090.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import gallery_variants as gv  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


def _set(db, beats=4, variants=2):
    sid = db.save_gallery_set(candidate_id=1, channel="c", niche="n",
                              topic="T", script_file="s.txt",
                              n_variants=variants)
    for v in range(variants):
        for b in range(beats):
            db.save_gallery_image(set_id=sid, variant=v, beat_index=b,
                                  path=f"/x/v{v}_{b}.png",
                                  prompt=f"prompt {b}", seed=b)
    return sid


def test_a_base_takes_every_shot_from_one_draw(db):
    sid = _set(db)
    assert db.choose_gallery_base(sid, 1) == 4
    assert {r["variant"] for r in db.gallery_images(sid, status="chosen")} == {1}


def test_a_swap_changes_one_shot_and_no_others(db):
    sid = _set(db)
    db.choose_gallery_base(sid, 0)
    assert db.swap_gallery_beat(sid, 2, 1)
    chosen = {r["beat_index"]: r["variant"]
              for r in db.gallery_images(sid, status="chosen")}
    assert chosen == {0: 0, 1: 0, 2: 1, 3: 0}


def test_a_shot_never_has_two_pictures_chosen(db):
    """The renderer takes one picture per beat. Two chosen rows for one shot is
    a coin toss decided by row order."""
    sid = _set(db)
    db.choose_gallery_base(sid, 0)
    db.swap_gallery_beat(sid, 1, 1)
    db.swap_gallery_beat(sid, 1, 0)
    per_beat = {}
    for r in db.gallery_images(sid, status="chosen"):
        per_beat[r["beat_index"]] = per_beat.get(r["beat_index"], 0) + 1
    assert set(per_beat.values()) == {1}


def test_swapping_to_a_draw_that_failed_this_shot_is_refused(db):
    """Fail-open per image means a variant can be short. Swapping to a picture
    that was never drawn would blank the shot instead of changing it."""
    sid = db.save_gallery_set(candidate_id=None, channel="c", niche="n",
                              topic="T", script_file="s", n_variants=2)
    db.save_gallery_image(set_id=sid, variant=0, beat_index=0, path="/a.png",
                          prompt="p", seed=1)
    assert db.swap_gallery_beat(sid, 0, 1) is False


def test_an_incomplete_set_hands_back_nothing(db):
    """clip[i] belongs to beat[i] everywhere downstream. A short list does not
    lose one picture — it slides every later one onto the wrong sentence, which
    is the exact failure this stage exists to prevent."""
    sid = _set(db, beats=3)
    db.choose_gallery_base(sid, 0)
    # a hole in the middle
    with db._conn() as c:
        c.execute("UPDATE gallery_images SET status='rejected' "
                  "WHERE set_id=? AND beat_index=1", (sid,))
    assert db.chosen_gallery(sid) == []


def test_a_complete_set_comes_back_in_beat_order(db):
    sid = _set(db, beats=4)
    db.choose_gallery_base(sid, 0)
    rows = db.chosen_gallery(sid)
    assert [r["beat_index"] for r in rows] == [0, 1, 2, 3]


def test_the_prompts_come_from_the_set_and_not_a_fresh_plan(db):
    """The storyboard is a model call and does not repeat itself. A run that
    re-planned would caption the chosen pictures with different beats than the
    ones they were drawn for."""
    sid = _set(db, beats=3)
    db.choose_gallery_base(sid, 1)
    assert gv.prompts_of(sid) == ["prompt 0", "prompt 1", "prompt 2"]


def test_clips_refuse_an_incomplete_set(db, capsys):
    sid = _set(db, beats=2)
    assert gv.clips_from(sid) == []
    assert "needs a picked picture" in capsys.readouterr().out


def test_clips_refuse_a_set_whose_files_are_gone(db, capsys):
    """A chosen row pointing at a deleted png would animate into nothing and
    take the beat alignment with it."""
    sid = _set(db, beats=2)
    db.choose_gallery_base(sid, 0)
    assert gv.clips_from(sid) == []
    assert "is gone" in capsys.readouterr().out


def test_how_many_defaults_to_two(monkeypatch):
    monkeypatch.delenv("RUFUS_GALLERY_VARIANTS", raising=False)
    assert gv.how_many() == 2


def test_a_nonsense_variant_count_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("RUFUS_GALLERY_VARIANTS", "lots")
    assert gv.how_many() == 2


# ── the picture matches the words playing over it ───────────────────────────

def test_the_prompts_are_planned_from_the_spoken_words(tmp_path, monkeypatch):
    """The whole reason the voice moved first. Prompts planned from a split of
    the script describe beat i of the TEXT; the renderer cuts beat i of the
    AUDIO, and nothing made those agree."""
    import main as rufus_main
    import voice_takes
    import beat_timing
    import comfy_client
    import db_manager as dbm

    script = tmp_path / "s.txt"
    script.write_text("One. Two. Three.", encoding="utf-8")
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "t.db")
    dbm.init_db()
    monkeypatch.setattr(gv, "gallery_dir", lambda sid: tmp_path / str(sid))
    monkeypatch.setattr(rufus_main, "_target_beats", lambda s: 3)
    monkeypatch.setattr(voice_takes, "build",
                        lambda *a, **k: [{"path": "/take.mp3", "tone": "n"}])
    monkeypatch.setattr(beat_timing, "spoken_shots", lambda mp3, n, *a: [
        {"index": 0, "start": 0, "end": 5, "seconds": 5, "text": "a computer"},
        {"index": 1, "start": 5, "end": 10, "seconds": 5, "text": "a cucumber"},
        {"index": 2, "start": 10, "end": 15, "seconds": 5, "text": "a coin"}])
    seen = {}
    monkeypatch.setattr(rufus_main, "_build_sd_prompts",
                        lambda s, n, max_scenes=10, grow=False, beats=None:
                        seen.update(beats=beats) or ["p1", "p2", "p3"])
    monkeypatch.setattr(comfy_client, "render_one_beat",
                        lambda *a, **k: False)

    gv.build(str(script), niche="money_history", channel="c", n_variants=1)
    assert seen["beats"] == ["a computer", "a cucumber", "a coin"]


def test_a_shot_with_no_words_falls_back_rather_than_sliding_everything(
        tmp_path, monkeypatch):
    """A partial list would silently slide every later picture onto the wrong
    window — worse than the text split it falls back to."""
    import main as rufus_main
    import voice_takes
    import beat_timing
    import comfy_client
    import db_manager as dbm

    script = tmp_path / "s.txt"
    script.write_text("One. Two. Three.", encoding="utf-8")
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "t2.db")
    dbm.init_db()
    monkeypatch.setattr(gv, "gallery_dir", lambda sid: tmp_path / f"g{sid}")
    monkeypatch.setattr(rufus_main, "_target_beats", lambda s: 3)
    monkeypatch.setattr(voice_takes, "build",
                        lambda *a, **k: [{"path": "/take.mp3", "tone": "n"}])
    monkeypatch.setattr(beat_timing, "spoken_shots", lambda mp3, n, *a: [
        {"index": 0, "text": "words"}, {"index": 1, "text": ""},
        {"index": 2, "text": "more"}])
    seen = {}
    monkeypatch.setattr(rufus_main, "_build_sd_prompts",
                        lambda s, n, max_scenes=10, grow=False, beats=None:
                        seen.update(beats=beats) or ["p1", "p2", "p3"])
    monkeypatch.setattr(comfy_client, "render_one_beat", lambda *a, **k: False)

    gv.build(str(script), niche="money_history", channel="c", n_variants=1)
    assert seen["beats"] is None, "a hole means use the text split, not a short list"


def test_the_prompt_count_follows_the_merge(tmp_path, monkeypatch):
    """max_scenes has to follow the MERGED count. Left at the original beat
    count it would let the prompt builder pad back to a number the audio no
    longer has cuts for — the duplicate problem reappearing one layer down."""
    import main as rufus_main
    import voice_takes
    import beat_timing
    import comfy_client
    import db_manager as dbm

    script = tmp_path / "s.txt"
    script.write_text("One. Two. Three.", encoding="utf-8")
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "m.db")
    dbm.init_db()
    monkeypatch.setattr(gv, "gallery_dir", lambda sid: tmp_path / f"m{sid}")
    monkeypatch.setattr(rufus_main, "_target_beats", lambda s: 3)
    monkeypatch.setattr(voice_takes, "build",
                        lambda *a, **k: [{"path": "/take.mp3", "tone": "n"}])
    monkeypatch.setattr(beat_timing, "spoken_shots", lambda mp3, n, *a: [
        {"index": 0, "start": 0, "end": 2, "seconds": 2,
         "text": "the Medici bank"},
        {"index": 1, "start": 2, "end": 4, "seconds": 2,
         "text": "the Medici bank again"},
        {"index": 2, "start": 4, "end": 7, "seconds": 3,
         "text": "a cucumber farmer"}])
    seen = {}
    monkeypatch.setattr(rufus_main, "_build_sd_prompts",
                        lambda s, n, max_scenes=10, grow=False, beats=None:
                        seen.update(max_scenes=max_scenes, beats=beats)
                        or ["p"] * len(beats or []))
    monkeypatch.setattr(comfy_client, "render_one_beat", lambda *a, **k: False)

    gv.build(str(script), niche="money_history", channel="c", n_variants=1)
    assert len(seen["beats"]) == 2, "the two bank shots became one"
    assert seen["max_scenes"] == 2


def test_the_stored_shot_lengths_describe_the_merged_shots(tmp_path,
                                                           monkeypatch):
    """The take's spans were measured against the UNMERGED count. Left alone,
    the page would show a shot list one length and a picture list another."""
    import json
    import main as rufus_main
    import voice_takes
    import beat_timing
    import comfy_client
    import db_manager as dbm

    script = tmp_path / "s.txt"
    script.write_text("One. Two.", encoding="utf-8")
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "sp.db")
    dbm.init_db()
    monkeypatch.setattr(gv, "gallery_dir", lambda sid: tmp_path / f"s{sid}")
    monkeypatch.setattr(rufus_main, "_target_beats", lambda s: 2)

    def _takes(script_file, *, set_id, **kw):
        dbm.save_voice_take(set_id=set_id, channel="c", topic="T", tone="n",
                            text="x", path="/t.mp3", seconds=4.0,
                            spans=json.dumps([{"index": 0}, {"index": 1}]))
        return [{"path": "/t.mp3", "tone": "n"}]
    monkeypatch.setattr(voice_takes, "build", _takes)
    monkeypatch.setattr(beat_timing, "spoken_shots", lambda mp3, n, *a: [
        {"index": 0, "start": 0, "end": 2, "seconds": 2, "text": "the bank"},
        {"index": 1, "start": 2, "end": 4, "seconds": 2,
         "text": "the bank again"}])
    monkeypatch.setattr(rufus_main, "_build_sd_prompts",
                        lambda s, n, max_scenes=10, grow=False, beats=None:
                        ["p"] * len(beats or []))
    monkeypatch.setattr(comfy_client, "render_one_beat", lambda *a, **k: False)

    sid = gv.build(str(script), niche="money_history", channel="c",
                   n_variants=1)
    stored = json.loads(dbm.voice_takes(set_id=sid)[0]["spans"])
    assert len(stored) == 1, "one held shot, one length"
    assert stored[0]["held"] == 2
