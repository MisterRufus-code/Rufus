"""One video in progress, across all of its stages.

The four choosing stages each grew their own table and their own page, and
separately they work — but a person making a video does not have four tasks,
they have one, and nothing tied a chosen script back to the topic it came from
or forward to the pictures drawn for it. Four tabs is what that looks like from
the outside, and the owner said so: "you made a mess".

A project is the thread, and the part that has to be right is going BACKWARDS.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import topic_options  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


def _project(db):
    return db.new_project(channel="main_en", niche="money_history")


def test_a_new_project_starts_at_the_topic_stage(db):
    p = db.project(_project(db))
    assert p["stage"] == "topic" and p["status"] == "open"
    assert p["title"] is None


def test_the_stages_are_in_the_order_they_happen():
    assert db_manager.STAGES == ("topic", "script", "gallery", "voice",
                                 "render")


def test_choosing_a_topic_records_the_ones_passed_over(db):
    pid = _project(db)
    db.save_project_topics(pid, [{"title": "A", "why": "x"},
                                 {"title": "B", "why": "y"},
                                 {"title": "C", "why": "z"}])
    opts = db.project_topics(pid)
    db.choose_project_topic(opts[1]["id"])
    rows = {t["title"]: t["status"] for t in db.project_topics(pid, status=None)}
    assert rows == {"A": "rejected", "B": "chosen", "C": "rejected"}


def test_a_topic_cannot_be_chosen_twice(db):
    pid = _project(db)
    db.save_project_topics(pid, [{"title": "A"}, {"title": "B"}])
    opts = db.project_topics(pid)
    assert db.choose_project_topic(opts[0]["id"])
    assert db.choose_project_topic(opts[1]["id"]) is None


# ── going back has to be honest ─────────────────────────────────────────────

def test_going_back_to_the_script_stage_drops_the_gallery(db):
    """Re-open the script stage, pick differently, and the gallery drawn for
    the old one is not merely stale — it is pictures of a script that is no
    longer being made. Leaving it attached would render the video the owner
    just decided against."""
    pid = _project(db)
    db.update_project(pid, title="Bretton Woods", script_id=7,
                      script_file="s.txt", gallery_id=3, voice_id=9,
                      stage="render")
    db.clear_project_from(pid, "script")
    p = db.project(pid)
    assert p["stage"] == "script"
    assert p["script_id"] is None and p["gallery_id"] is None
    assert p["voice_id"] is None
    assert p["title"] == "Bretton Woods", "the topic was decided before this"


def test_going_back_to_the_topic_stage_drops_everything(db):
    pid = _project(db)
    db.update_project(pid, title="T", script_id=1, gallery_id=2, voice_id=3,
                      stage="voice")
    db.clear_project_from(pid, "topic")
    p = db.project(pid)
    assert p["title"] is None and p["script_id"] is None
    assert p["gallery_id"] is None and p["voice_id"] is None


def test_going_back_to_the_last_stage_drops_only_it(db):
    pid = _project(db)
    db.update_project(pid, title="T", script_id=1, gallery_id=2, voice_id=3,
                      stage="render")
    db.clear_project_from(pid, "voice")
    p = db.project(pid)
    assert p["voice_id"] is None
    assert p["gallery_id"] == 2 and p["script_id"] == 1


def test_an_unknown_stage_is_refused(db):
    with pytest.raises(ValueError):
        db.clear_project_from(_project(db), "colour-grade")


def test_a_typo_in_a_column_name_is_refused_rather_than_ignored(db):
    """A field that silently does nothing is a stage that looks saved and is
    not, and this is the one table the wizard has to be able to trust."""
    with pytest.raises(ValueError):
        db.update_project(_project(db), titel="oops")


# ── topics, with nothing configured ─────────────────────────────────────────

def test_suggestions_work_on_a_fresh_install(db, monkeypatch):
    """scout.py needs competitors.json to say anything at all. This is the
    first thing a person touches, so it has to answer with nothing set up."""
    monkeypatch.setattr(db_manager, "DB_FILE", db.DB_FILE)
    got = topic_options.suggest("money_history", 3)
    assert len(got) == 3
    assert all(o["title"] and o["why"] for o in got)


def test_every_suggestion_says_why_it_is_there(db):
    """"Make a video about the Panic of 1893" is an instruction, and an
    instruction from an agent is the thing a person cannot audit."""
    for o in topic_options.suggest("money_history", 3):
        assert len(o["why"]) > 10, o


def test_a_typed_topic_skips_the_suggesting_entirely(monkeypatch):
    """You said what you want. Second-guessing it is the pipeline arguing with
    the person operating it."""
    import research
    monkeypatch.setattr(research, "get_seed",
                        lambda niche, topic=None: {"title": "Bretton Woods system",
                                                   "source": "Wikipedia"})
    got = topic_options.take_topic("Bretton Woods")
    assert got["title"] == "Bretton Woods system"
    assert got["source"] == "Wikipedia"


def test_a_typed_topic_survives_a_research_outage(monkeypatch, capsys):
    import research
    monkeypatch.setattr(research, "get_seed",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert topic_options.take_topic("Tulip mania")["title"] == "Tulip mania"
    assert "using it as typed" in capsys.readouterr().out
