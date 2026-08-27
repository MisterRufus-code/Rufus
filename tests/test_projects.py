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


# ── the wizard ──────────────────────────────────────────────────────────────
#
# "You made a mess": four pages, none of which knew what the others decided.
# One page now, the current stage on it, every earlier stage a link back.

import pytest as _pytest  # noqa: E402


@_pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard
    import auth
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setenv("RUFUS_AUTH_DISABLED", "1")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setenv("RUFUS_LOG_DIR", str(tmp_path / "logs"))
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def _start(client):
    client.post("/create/new")
    return db_manager.projects(status="open")[0]["id"]


def test_a_fresh_dashboard_offers_to_start_one(client):
    page = client.get("/create").get_data(as_text=True)
    assert "Start a video" in page


def test_a_new_project_opens_at_the_topic_stage(client):
    pid = _start(client)
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert "What is it about?" in page
    assert db_manager.project(pid)["stage"] == "topic"


def test_suggesting_topics_fills_the_stage(client):
    pid = _start(client)
    client.post(f"/create/{pid}/regen/topic")
    assert len(db_manager.project_topics(pid)) == 3


def test_regen_replaces_the_options_rather_than_adding_to_them(client):
    """The options are samples, not answers. A REGEN that appends gives you six
    to read instead of three to choose between."""
    pid = _start(client)
    client.post(f"/create/{pid}/regen/topic")
    client.post(f"/create/{pid}/regen/topic")
    assert len(db_manager.project_topics(pid)) == 3


def test_a_typed_topic_goes_straight_to_writing(client, monkeypatch):
    """You said what you want. Nothing else is a question worth asking."""
    import dashboard
    started = {}
    monkeypatch.setattr(dashboard, "_launch_candidates",
                        lambda **kw: started.update(kw) or Path("c.log"))
    pid = _start(client)
    client.post(f"/create/{pid}/topic/custom", data={"topic": "Tulip mania"},
                follow_redirects=True)
    p = db_manager.project(pid)
    assert p["title"] and p["topic_source"] == "typed"
    assert p["stage"] == "script"
    assert started.get("project_id") == pid


def test_an_empty_typed_topic_is_refused(client):
    pid = _start(client)
    client.post(f"/create/{pid}/topic/custom", data={"topic": "  "})
    assert db_manager.project(pid)["title"] is None


# ── going back, which is the part that has to be right ──────────────────────

def test_going_back_from_the_bar_forgets_what_came_after(client):
    pid = _start(client)
    db_manager.update_project(pid, title="T", script_id=1, script_file="s.txt",
                              gallery_id=2, voice_id=3, stage="render")
    client.post(f"/create/{pid}/back/script")
    p = db_manager.project(pid)
    assert p["stage"] == "script"
    assert (p["script_id"], p["gallery_id"], p["voice_id"]) == (None, None, None)
    assert p["title"] == "T"


def test_the_bar_offers_a_way_back_to_earlier_stages_only(client):
    """A link to a stage you have not reached is a link to an empty page."""
    pid = _start(client)
    db_manager.update_project(pid, title="T", script_id=1, script_file="s.txt",
                              stage="gallery")
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert f"/create/{pid}/back/topic" in page
    assert f"/create/{pid}/back/script" in page
    assert f"/create/{pid}/back/voice" not in page


def test_an_unknown_stage_is_refused_rather_than_guessed(client):
    pid = _start(client)
    client.post(f"/create/{pid}/back/colour-grade")
    assert db_manager.project(pid)["stage"] == "topic"


# ── the render is last, behind every judgement ──────────────────────────────

def test_rendering_needs_every_decision_made(client, monkeypatch):
    import dashboard
    rendered = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: rendered.append(kw) or (None, Path("r.log")))
    pid = _start(client)
    db_manager.update_project(pid, title="T", stage="render")
    client.post(f"/create/{pid}/render")
    assert rendered == [], "a half-decided project must not render"


def test_rendering_carries_every_choice_and_regenerates_none(client,
                                                             monkeypatch):
    import dashboard
    launched = {}
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: launched.update(kw) or (None, Path("r.log")))
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=1, channel="main_en",
                                      niche="n", topic="T",
                                      script_file="s.txt", n_variants=2)
    tid = db_manager.save_voice_take(set_id=sid, channel="main_en", topic="T",
                                     tone="weight", text="t", path="/a.mp3")
    db_manager.update_project(pid, title="T", script_id=1, script_file="s.txt",
                              gallery_id=sid, voice_id=tid, stage="render")
    client.post(f"/create/{pid}/render")
    assert launched["script_file"] == "s.txt"
    assert launched["gallery_id"] == sid
    assert launched["hook_tone"] == "weight"


def test_abandoning_deletes_nothing(client):
    """It cost real money and real GPU, and a project you gave up on is still
    the record of three scripts somebody read and rejected."""
    pid = _start(client)
    client.post(f"/create/{pid}/regen/topic")
    client.post(f"/create/{pid}/abandon")
    assert db_manager.project(pid)["status"] == "abandoned"
    assert len(db_manager.project_topics(pid)) == 3


# ── going back has to give the options back ─────────────────────────────────
#
# "When you go back it doesn't save." Forgetting FORWARD was right — pick a
# different script and the gallery drawn for the old one is pictures of a
# different video. But the first version cleared the project's pointers and
# left the options themselves marked chosen and rejected, and every stage lists
# what is still pending. So going back landed on a stage that looked empty and
# offered to regenerate: three scripts already paid for, invisible, and the
# only button on screen spending money to write three more.

def test_going_back_to_the_script_stage_shows_the_scripts_again(db):
    pid = _project(db)
    ids = [db.save_candidate(proposal_id=None, project_id=pid, channel="c",
                             niche="n", topic="T", hook_style=s, hook="h",
                             script="s", score=8)
           for s in ("counterintuitive", "shocking_stat", "warning")]
    db.choose_candidate(ids[0])
    db.update_project(pid, script_id=ids[0], script_file="s.txt",
                      stage="gallery")
    assert db.candidates(project_id=pid, status="pending") == []

    db.clear_project_from(pid, "script")
    back = db.candidates(project_id=pid, status="pending")
    assert len(back) == 3, "the three you already paid for must come back"
    assert db.project(pid)["script_id"] is None


def test_going_back_to_the_voice_stage_shows_the_takes_again(db):
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=1, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    tids = [db.save_voice_take(set_id=sid, channel="c", topic="T", tone=t,
                               text="x", path=f"/{t}.mp3")
            for t in ("curiosity", "tension", "weight")]
    db.choose_voice_take(tids[1])
    db.update_project(pid, gallery_id=sid, voice_id=tids[1], stage="render")

    db.clear_project_from(pid, "voice")
    assert len(db.voice_takes(set_id=sid, status="pending")) == 3
    assert db.project(pid)["voice_id"] is None
    assert db.project(pid)["gallery_id"] == sid, "the pictures were decided before"


def test_going_back_to_the_gallery_keeps_the_per_shot_picks(db):
    """Re-opening the SET is what makes it choosable again. Wiping which
    picture won each shot would throw away the one part of that stage that
    took real attention."""
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=1, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    for v in range(2):
        for b in range(3):
            db.save_gallery_image(set_id=sid, variant=v, beat_index=b,
                                  path=f"/v{v}_{b}.png", prompt="p", seed=1)
    db.choose_gallery_base(sid, 1)
    db.swap_gallery_beat(sid, 0, 0)
    db.decide_gallery_set(sid, "chosen")
    db.update_project(pid, gallery_id=sid, stage="voice")

    db.clear_project_from(pid, "gallery")
    assert db.gallery_sets(status="pending", limit=10)[0]["id"] == sid
    picks = {r["beat_index"]: r["variant"]
             for r in db.gallery_images(sid, status="chosen")}
    assert picks == {0: 0, 1: 1, 2: 1}, "the shots you chose are still chosen"


def test_going_back_to_the_topic_shows_the_topics_again(db):
    pid = _project(db)
    db.save_project_topics(pid, [{"title": "A"}, {"title": "B"},
                                 {"title": "C"}])
    opts = db.project_topics(pid)
    db.choose_project_topic(opts[0]["id"])
    db.update_project(pid, title="A", stage="script")

    db.clear_project_from(pid, "topic")
    assert len(db.project_topics(pid)) == 3
    assert db.project(pid)["title"] is None


def test_going_back_never_costs_money(db, monkeypatch):
    """The whole complaint: the only button on an emptied stage was one that
    spends. Nothing about going back may need a model call."""
    import script_candidates
    import topic_options
    for mod, fn in ((script_candidates, "write_for"),
                    (topic_options, "suggest")):
        monkeypatch.setattr(mod, fn, lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(f"{fn} must not run when going back")))
    pid = _project(db)
    db.save_project_topics(pid, [{"title": "A"}, {"title": "B"}])
    db.update_project(pid, title="A", script_id=1, script_file="s.txt",
                      stage="gallery")
    db.clear_project_from(pid, "script")
    db.clear_project_from(pid, "topic")


def test_choosing_a_wizard_script_rejects_its_siblings(db):
    """SIBLINGS SHARE A PROPOSAL *OR* A PROJECT, and only the first was
    checked. The wizard writes candidates with a project and no proposal, so
    choosing one rejected nothing: all three stayed pending, the stage still
    offered them, and the preference pair — the entire reason the losers are
    kept — was never recorded."""
    pid = _project(db)
    ids = [db.save_candidate(proposal_id=None, project_id=pid, channel="c",
                             niche="n", topic="T", hook_style=s, hook="h",
                             script="s", score=8)
           for s in ("a", "b", "c")]
    db.choose_candidate(ids[1])
    rows = {r["id"]: r["status"] for r in db.candidates(project_id=pid)}
    assert rows[ids[1]] == "chosen"
    assert rows[ids[0]] == rows[ids[2]] == "rejected"


def test_another_projects_scripts_are_not_siblings(db):
    a, b = _project(db), _project(db)
    ia = db.save_candidate(proposal_id=None, project_id=a, channel="c",
                           niche="n", topic="A", hook_style="x", hook="h",
                           script="s", score=8)
    ib = db.save_candidate(proposal_id=None, project_id=b, channel="c",
                           niche="n", topic="B", hook_style="x", hook="h",
                           script="s", score=8)
    db.choose_candidate(ia)
    assert db.candidates(project_id=b)[0]["status"] == "pending"
    assert ib
