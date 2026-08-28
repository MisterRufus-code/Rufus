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
    # status=None, because abandoning now RETIRES what was still pending — the
    # rows are all still there, they have just stopped being offered.
    assert len(db_manager.project_topics(pid, status=None)) == 3


def test_abandoning_takes_its_leftovers_out_of_the_queues(client):
    """"I don't want all this mass with all the scripts mixed and mixed
    gallery — I want to make the generation live, not save it for after."

    The stage queues count every pending row in the database with no idea
    which project it belongs to, so the leftovers of a project abandoned last
    week sat in /scripts, /galleries and /voice forever, next to today's. The
    badges said seven reads were waiting when one was."""
    pid = _start(client)
    client.post(f"/create/{pid}/regen/topic")
    assert len(db_manager.project_topics(pid)) == 3

    client.post(f"/create/{pid}/abandon")

    assert db_manager.project_topics(pid) == [], "nothing of it is still offered"
    assert len(db_manager.project_topics(pid, status=None)) == 3, "and nothing is gone"


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


# ── how far in, and how much longer ─────────────────────────────────────────
#
# "Reload this page" is not a design, it is an apology. Three of the five
# stages take real time and the page said nothing about any of it: you pressed
# a button, the screen went quiet, and the only way to learn anything was F5.
#
# Nothing new had to be instrumented — every slow stage fills a table as it
# goes, so progress is a row count against a target that is known before the
# work starts. Exact rather than estimated, one small query.

def test_a_stage_with_nothing_yet_is_working_not_finished(db):
    pid = _project(db)
    db.update_project(pid, title="T", stage="script")
    import dashboard
    prog = dashboard._project_progress(db.project(pid))
    assert prog["total"] == 3 and prog["done"] == 0


def test_progress_counts_rows_as_they_arrive(db):
    import dashboard
    pid = _project(db)
    db.update_project(pid, title="T", stage="script")
    for i in range(2):
        db.save_candidate(proposal_id=None, project_id=pid, channel="c",
                          niche="n", topic="T", hook_style=str(i), hook="h",
                          script="s", score=8)
    prog = dashboard._project_progress(db.project(pid))
    assert (prog["done"], prog["total"], prog["working"]) == (2, 3, True)


def test_a_finished_stage_stops_reporting_work(db):
    import dashboard
    pid = _project(db)
    db.update_project(pid, title="T", stage="script")
    for i in range(3):
        db.save_candidate(proposal_id=None, project_id=pid, channel="c",
                          niche="n", topic="T", hook_style=str(i), hook="h",
                          script="s", score=8)
    assert dashboard._project_progress(db.project(pid))["working"] is False


def test_a_half_drawn_gallery_does_not_claim_to_be_finished(db):
    """A set half-drawn does not yet know its own beat count. Guessing the
    target low would show 100% while pictures were still arriving."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=1, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    db.update_project(pid, script_id=1, gallery_id=sid, stage="gallery")
    for b in range(4):
        db.save_gallery_image(set_id=sid, variant=0, beat_index=b,
                              path=f"/a{b}.png", prompt="p", seed=1)
    prog = dashboard._project_progress(db.project(pid))
    assert prog["total"] >= 8, "four beats over two variants is eight pictures"
    assert prog["working"] is True


def test_the_eta_is_only_offered_once_there_is_something_to_divide(db):
    """elapsed ÷ done × remaining needs a done. Before the first row it would
    be a number made up to look reassuring."""
    import dashboard
    pid = _project(db)
    db.update_project(pid, title="T", stage="voice")
    assert dashboard._project_progress(db.project(pid))["eta_seconds"] is None


def test_the_eta_reads_as_time_rather_than_seconds():
    import dashboard
    assert dashboard._eta_words(None) == ""
    assert "under a minute" in dashboard._eta_words(30)
    assert "minute" in dashboard._eta_words(600)
    assert "hour" in dashboard._eta_words(7200)


def test_the_page_polls_instead_of_asking_to_be_reloaded(client):
    import dashboard
    pid = _start(client)
    db_manager.update_project(pid, title="T", stage="script")
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert "Reload this page" not in page
    assert 'id="wizard-progress"' in page
    assert "/api/create/" in page


def test_the_progress_endpoint_is_cheap_and_answers_json(client):
    pid = _start(client)
    db_manager.update_project(pid, title="T", stage="script")
    r = client.get(f"/api/create/{pid}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["total"] == 3


def test_the_progress_endpoint_says_so_for_a_project_that_is_gone(client):
    assert client.get("/api/create/99999").status_code == 404


# ── watching the pictures arrive ────────────────────────────────────────────

def test_a_gallery_still_drawing_shows_progress_not_a_half_empty_table(client):
    """The set row is written before the first picture, so as soon as drawing
    starts that branch renders — and it used to render a half-empty table with
    nothing to say more was coming. Forty minutes of that reads as broken."""
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=5, channel="c", niche="n",
                                      topic="T", script_file="s.txt",
                                      n_variants=2)
    db_manager.update_project(pid, title="T", script_id=5,
                              script_file="s.txt", stage="gallery")
    for b in range(3):
        db_manager.save_gallery_image(set_id=sid, variant=0, beat_index=b,
                                      path=f"/a{b}.png", prompt="p", seed=1)
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert 'id="wizard-progress"' in page
    assert "Drawing the pictures" in page
    assert "These pictures" not in page, "nothing to settle while it is drawing"


def test_the_pictures_drawn_so_far_are_shown_while_it_works(client):
    """They are shown in the drawing room now, not squeezed into the wizard
    step. Forty minutes of GPU behind a bar and a sentence is why the owner
    watched ComfyUI's console in another window instead."""
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=6, channel="c", niche="n",
                                      topic="T", script_file="s.txt",
                                      n_variants=2)
    db_manager.update_project(pid, title="T", script_id=6,
                              script_file="s.txt", stage="gallery")
    db_manager.set_gallery_beats(sid, 2)
    img = db_manager.save_gallery_image(set_id=sid, variant=0, beat_index=0,
                                        path="/a.png", prompt="p", seed=1)

    # the wizard sends you there
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert f'href="/drawing/{sid}"' in page

    # and there they are
    d = client.get(f"/api/drawing/{sid}").get_json()
    assert d["done"] == 1 and d["total"] == 4
    assert any(sl["image"] == img for sl in d["slots"])
    assert len(d["slots"]) == 4, "the ones still to come are slots too"


def test_a_finished_gallery_offers_the_choice(client):
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=7, channel="c", niche="n",
                                      topic="T", script_file="s.txt",
                                      n_variants=2)
    db_manager.update_project(pid, title="T", script_id=7,
                              script_file="s.txt", stage="gallery")
    for v in range(2):
        for b in range(2):
            db_manager.save_gallery_image(set_id=sid, variant=v, beat_index=b,
                                          path=f"/v{v}{b}.png", prompt="p",
                                          seed=1)
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert "Which pictures?" in page and "These pictures" in page


def test_an_in_progress_set_gets_an_eta(db):
    """The estimate comes from the spacing between the pictures that actually
    landed. Three written in the same instant have no spacing to measure — and
    there is no fallback any more, because the fallback was the sum that
    reported twelve hours for a draw doing one every nineteen seconds."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=8, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    db.update_project(pid, script_id=8, gallery_id=sid, stage="gallery")
    db.set_gallery_beats(sid, 8)
    with db._conn() as c:
        for b in range(3):
            c.execute("INSERT INTO gallery_images (set_id,variant,beat_index,"
                      "path,prompt,seed,created_at) VALUES (?,?,?,?,?,?,"
                      "datetime('now',?))",
                      (sid, 0, b, f"/a{b}.png", "p", 1, f"-{(3-b)*19} seconds"))

    prog = dashboard._project_progress(db.project(pid))

    assert prog["working"] is True
    assert prog["stalled"] is False
    assert prog["eta_seconds"] is not None
    assert 18 <= prog["rate_seconds"] <= 20


def test_a_draw_that_stopped_says_so_instead_of_estimating(db):
    """THE SCREEN THAT CAUSED REAL DAMAGE. The bar promised "~12h 10m left"
    for a set that had not gained a picture in hours, so the owner turned
    ComfyUI OFF trying to unstick it — which is what a person does when the
    screen insists something is still happening."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=8, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    db.update_project(pid, script_id=8, gallery_id=sid, stage="gallery")
    db.set_gallery_beats(sid, 19)
    with db._conn() as c:
        for b in range(9):
            c.execute("INSERT INTO gallery_images (set_id,variant,beat_index,"
                      "path,prompt,seed,created_at) VALUES (?,?,?,?,?,?,"
                      "datetime('now','-3 hours'))",
                      (sid, 0, b, f"/a{b}.png", "p", 1))

    prog = dashboard._project_progress(db.project(pid))

    assert prog["stalled"] is True
    assert prog["eta_seconds"] is None, "no estimate for work that stopped"
    assert prog["quiet_seconds"] > 3000


def test_a_set_still_moving_is_not_called_stalled(db):
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=8, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    db.update_project(pid, script_id=8, gallery_id=sid, stage="gallery")
    db.set_gallery_beats(sid, 19)
    with db._conn() as c:
        for b in range(9):
            c.execute("INSERT INTO gallery_images (set_id,variant,beat_index,"
                      "path,prompt,seed,created_at) VALUES (?,?,?,?,?,?,"
                      "datetime('now',?))",
                      (sid, 0, b, f"/a{b}.png", "p", 1, f"-{(9-b)*19} seconds"))
    prog = dashboard._project_progress(db.project(pid))
    assert prog["stalled"] is False
    assert prog["eta_seconds"] is not None


def test_the_gallery_builder_records_the_voice_before_it_draws():
    """The page promised "the seconds beside each shot are measured from the
    recorded voice" while nothing had been recorded to measure — the takes were
    only made at the voice stage, one step later."""
    src = (Path(__file__).parent.parent / "scripts" / "gallery_variants.py"
           ).read_text(encoding="utf-8")
    assert "voice_takes.build" in src
    assert src.index("voice_takes.build") < src.index("render_one_beat"), (
        "the takes have to exist before the pictures, or there is nothing to "
        "measure the shot lengths against")


def test_a_gallery_with_nothing_drawn_yet_is_working_not_finished(db):
    """THE ONE THE OWNER CAUGHT. ComfyUI was rendering, and the wizard showed
    the completed gallery view over an empty table — because the target was
    inferred from the pictures already drawn, so a set with none reported a
    target of zero, and zero reads as finished."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=9, channel="c", niche="n",
                              topic="T", script_file="s", n_variants=2)
    db.set_gallery_beats(sid, 16)
    db.update_project(pid, title="T", script_id=9, script_file="s",
                      stage="gallery")
    prog = dashboard._project_progress(db.project(pid))
    assert prog["working"] is True
    assert (prog["done"], prog["total"]) == (0, 32)


def test_the_target_comes_from_the_plan_not_the_output(db):
    """Counting distinct variants in a half-drawn set answers 1, because the
    first variant finishes before the second starts."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=9, channel="c", niche="n",
                              topic="T", script_file="s", n_variants=2)
    db.set_gallery_beats(sid, 5)
    db.update_project(pid, title="T", script_id=9, script_file="s",
                      stage="gallery", gallery_id=sid)
    for b in range(5):
        db.save_gallery_image(set_id=sid, variant=0, beat_index=b,
                              path=f"/a{b}.png", prompt="p", seed=1)
    prog = dashboard._project_progress(db.project(pid))
    assert prog["total"] == 10 and prog["done"] == 5
    assert prog["working"] is True, "half of it has not been drawn"


def test_a_set_from_before_the_target_existed_still_reports_something(db):
    """An old row has n_beats 0. Falling back to the widest row seen so far is
    worse than the plan and much better than zero."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=9, channel="c", niche="n",
                              topic="T", script_file="s", n_variants=2)
    db.update_project(pid, title="T", script_id=9, script_file="s",
                      stage="gallery", gallery_id=sid)
    for b in range(3):
        db.save_gallery_image(set_id=sid, variant=0, beat_index=b,
                              path=f"/a{b}.png", prompt="p", seed=1)
    assert dashboard._project_progress(db.project(pid))["total"] == 6


def test_the_page_shows_the_bar_before_the_first_picture(client):
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=4, channel="c", niche="n",
                                      topic="T", script_file="s.txt",
                                      n_variants=2)
    db_manager.set_gallery_beats(sid, 16)
    db_manager.update_project(pid, title="T", script_id=4,
                              script_file="s.txt", stage="gallery")
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert "Drawing the pictures" in page
    assert "0 of 32" in page
    assert "Which pictures?" not in page


def test_the_builder_records_its_target_before_drawing():
    src = (Path(__file__).parent.parent / "scripts" / "gallery_variants.py"
           ).read_text(encoding="utf-8")
    assert "set_gallery_beats" in src
    assert src.index("set_gallery_beats") < src.index("render_one_beat"), (
        "the target has to exist before the first picture, or the panel "
        "reports finished while nothing has been drawn")


def test_regen_retires_the_scripts_it_is_replacing(tmp_path, monkeypatch):
    """THE DEFECT THIS PINS. save_project_topics has always deleted the pending
    options before writing the replacements. The other three stages never did,
    so pressing "again" wrote three MORE scripts alongside the three already
    there — the stage built to narrow a choice widening it every time."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "regen.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    for style in ("counterintuitive", "shocking_stat", "warning"):
        dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                           topic="T", hook_style=style, hook="h",
                           script="s", score=8, project_id=pid)
    assert len(dbm.candidates(project_id=pid, status="pending")) == 3

    gone = dbm.superseded_by_regen(pid, "script")

    assert gone == 3
    assert dbm.candidates(project_id=pid, status="pending") == []


def test_a_retired_option_is_kept_not_deleted(tmp_path, monkeypatch):
    """A passed-over option is half of a labelled pair — what won is only
    meaningful beside what it beat. Deleting the rows would trade a tidy table
    for the measurements."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "keep.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    dbm.save_candidate(proposal_id=None, channel="c", niche="n", topic="T",
                       hook_style="warning", hook="h", script="s", score=8,
                       project_id=pid)
    dbm.superseded_by_regen(pid, "script")
    assert dbm.candidates(project_id=pid, status="superseded"), (
        "the row must survive with a new status, not be deleted")


def test_regen_of_the_pictures_retires_only_the_pictures(tmp_path, monkeypatch):
    """Regenerating the pictures says nothing about the script that was
    already settled. Clearing forward from here is a different operation with
    a different name."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "scope.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    cid = dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                             topic="T", hook_style="warning", hook="h",
                             script="s", score=8, project_id=pid)
    dbm.update_project(pid, script_id=cid)
    sid = dbm.save_gallery_set(candidate_id=cid, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)

    gone = dbm.superseded_by_regen(pid, "gallery")

    assert gone == 1
    assert [g["id"] for g in dbm.gallery_sets(status="pending")] == []
    assert dbm.gallery_sets(status="superseded")[0]["id"] == sid
    # the script is untouched
    assert dbm.candidates(project_id=pid)[0]["id"] == cid


def test_regen_with_nothing_pending_retires_nothing(tmp_path, monkeypatch):
    """The first press of a stage has no options to replace, and the message
    must not claim otherwise."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "empty.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    assert dbm.superseded_by_regen(pid, "script") == 0
    assert dbm.superseded_by_regen(pid, "gallery") == 0
    assert dbm.superseded_by_regen(pid, "voice") == 0


def test_regen_rejects_a_stage_that_does_not_exist(tmp_path, monkeypatch):
    import db_manager as dbm
    import pytest
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "bad.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    with pytest.raises(ValueError):
        dbm.superseded_by_regen(pid, "pictures")


def test_the_regen_route_retires_before_it_launches():
    """Order matters: launching first would leave a window where both sets are
    pending, and /galleries reads pending."""
    from pathlib import Path
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    body = src.split("def create_regen", 1)[1].split("\n@app.route", 1)[0]
    for stage, launch in (("script", "_launch_candidates"),
                          ("gallery", "_launch_galleries"),
                          ("voice", "_launch_voice_takes")):
        block = body.split(f'if stage == "{stage}":', 1)[1]
        assert f'superseded_by_regen(project_id, "{stage}")' in block
        assert block.index("superseded_by_regen") < block.index(launch), (
            f"{stage}: the old options must be retired before the new run starts")


# ── who decided ──────────────────────────────────────────────────────────────

def test_every_stage_records_who_decided(tmp_path, monkeypatch):
    """THE GAP THIS CLOSES. Two people work this channel, and until these
    columns the database recorded WHAT was chosen and never once WHOM by.
    "Who made this video" had no answer to give."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "who.db")
    dbm.init_db()

    pid = dbm.new_project(channel="c", niche="n", by="Daniel")
    cid = dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                             topic="T", hook_style="warning", hook="h",
                             script="s", score=8, project_id=pid)
    dbm.choose_candidate(cid, by="Daniel")
    sid = dbm.save_gallery_set(candidate_id=cid, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    for beat in range(2):
        for v in range(2):
            dbm.save_gallery_image(set_id=sid, variant=v, beat_index=beat,
                                   path=f"/{v}{beat}.png", prompt="p", seed=1)
    dbm.choose_gallery_base(sid, 0, by="Daniel")
    dbm.swap_gallery_beat(sid, 1, 1, by="Partner")
    dbm.decide_gallery_set(sid, "chosen", by="Daniel")
    tid = dbm.save_voice_take(set_id=sid, channel="c", topic="T", tone="calm",
                              text="t", path="/a.mp3")
    dbm.choose_voice_take(tid, by="Partner")

    assert dbm.project(pid)["created_by"] == "Daniel"
    assert dbm.candidates(project_id=pid)[0]["decided_by"] == "Daniel"
    assert dbm.gallery_sets(status=None)[0]["decided_by"] == "Daniel"
    assert dbm.voice_takes(set_id=sid)[0]["decided_by"] == "Partner"


def test_the_swaps_are_credited_to_whoever_made_them(tmp_path, monkeypatch):
    """Per shot, because the swaps are where the attention went: taking a base
    is one click and correcting eight shots is eight judgements. A set credited
    only to whoever pressed "take all from A" hides the person who did that
    work."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "swap.db")
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=None, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    for beat in range(2):
        for v in range(2):
            dbm.save_gallery_image(set_id=sid, variant=v, beat_index=beat,
                                   path=f"/{v}{beat}.png", prompt="p", seed=1)
    dbm.choose_gallery_base(sid, 0, by="Daniel")
    dbm.swap_gallery_beat(sid, 1, 1, by="Partner")

    picked = {im["beat_index"]: im["decided_by"]
              for im in dbm.gallery_images(sid, status="chosen")}
    assert picked == {0: "Daniel", 1: "Partner"}


def test_the_rejected_side_is_credited_too(tmp_path, monkeypatch):
    """"I rejected these two" is the same act as "I chose that one" — a
    preference pair labelled on only one side is half a record."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "pair.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n", by="Daniel")
    ids = [dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                              topic="T", hook_style=st, hook="h", script="s",
                              score=8, project_id=pid)
           for st in ("warning", "shocking_stat", "counterintuitive")]
    dbm.choose_candidate(ids[0], by="Daniel")
    rows = dbm.candidates(project_id=pid)
    assert {r["status"] for r in rows} == {"chosen", "rejected"}
    assert all(r["decided_by"] == "Daniel" for r in rows)


def test_an_unattributed_decision_stays_blank_rather_than_inventing_a_name(
        tmp_path, monkeypatch):
    """NULL honestly means "this predates attribution, or auth was off".
    Inventing "local" would make an anonymous decision indistinguishable from
    one somebody actually made."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "anon.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    cid = dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                             topic="T", hook_style="warning", hook="h",
                             script="s", score=8, project_id=pid)
    dbm.choose_candidate(cid)
    assert dbm.project(pid)["created_by"] is None
    assert dbm.candidates(project_id=pid)[0]["decided_by"] is None


def test_approving_a_video_records_who(tmp_path, monkeypatch):
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "vid.db")
    dbm.init_db()
    vid = dbm.save_video(niche="n", script_hook="h", scene_desc="d",
                         video_file="v.mp4", score=8)
    dbm.set_upload_status(vid, "approved", by="Daniel")
    with dbm._conn() as c:
        row = c.execute("SELECT upload_status, decided_by FROM videos "
                        "WHERE id=?", (vid,)).fetchone()
    assert row == ("approved", "Daniel")


def test_an_unnamed_status_change_does_not_blank_the_existing_name(
        tmp_path, monkeypatch):
    """The auto-approve sweep passes no name. It must not erase the person who
    actually made the earlier decision."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "keepname.db")
    dbm.init_db()
    vid = dbm.save_video(niche="n", script_hook="h", scene_desc="d",
                         video_file="v.mp4", score=8)
    dbm.set_upload_status(vid, "approved", by="Daniel")
    dbm.set_upload_status(vid, "pending")
    with dbm._conn() as c:
        row = c.execute("SELECT decided_by FROM videos WHERE id=?",
                        (vid,)).fetchone()
    assert row[0] == "Daniel"


def test_every_decision_route_passes_the_signed_in_name():
    """A column nothing writes to is the shape of bug this repo keeps having:
    built, wired, and never actually fed."""
    from pathlib import Path
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    for call in ("choose_candidate(", "choose_gallery_base(",
                 "swap_gallery_beat(", "choose_voice_take(",
                 "decide_gallery_set(", "new_project(",
                 "set_upload_status(video_id"):
        for line in src.splitlines():
            if call in line and "def " not in line:
                assert "_whoami()" in line or "by=" in line, (
                    f"{call} called without a name: {line.strip()}")


def test_a_name_gets_the_same_colour_everywhere():
    """A name printed in the same grey as everything else is a word you have to
    read; one that is always the same colour is one you recognise."""
    import dashboard
    assert dashboard._by_hue("Daniel") == dashboard._by_hue("Daniel")
    chip = dashboard._by_badge("Daniel")
    assert "Daniel" in chip and "--who:" in chip


def test_nobody_recorded_shows_a_dash_not_a_made_up_name():
    import dashboard
    assert "—" in dashboard._by_badge(None)
    assert "—" in dashboard._by_badge("")


# ── notes ────────────────────────────────────────────────────────────────────

def test_a_note_survives_a_failed_push(tmp_path, monkeypatch):
    """A notification is not a record. The note is the half that has to
    survive — a webhook that was deleted must not cost the thing you were
    trying not to forget."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "n.db")
    dbm.init_db()
    dbm.add_note("the pictures on 105 are wrong", author="Daniel",
                 notified=False)
    rows = dbm.notes()
    assert len(rows) == 1
    assert rows[0]["notified"] == 0, "it records that nothing went out"
    assert rows[0]["text"] == "the pictures on 105 are wrong"


def test_the_store_happens_even_when_no_ping_was_asked_for():
    """The ping is a checkbox, not the point. A note nobody was pinged about
    is still a note."""
    from pathlib import Path
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    body = src.split("def message_send", 1)[1].split("\n@app.route", 1)[0]
    assert "dbm.add_note(" in body
    # add_note is not inside the `if want_ping` branch
    ping_at, save_at = body.index("if want_ping:"), body.index("dbm.add_note(")
    assert save_at > ping_at
    assert body[ping_at:save_at].count("\n    dbm.add_note") == 0


def test_open_notes_are_oldest_first_and_done_notes_newest(tmp_path, monkeypatch):
    """The orders differ on purpose. An open list is a queue — what has waited
    longest is most likely to be forgotten. A done list is a record, and what
    you want from a record is what happened last."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "o.db")
    dbm.init_db()
    ids = [dbm.add_note(f"note {i}", author="D") for i in range(3)]
    assert [n["id"] for n in dbm.notes()] == ids
    for i in ids:
        dbm.finish_note(i, by="D")
    assert [n["id"] for n in dbm.notes(done=True)] == list(reversed(ids))


def test_two_people_cannot_both_claim_the_same_note(tmp_path, monkeypatch):
    """Whoever got there first keeps the credit; the second click is refused
    rather than silently overwriting who actually did it."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "race.db")
    dbm.init_db()
    n = dbm.add_note("raise cfg to 2", author="Daniel")
    assert dbm.finish_note(n, by="elroee") is True
    assert dbm.finish_note(n, by="Daniel") is False
    assert dbm.notes(done=True)[0]["done_by"] == "elroee"


def test_a_note_can_come_back(tmp_path, monkeypatch):
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "re.db")
    dbm.init_db()
    n = dbm.add_note("check the Bretton Woods hook", author="D")
    dbm.finish_note(n, by="D")
    assert dbm.open_note_count() == 0
    assert dbm.reopen_note(n) is True
    assert dbm.open_note_count() == 1
    assert dbm.notes()[0]["done_by"] is None


def test_the_note_count_survives_a_database_that_will_not_answer(monkeypatch):
    """It is read to badge a nav item on every page; a hiccup costs the badge,
    not the page."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "_conn", lambda: (_ for _ in ()).throw(RuntimeError))
    assert dbm.open_note_count() == 0


def test_the_sweep_clears_finished_projects_and_spares_the_open_one(
        tmp_path, monkeypatch):
    """The backlog that already exists. Retiring on close fixes it going
    forward and does nothing about the seven reads, two galleries and three
    scripts already piled up from sittings that ended days ago."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "sweep.db")
    dbm.init_db()

    done = dbm.new_project(channel="c", niche="n")
    live = dbm.new_project(channel="c", niche="n")
    for pid in (done, live):
        for style in ("warning", "shocking_stat", "counterintuitive"):
            dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                               topic="T", hook_style=style, hook="h",
                               script="s", score=8, project_id=pid)
    dbm.update_project(done, status="abandoned")
    assert len(dbm.candidates(status="pending", limit=100)) == 6

    n = dbm.retire_all_stale_options()

    assert n == 3
    left = dbm.candidates(status="pending", limit=100)
    assert len(left) == 3
    assert {c["project_id"] for c in left} == {live}, (
        "the project you still have open must be untouched")


def test_the_sweep_clears_rows_that_belong_to_no_project(tmp_path, monkeypatch):
    """These are the pile. In the owner's database only 4 of 30 waiting
    options belonged to a project at all — the rest came from the older
    per-stage path, which starts a stage without opening a project, so there
    is nothing to finish and nothing that ever takes them out of the queues.

    An earlier version of this spared them, reasoning that a queue somebody
    might still be working from should not be swept. Wrong call: they ARE the
    pile, and the protection that matters is the project open right now."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "orphan.db")
    dbm.init_db()
    dbm.save_candidate(proposal_id=7, channel="c", niche="n", topic="T",
                       hook_style="warning", hook="h", script="s", score=8)
    assert dbm.retire_all_stale_options() == 1
    assert dbm.candidates(status="pending", limit=10) == []
    assert len(dbm.candidates(status=None, limit=10)) == 1, "kept, not deleted"


def test_the_sweep_spares_the_project_you_have_open(tmp_path, monkeypatch):
    """Including its gallery and its reads, which hang off it through the
    chosen script rather than carrying a project id of their own."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "live.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    cid = dbm.save_candidate(proposal_id=None, channel="c", niche="n",
                             topic="T", hook_style="warning", hook="h",
                             script="s", score=8, project_id=pid)
    sid = dbm.save_gallery_set(candidate_id=cid, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    dbm.save_voice_take(set_id=sid, channel="c", topic="T", tone="calm",
                        text="t", path="/a.mp3")
    # ...and a stray from an older sitting, with no project.
    dbm.save_candidate(proposal_id=9, channel="c", niche="n", topic="Old",
                       hook_style="warning", hook="h", script="s", score=8)

    assert dbm.retire_all_stale_options() == 1

    assert [c["id"] for c in dbm.candidates(status="pending", limit=10)] == [cid]
    assert [g["id"] for g in dbm.gallery_sets(status="pending")] == [sid]
    assert len(dbm.voice_takes(set_id=sid, status="pending")) == 1


def test_the_sweep_is_safe_to_run_twice(tmp_path, monkeypatch):
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "twice.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    dbm.save_candidate(proposal_id=None, channel="c", niche="n", topic="T",
                       hook_style="warning", hook="h", script="s", score=8,
                       project_id=pid)
    dbm.update_project(pid, status="abandoned")
    assert dbm.retire_all_stale_options() == 1
    assert dbm.retire_all_stale_options() == 0


def test_nothing_is_deleted_by_the_sweep(tmp_path, monkeypatch):
    """A retired option is still half of a labelled pair, which is why the
    losers are kept at all."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "keep.db")
    dbm.init_db()
    pid = dbm.new_project(channel="c", niche="n")
    dbm.save_candidate(proposal_id=None, channel="c", niche="n", topic="T",
                       hook_style="warning", hook="h", script="s", score=8,
                       project_id=pid)
    dbm.update_project(pid, status="abandoned")
    dbm.retire_all_stale_options()
    assert len(dbm.candidates(project_id=pid, status=None)) == 1


def test_the_tidy_offer_only_appears_when_there_is_something_to_tidy():
    """A permanently visible "tidy up" button is a chore the page invents; one
    that appears because seven reads really are waiting is the page telling
    you something true."""
    from pathlib import Path
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    body = src.split("def _stale_notice", 1)[1].split("\ndef ", 1)[0]
    assert "if not stale:" in body and 'return ""' in body
    assert "/create/tidy" in body


# ── the drawing room ─────────────────────────────────────────────────────────

def _drawn(dbm, sid, n, seconds_apart=19):
    """n pictures, evenly spaced, ending now."""
    with dbm._conn() as c:
        for i in range(n):
            c.execute(
                "INSERT INTO gallery_images (set_id,variant,beat_index,path,"
                "prompt,seed,created_at) VALUES (?,?,?,?,?,?,datetime('now',?))",
                (sid, i % 2, i // 2, f"/{i}.png", "p", 1,
                 f"-{(n - i) * seconds_apart} seconds"))


def test_the_estimate_is_measured_from_the_pictures_not_the_set_row(
        tmp_path, monkeypatch):
    """THE NUMBER THAT WAS WRONG ON SCREEN. The old estimate divided the time
    since the SET ROW was written by the pictures drawn — and that row exists
    before three voice takes and a storyboard call. Nine of thirty-eight drawn
    reported ELEVEN HOURS remaining while ComfyUI was visibly doing one every
    nineteen seconds."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "eta.db")
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=1, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    # The set row is three hours old; drawing began three minutes ago.
    with dbm._conn() as c:
        c.execute("UPDATE gallery_sets SET created_at=datetime('now','-3 hours') "
                  "WHERE id=?", (sid,))
    dbm.set_gallery_beats(sid, 19)
    _drawn(dbm, sid, 9)

    rate = dbm.gallery_draw_rate(sid)

    assert 18 <= rate <= 20, f"measured {rate}s, expected about 19"
    eta_minutes = rate * (38 - 9) / 60
    assert eta_minutes < 15, f"{eta_minutes:.0f} min — the set row leaked in again"


def test_no_estimate_at_all_rather_than_a_made_up_one(tmp_path, monkeypatch):
    """A set drawn before created_at existed, or one with a single picture,
    has nothing to measure. The page shows no estimate instead of inventing."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "none.db")
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=1, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    assert dbm.gallery_draw_rate(sid) is None
    _drawn(dbm, sid, 1)
    assert dbm.gallery_draw_rate(sid) is None, "one picture is not a rate"


def test_the_rate_follows_a_machine_that_slows_down(tmp_path, monkeypatch):
    """A trailing window, not the whole run: a box that starts throttling
    halfway through should move the estimate rather than be averaged away by
    the fast start."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "slow.db")
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=1, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    with dbm._conn() as c:
        # ten fast ones long ago, then six slow ones just now
        for i in range(10):
            c.execute("INSERT INTO gallery_images (set_id,variant,beat_index,"
                      "path,prompt,seed,created_at) VALUES (?,?,?,?,?,?,"
                      "datetime('now',?))",
                      (sid, 0, i, f"/f{i}.png", "p", 1, f"-{2000 - i*5} seconds"))
        for i in range(6):
            c.execute("INSERT INTO gallery_images (set_id,variant,beat_index,"
                      "path,prompt,seed,created_at) VALUES (?,?,?,?,?,?,"
                      "datetime('now',?))",
                      (sid, 1, i, f"/s{i}.png", "p", 1, f"-{600 - i*100} seconds"))
    assert dbm.gallery_draw_rate(sid) > 50, "the recent slowdown has to show"


def test_the_room_sends_every_slot_including_the_empty_ones(tmp_path, monkeypatch):
    """The shape of the job is visible from the first poll rather than growing
    a row at a time out of nothing."""
    import db_manager as dbm
    import dashboard
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "slots.db")
    monkeypatch.setattr(dashboard, "db_manager", dbm)
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=1, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    dbm.set_gallery_beats(sid, 5)
    _drawn(dbm, sid, 3)

    d = dashboard.app.test_client().get(f"/api/drawing/{sid}").get_json()

    assert d["total"] == 10 and d["done"] == 3
    assert len(d["slots"]) == 10
    assert sum(1 for s in d["slots"] if s["image"]) == 3
    assert d["finished"] is False


def test_the_room_knows_when_it_is_finished(tmp_path, monkeypatch):
    import db_manager as dbm
    import dashboard
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "fin.db")
    monkeypatch.setattr(dashboard, "db_manager", dbm)
    dbm.init_db()
    sid = dbm.save_gallery_set(candidate_id=1, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    dbm.set_gallery_beats(sid, 2)
    _drawn(dbm, sid, 4)
    d = dashboard.app.test_client().get(f"/api/drawing/{sid}").get_json()
    assert d["finished"] is True
    assert d["eta_seconds"] is None, "nothing left to wait for"


def test_the_status_bar_ignores_the_dashboards_own_log(tmp_path, monkeypatch):
    """THE LINES THAT WERE ON SCREEN. serve.ps1 redirects the dashboard's
    stdout to logs/dashboard.log and Flask writes an access line for every
    poll — including the poll that draws the bar — so it was always the most
    recently modified file and always won. The owner's status bar reported
    `GET /api/status HTTP/1.1 200` back at him while a gallery rendered."""
    import dashboard, paths
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    (tmp_path / "galleries_1.log").write_text("[galleries] drawing\n",
                                              encoding="utf-8")
    (tmp_path / "dashboard.log").write_text(
        '127.0.0.1 - - "GET /api/status HTTP/1.1" 200 -\n', encoding="utf-8")
    lines = dashboard._newest_log_lines()
    assert lines == ["[galleries] drawing"]
    assert not any("api/status" in ln for ln in lines)


def test_the_launch_waits_for_the_row_before_redirecting(tmp_path, monkeypatch):
    """gallery_variants writes its set row as its first act, but it is a
    separate process. Redirecting immediately would land on a page for a set
    that does not exist; guessing the next id is wrong the moment two things
    run at once."""
    import db_manager as dbm
    import dashboard
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "wait.db")
    dbm.init_db()
    assert dashboard._await_new_gallery(None) is None
    assert dashboard._await_new_gallery(42, timeout=0.5) is None, (
        "a launch that never wrote a row must not send you to an empty room")
    sid = dbm.save_gallery_set(candidate_id=42, channel="c", niche="n",
                               topic="T", script_file="s.txt", n_variants=2)
    assert dashboard._await_new_gallery(42, timeout=2) == sid
