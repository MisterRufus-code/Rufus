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
    pid = _start(client)
    sid = db_manager.save_gallery_set(candidate_id=6, channel="c", niche="n",
                                      topic="T", script_file="s.txt",
                                      n_variants=2)
    db_manager.update_project(pid, title="T", script_id=6,
                              script_file="s.txt", stage="gallery")
    img = db_manager.save_gallery_image(set_id=sid, variant=0, beat_index=0,
                                        path="/a.png", prompt="p", seed=1)
    page = client.get(f"/create?project={pid}").get_data(as_text=True)
    assert f"/galleries/image/{img}" in page


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
    """created_at was read only on the not-yet-chosen path, so every set the
    project already pointed at reported no estimate at all."""
    import dashboard
    pid = _project(db)
    sid = db.save_gallery_set(candidate_id=8, channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    db.update_project(pid, script_id=8, gallery_id=sid, stage="gallery")
    for b in range(3):
        db.save_gallery_image(set_id=sid, variant=0, beat_index=b,
                              path=f"/a{b}.png", prompt="p", seed=1)
    prog = dashboard._project_progress(db.project(pid))
    assert prog["working"] is True
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
