"""The agent that watches the neighbours and proposes what to make.

WHAT "RESEARCHING FOR DAYS" MEANS HERE, since the tempting reading is wrong. It
is not a model thinking for hours — that costs a great deal, has no ceiling,
and does not improve with time because nothing new arrives between one hour and
the next. It is a small cheap pass on a schedule that WRITES DOWN WHAT IT SAW,
so the same competitor video seen on Monday at 4,000 views and Thursday at
40,000 becomes a fact no single pass could have had.

Most of what is tested below is the bounds. An agent with a model, a schedule
and no ceiling is a runaway bill and a fight over the GPU, and the difference
between a useful autonomous process and a dangerous one is entirely in the
places it refuses to act.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import scout  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


# ── not proposing what has already been made ─────────────────────────────────

def test_a_topic_already_covered_is_recognised_by_its_words():
    """Two videos about the same event never share a title, so exact matching
    would catch nothing. The cost of a false positive is one skipped idea; the
    cost of a false negative is a duplicate video."""
    assert scout.already_covered(
        "The Man Who Bought Manhattan For $24",
        ["How Manhattan Was Bought For Almost Nothing"])


def test_a_genuinely_new_topic_is_not_blocked():
    assert not scout.already_covered(
        "The Panic of 1893",
        ["How Manhattan Was Bought", "The Tulip Bubble Explained"])


def test_a_title_with_nothing_in_it_is_treated_as_covered():
    """A title that is all stopwords gives the check nothing to compare, and
    proposing on no information is worse than skipping."""
    assert scout.already_covered("How and why?", [])


def test_choose_takes_the_strongest_uncovered_candidate():
    candidates = [
        {"title": "The Tulip Bubble", "outperformance": 9.0},
        {"title": "The Panic of 1893", "outperformance": 4.0},
    ]
    pick = scout.choose(candidates, ["Tulip Bubble Mania Explained"])
    assert pick["title"] == "The Panic of 1893"


def test_choose_returns_nothing_when_everything_is_covered():
    """A real answer, and the scout says so rather than proposing a duplicate
    because it had to propose something."""
    assert scout.choose([{"title": "The Panic of 1893"}],
                        ["The Panic of 1893, explained"]) is None


def test_pending_proposals_count_as_covered(db):
    """A proposal still waiting for approval is exactly as much of a duplicate
    as a finished video — without this the scout proposes the same idea every
    four hours until someone reads the queue."""
    db.save_proposal(channel="c", niche="n", topic="The Panic of 1893",
                     hook="h", script="s", score=8, evidence="e")
    assert "The Panic of 1893" in scout._made_titles()


# ── the bounds ───────────────────────────────────────────────────────────────

def test_it_stops_when_the_queue_is_full(db, monkeypatch):
    """An agent that fills a queue nobody has read is generating work, not
    doing it."""
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "2")
    for i in range(2):
        db.save_proposal(channel="c", niche="n", topic=f"t{i}", hook="h",
                         script="s", score=8, evidence="e")
    why = scout.blocked()
    assert "already waiting" in why


def test_it_stops_when_the_day_is_spent(db, monkeypatch):
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "99")
    monkeypatch.setenv("RUFUS_SCOUT_MAX_COST", "0.10")
    db.save_proposal(channel="c", niche="n", topic="t", hook="h", script="s",
                     score=8, evidence="e", cost_usd=0.25)
    assert "spent on proposals today" in scout.blocked()


def test_it_stands_down_while_a_render_holds_the_gpu(db, monkeypatch):
    """The writing phase competes for the same card when a local model serves
    it, and a video already rendering is worth more than a proposal."""
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "99")
    monkeypatch.setenv("RUFUS_SCOUT_MAX_COST", "99")
    import run_review
    monkeypatch.setattr(run_review, "_gpu_is_busy", lambda: "main_en")
    assert "using the GPU" in scout.blocked()


def test_an_unblocked_scout_says_nothing(db, monkeypatch):
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "99")
    monkeypatch.setenv("RUFUS_SCOUT_MAX_COST", "99")
    import run_review
    monkeypatch.setattr(run_review, "_gpu_is_busy", lambda: "")
    assert scout.blocked() == ""


def test_a_junk_ceiling_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "lots")
    monkeypatch.setenv("RUFUS_SCOUT_MAX_COST", "cheap")
    assert scout.max_pending() == scout.DEFAULT_MAX_PENDING
    assert scout.max_cost() == scout.DEFAULT_MAX_COST


# ── the pass ─────────────────────────────────────────────────────────────────

def _unblocked(monkeypatch):
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "99")
    monkeypatch.setenv("RUFUS_SCOUT_MAX_COST", "99")
    import run_review
    monkeypatch.setattr(run_review, "_gpu_is_busy", lambda: "")


def test_a_blocked_pass_spends_nothing(db, monkeypatch):
    monkeypatch.setenv("RUFUS_SCOUT_MAX_PENDING", "1")
    db.save_proposal(channel="c", niche="n", topic="t", hook="h", script="s",
                     score=8, evidence="e")
    called = []
    monkeypatch.setattr(scout, "observe_and_remember",
                        lambda: called.append(1) or 0)
    out = scout.pass_once()
    assert out["skipped"] and not called
    assert out["proposal_id"] is None


def test_a_quiet_week_is_a_real_answer(db, monkeypatch, capsys):
    """Nothing outperforming is not a failure and must not be reported as
    one."""
    _unblocked(monkeypatch)
    monkeypatch.setattr(scout, "observe_and_remember", lambda: 12)
    out = scout.pass_once()
    assert out["proposal_id"] is None
    assert "quiet week" in out["skipped"]


def test_a_dry_run_chooses_and_spends_nothing(db, monkeypatch, capsys):
    """The step that makes the CHOOSING judgeable before any money goes on
    prose."""
    _unblocked(monkeypatch)
    monkeypatch.setattr(scout, "observe_and_remember", lambda: 1)
    db.record_observations([{
        "video_id": "v1", "channel_id": "c", "channel_title": "Neighbour",
        "title": "The Panic of 1893", "published_at": "", "views": 90_000,
        "channel_median": 10_000, "outperformance": 9.0}])
    subject_calls = []
    monkeypatch.setattr(scout, "subject_of",
                        lambda t, n: subject_calls.append(t) or t)

    out = scout.pass_once(dry_run=True)
    assert out["candidate"]["title"] == "The Panic of 1893"
    assert out["proposal_id"] is None
    assert not subject_calls, "a dry run must not call the model"
    assert "9.0x" in capsys.readouterr().out


def test_a_proposal_carries_the_evidence_that_chose_it(db, monkeypatch):
    """A proposal without its evidence is an instruction, and an instruction
    from an agent is the thing a person cannot audit."""
    _unblocked(monkeypatch)
    monkeypatch.setattr(scout, "observe_and_remember", lambda: 1)
    db.record_observations([{
        "video_id": "v1", "channel_id": "c", "channel_title": "Neighbour",
        "title": "The Panic of 1893", "published_at": "", "views": 90_000,
        "channel_median": 10_000, "outperformance": 9.0}])
    monkeypatch.setattr(scout, "subject_of", lambda t, n: "Panic of 1893")

    import research
    import script_writer
    monkeypatch.setattr(research, "get_seed",
                        lambda niche, topic=None: {"content": "a real source",
                                                   "title": topic})
    monkeypatch.setattr(script_writer, "preanalyze",
                        lambda seed, scene="": ("analysis", "run1", 0.01))
    monkeypatch.setattr(script_writer, "write_script_until_good",
                        lambda scene, seed=None, precomputed_analysis=None,
                        run_id=None: {"script": "the script", "hook": "the hook",
                                      "score": 9, "cost_usd": 0.04})

    out = scout.pass_once()
    assert out["proposal_id"]
    row = db.proposals()[0]
    assert row["score"] == 9
    assert row["topic"] == "Panic of 1893"
    assert "Neighbour" in row["evidence"]
    assert "9.0x" in row["evidence"]
    assert row["cost_usd"] == pytest.approx(0.05)


def test_a_writer_failure_is_a_skipped_pass_not_a_crash(db, monkeypatch):
    """This runs in a scheduled task every few hours. A raise is a dead agent
    and a log nobody reads."""
    _unblocked(monkeypatch)
    monkeypatch.setattr(scout, "observe_and_remember", lambda: 1)
    db.record_observations([{
        "video_id": "v1", "channel_id": "c", "channel_title": "N",
        "title": "The Panic of 1893", "published_at": "", "views": 9,
        "channel_median": 1, "outperformance": 9.0}])
    monkeypatch.setattr(scout, "subject_of", lambda t, n: "Panic")
    import research
    monkeypatch.setattr(research, "get_seed", lambda *a, **k: (_ for _ in ())
                        .throw(RuntimeError("no article")))
    out = scout.pass_once()
    assert out["proposal_id"] is None
    assert "no article" in out["skipped"]


def test_observing_never_raises(monkeypatch):
    import competitors
    monkeypatch.setattr(competitors, "observe",
                        lambda: (_ for _ in ()).throw(RuntimeError("api down")))
    assert scout.observe_and_remember() == 0


# ── the memory ───────────────────────────────────────────────────────────────

def test_observations_accumulate_rather_than_overwrite(db):
    """The one thing that makes days of watching different from one pass: the
    same video at 4,000 views and later at 40,000 is the fact worth having,
    and an UPDATE would throw the first half of it away."""
    for views in (4_000, 40_000):
        db.record_observations([{
            "video_id": "v1", "channel_id": "c", "channel_title": "N",
            "title": "T", "published_at": "", "views": views,
            "channel_median": 1_000, "outperformance": views / 1_000}])
    rows = db.rising(min_outperformance=2.0)
    assert len(rows) == 1, "one row per video, not one per sighting"
    assert rows[0]["sightings"] == 2
    assert rows[0]["views"] == 40_000


def test_a_weak_observation_is_not_a_candidate(db):
    db.record_observations([{
        "video_id": "v1", "channel_id": "c", "channel_title": "N", "title": "T",
        "published_at": "", "views": 1_100, "channel_median": 1_000,
        "outperformance": 1.1}])
    assert db.rising(min_outperformance=2.0) == []


def test_the_subject_falls_back_to_the_title_without_a_model(monkeypatch):
    """A model outage costs precision, not the whole pass — the title is what
    a lookup would have received anyway."""
    import llm
    monkeypatch.setattr(llm, "usable", lambda: False)
    assert scout.subject_of("The Panic of 1893", "money_history") == \
        "The Panic of 1893"
