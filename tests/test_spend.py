"""What a video costs, which nothing added up.

Cost was already recorded in three places — every topic proposal, every script
candidate, every writer attempt — and no query ever summed them, so "what does
one video cost me" was answered by opening the OpenAI dashboard and guessing
which charges belonged to which channel. That is the first question anyone asks
about a business.

The tests below are mostly about the number being honest rather than flattering:
what it excludes, what it includes that you might wish it did not, and what it
refuses to say when it cannot say it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "spend.db")
    db_manager.init_db()
    return db_manager


def test_an_empty_channel_reports_nothing_rather_than_zero_dollars(db):
    s = db.spend(30)
    assert s["total_usd"] == 0.0
    assert s["videos"] == 0
    assert s["per_video_usd"] is None, (
        "no videos means no per-video figure, not a division by zero and not a "
        "confident $0.00")


def test_all_three_places_a_dollar_is_recorded_are_counted(db):
    """Cost lands in three tables and a rollup that knew about two would be
    quietly low — the worst kind of wrong for a number somebody budgets on."""
    db.save_proposal(channel="c", niche="n", topic="T", evidence="e",
                     cost_usd=0.01)
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.02)
    db.save_attempt(run_id="r", niche="n", seed_type="s", phase="p",
                    attempt_n=1, cost_usd=0.03)
    s = db.spend(30)
    assert s["total_usd"] == pytest.approx(0.06)
    assert len(s["parts"]) == 3
    assert all(v > 0 for v in s["parts"].values())


def test_work_that_was_thrown_away_is_counted(db):
    """Three scripts written so a person could choose one cost three scripts.
    Charging the video only for the one that shipped would make the number look
    better than the bank statement."""
    ids = [db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                             hook_style=s, hook="h", script="s", score=8,
                             cost_usd=0.02)
           for s in ("warning", "shocking_stat", "counterintuitive")]
    db.choose_candidate(ids[0])
    assert db.spend(30)["total_usd"] == pytest.approx(0.06)


def test_the_per_video_figure_is_spend_over_videos_in_the_same_window(db):
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.10)
    for _ in range(4):
        db.save_video(niche="n", script_hook="h", scene_desc="s",
                      video_file="a.mp4", score=8)
    s = db.spend(30)
    assert s["videos"] == 4
    assert s["per_video_usd"] == pytest.approx(0.025)


def test_spend_with_no_finished_video_says_so_instead_of_dividing(db):
    """Money spent and nothing shipped is a real state — mid-project, or a run
    that failed — and it deserves a sentence rather than a number that cannot
    be computed."""
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.10)
    s = db.spend(30)
    assert s["total_usd"] == pytest.approx(0.10)
    assert s["per_video_usd"] is None


def test_the_window_actually_narrows(db):
    """A figure quoted as "last 30 days" that silently covers all time is the
    kind of wrong nobody checks."""
    import sqlite3
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.50)
    with sqlite3.connect(str(db.DB_FILE)) as c:
        c.execute("UPDATE script_candidates SET created_at = "
                  "datetime('now', '-90 days')")
    assert db.spend(30)["total_usd"] == 0.0
    assert db.spend(365)["total_usd"] == pytest.approx(0.50)


def test_one_list_of_which_tables_carry_cost():
    """"Which tables carry cost" is a fact about the schema, and a second copy
    of it is the one that stops being updated when a fourth is added."""
    assert len(db_manager.COST_TABLES) == 3
    tables = {t for t, _when, _label in db_manager.COST_TABLES}
    assert tables == {"proposals", "script_candidates", "script_attempts"}


def test_an_older_database_missing_a_table_still_reports_the_rest(db):
    """Report the two it has rather than nothing at all — somebody looking at
    this is usually trying to understand a bill."""
    import sqlite3
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.04)
    with sqlite3.connect(str(db.DB_FILE)) as c:
        c.execute("DROP TABLE script_attempts")
    s = db.spend(30)
    assert s["total_usd"] == pytest.approx(0.04)


# ── and it reaches the page where somebody would look ───────────────────────

def test_the_page_carries_the_number_and_its_limits(db):
    """A figure people quote has to carry its own limits. The pictures are
    drawn on this machine's own GPU and the voice is local, so they cost
    electricity and hours rather than dollars — a per-video figure that reads
    as "what a video costs" without saying that is misleading by omission."""
    import dashboard
    db.save_candidate(proposal_id=1, channel="c", niche="n", topic="T",
                      hook_style="warning", hook="h", script="s", score=8,
                      cost_usd=0.20)
    db.save_video(niche="n", script_hook="h", scene_desc="s",
                  video_file="a.mp4", score=8)
    panel = dashboard._spend_panel()
    assert "$0.20" in panel
    assert "per video" in panel
    assert "electricity" in panel, "the excluded cost has to be named"
    assert "all three" in panel, "counting the discarded scripts has to be said"


def test_an_unreadable_database_costs_the_panel_and_not_the_page(monkeypatch):
    """Every other panel on this page fails open; this one must too."""
    import dashboard, db_manager as dbm
    monkeypatch.setattr(dbm, "spend",
                        lambda days=30: (_ for _ in ()).throw(OSError("gone")))
    assert "could not be read" in dashboard._spend_panel()
