"""Recording a video that was published by hand.

THE REPORT: "I uploaded a few videos just manually — we need to fix the
analytics so we can track the info better."

That is the whole learning loop in one sentence. analytics_fetcher only looks
at rows carrying a youtube_id, and only the pipeline's own uploader ever set
one. Publishing by hand is the correct thing to do while nothing auto-uploads,
and every one of those videos was invisible: no metrics fetched, no views
recorded, so feedback_analyzer had no winners to learn hooks from, and every
quality judgement in this pipeline — the hook scorer, the critic, the 8/10
threshold — stayed a guess about what works.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import dashboard  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


# ── the link people actually paste ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3s",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
    "   dQw4w9WgXcQ   ",
])
def test_the_id_is_found_in_whatever_form_it_arrives(text):
    """Nobody should extract an id from a share link by hand, and the person
    doing it at 1am will get it wrong."""
    assert db_manager.extract_youtube_id(text) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("text", ["", None, "not a link", "https://youtube.com/"])
def test_nonsense_is_refused_rather_than_stored(text):
    assert db_manager.extract_youtube_id(text) is None


# ── the row the video was missing ───────────────────────────────────────────

def test_marking_published_makes_a_video_trackable(db):
    vid = db.save_video("money_history", "hook", "scene", "/tmp/a.mp4")
    assert db.get_recent_tracked_videos(days=30) == []

    assert db.mark_published(vid, "https://youtu.be/dQw4w9WgXcQ") is True
    tracked = db.get_recent_tracked_videos(days=30)
    assert [t["youtube_id"] for t in tracked] == ["dQw4w9WgXcQ"]


def test_a_manual_publish_counts_as_approved(db):
    """A manual upload is not a lesser kind of publish."""
    vid = db.save_video("money_history", "hook", "scene", "/tmp/a.mp4")
    db.mark_published(vid, "dQw4w9WgXcQ")
    with db._conn() as c:
        status = c.execute("SELECT upload_status FROM videos WHERE id=?",
                           (vid,)).fetchone()[0]
    assert status == "approved"


def test_an_unparseable_link_changes_nothing(db):
    vid = db.save_video("money_history", "hook", "scene", "/tmp/a.mp4")
    assert db.mark_published(vid, "my video") is False
    assert db.get_recent_tracked_videos(days=30) == []


# ── how much of the loop is real ────────────────────────────────────────────

def test_published_without_metrics_is_the_honest_gap(db):
    """A youtube_id means a video CAN be tracked; a metrics row means it has
    been. The dashboard could show 79 videos and imply a working channel
    while not one had a view count attached."""
    a = db.save_video("money_history", "a", "s", "/tmp/a.mp4")
    b = db.save_video("money_history", "b", "s", "/tmp/b.mp4")
    db.save_video("money_history", "c", "s", "/tmp/c.mp4")   # never published
    db.mark_published(a, "dQw4w9WgXcQ")
    db.mark_published(b, "AAAAAAAAAAA")

    assert {r["id"] for r in db.published_without_metrics()} == {a, b}
    db.save_metrics(a, views=120, watch_pct=41.0, ctr=0.0, likes=3)
    assert [r["id"] for r in db.published_without_metrics()] == [b]


# ── the dashboard ───────────────────────────────────────────────────────────

@pytest.fixture
def client(db):
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def test_the_form_records_a_pasted_link(client, db):
    vid = db.save_video("money_history", "hook", "scene", "/tmp/a.mp4")
    r = client.post(f"/video/{vid}/published",
                    data={"youtube": "https://www.youtube.com/shorts/dQw4w9WgXcQ"})
    assert "msg=" in r.headers["Location"]
    assert db.get_recent_tracked_videos(days=30)[0]["youtube_id"] == "dQw4w9WgXcQ"


def test_a_bad_paste_says_what_it_wanted(client, db):
    vid = db.save_video("money_history", "hook", "scene", "/tmp/a.mp4")
    r = client.post(f"/video/{vid}/published", data={"youtube": "my video"})
    assert "error=" in r.headers["Location"]


def test_the_tracking_page_separates_can_be_tracked_from_is_tracked(client, db):
    a = db.save_video("money_history", "a", "s", "/tmp/a.mp4")
    db.save_video("money_history", "b", "s", "/tmp/b.mp4")
    db.mark_published(a, "dQw4w9WgXcQ")
    page = client.get("/tracking").get_data(as_text=True)
    assert "published (trackable)" in page
    assert "with view counts" in page
    assert "never published" in page


def test_a_channel_with_nothing_published_is_told_plainly(client, db):
    db.save_video("money_history", "a", "s", "/tmp/a.mp4")
    page = client.get("/tracking").get_data(as_text=True)
    assert "learning loop has never had any data" in page


def test_tracking_is_in_the_nav():
    assert any(href == "/tracking" for href, _l, _p in dashboard.NAV_ITEMS)


# ── closing the loop from the page ──────────────────────────────────────────

def test_the_page_can_start_a_fetch(client, db, monkeypatch):
    """Both halves of the loop needed a script run by hand and knowledge that
    they existed — the same gap the settings page had."""
    started = {}

    class _P:
        def __init__(self, *a, **k):
            started["cmd"] = a[0]

    monkeypatch.setattr(dashboard.subprocess, "Popen", _P)
    r = client.post("/tracking/fetch")
    assert "msg=" in r.headers["Location"]
    joined = " ".join(started["cmd"])
    assert "analytics_fetcher" in joined and "feedback_analyzer" in joined


def test_the_fetch_runs_out_of_process(client, db, monkeypatch):
    """This Flask app is single-threaded on purpose; a YouTube round-trip per
    video would freeze every other page for its duration."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("def tracking_fetch")[1].split("@app.route")[0]
    assert "Popen" in block


def test_the_first_fetch_is_honest_about_needing_a_browser(client, db, monkeypatch):
    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *a, **k: None)
    r = client.post("/tracking/fetch")
    assert "sign-in" in r.headers["Location"] or "sign" in r.headers["Location"]


def test_it_says_how_far_off_the_learning_threshold_is(client, db):
    """feedback_analyzer refuses to draw conclusions from fewer than three
    measured videos. An empty section that does not say why looks broken."""
    a = db.save_video("money_history", "a", "s", "/tmp/a.mp4")
    db.mark_published(a, "dQw4w9WgXcQ")
    db.save_metrics(a, views=10, watch_pct=30.0, ctr=0.0, likes=1)
    page = client.get("/tracking").get_data(as_text=True)
    assert "there is 1" in page
    assert "guess about what works" in page


def test_learned_hooks_are_shown_when_they_exist(client, db, monkeypatch):
    monkeypatch.setattr(dashboard, "_learnings", lambda channel=None: {
        "winning_hooks": ["The coin that broke a kingdom"],
        "losing_hooks": ["A brief history of money"]})
    page = client.get("/tracking").get_data(as_text=True)
    assert "The coin that broke a kingdom" in page
    assert "A brief history of money" in page
    assert "the loop actually closing" in page


def test_a_missing_learnings_file_is_not_an_error(db):
    assert dashboard._learnings("nope-not-a-channel") == {}
