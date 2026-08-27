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
    page = client.get("/measure").get_data(as_text=True)
    assert "published (trackable)" in page
    assert "with view counts" in page
    assert "never published" in page


def test_a_channel_with_nothing_published_is_told_plainly(client, db):
    db.save_video("money_history", "a", "s", "/tmp/a.mp4")
    page = client.get("/measure").get_data(as_text=True)
    assert "learning loop has never had any data" in page


def test_tracking_is_reachable_as_a_section():
    """REACHABLE, WHICH IS NOW A DIFFERENT ASSERTION THAN IT WAS.

    This page is a SECTION of /measure rather than a tab of its own — four
    pages asking variations of "what does the data say" was four places to
    look for one answer. What has to stay true is that the section is reachable
    and that the old link still leads to it, because a tidy-up that breaks
    every bookmark is not a tidy-up.
    """
    assert any(href == "/measure" for href, _l, _p in dashboard.NAV_ITEMS)
    assert ("is-the-loop-closed", "Is the loop closed", "_tracking_body") in dashboard._MEASURE_SECTIONS


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
    page = client.get("/measure").get_data(as_text=True)
    assert "there is 1" in page
    assert "guess about what works" in page


def test_learned_hooks_are_shown_when_they_exist(client, db, monkeypatch):
    monkeypatch.setattr(dashboard, "_learnings", lambda channel=None: {
        "winning_hooks": ["The coin that broke a kingdom"],
        "losing_hooks": ["A brief history of money"]})
    page = client.get("/measure").get_data(as_text=True)
    assert "The coin that broke a kingdom" in page
    assert "A brief history of money" in page
    assert "the loop actually closing" in page


def test_a_missing_learnings_file_is_not_an_error(db):
    assert dashboard._learnings("nope-not-a-channel") == {}


def test_one_youtube_id_belongs_to_one_video(tmp_path, monkeypatch):
    """THE DAMAGE THIS PREVENTS. Six rows in the owner's real database carry
    the id kGVAHaObJ38 — six different mp4s, one link pasted six times. It is
    an easy mistake from a phone and there was no guard.

    The harm is not the wrong link. Analytics joins metrics on this column, so
    all six were credited with a seventh video's views: identical counts, a
    watch percentage of zero, across scripts with nothing in common. Five
    videos that were never published looked published and performed like a
    video they are not — data that teaches the wrong lesson, which is worse
    than no data."""
    import pytest
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "dup.db")
    dbm.init_db()
    a = dbm.save_video(niche="n", script_hook="a", scene_desc="d",
                       video_file="a.mp4", score=8)
    b = dbm.save_video(niche="n", script_hook="b", scene_desc="d",
                       video_file="b.mp4", score=8)
    assert dbm.mark_published(a, "https://youtu.be/kGVAHaObJ38")
    with pytest.raises(ValueError) as e:
        dbm.mark_published(b, "https://youtu.be/kGVAHaObJ38")
    assert str(a) in str(e.value), "the error has to name the row that owns it"

    with dbm._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM videos WHERE youtube_id=?",
                      ("kGVAHaObJ38",)).fetchone()[0]
    assert n == 1, "the second write must not have landed"


def test_re_recording_the_same_video_is_still_allowed(tmp_path, monkeypatch):
    """Correcting a link on the row that already owns it is not a duplicate."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "same.db")
    dbm.init_db()
    v = dbm.save_video(niche="n", script_hook="a", scene_desc="d",
                       video_file="a.mp4", score=8)
    assert dbm.mark_published(v, "https://youtu.be/kGVAHaObJ38")
    assert dbm.mark_published(v, "https://youtu.be/kGVAHaObJ38")


def test_the_audit_finds_ids_that_got_in_before_the_guard(tmp_path, monkeypatch):
    """A guard added today does not clean up the rows that predate it, and
    those rows still feed another video's views into every average."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "audit.db")
    dbm.init_db()
    ids = [dbm.save_video(niche="n", script_hook=str(i), scene_desc="d",
                          video_file=f"{i}.mp4", score=8) for i in range(3)]
    with dbm._conn() as c:          # written the way the old code would have
        for i in ids:
            c.execute("UPDATE videos SET youtube_id='kGVAHaObJ38' WHERE id=?", (i,))
    dupes = dbm.duplicate_youtube_ids()
    assert len(dupes) == 1
    assert dupes[0]["youtube_id"] == "kGVAHaObJ38"
    assert dupes[0]["count"] == 3
    assert sorted(dupes[0]["video_ids"]) == sorted(ids)
