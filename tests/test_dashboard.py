"""Tests for dashboard.py — the approval-queue + status dashboard.

Uses Flask's test client (no network, no running server) against an
isolated temp DB so these run exactly like the rest of the suite. The real
upload path (youtube_uploader.upload) is monkeypatched everywhere — these
tests never touch the network or Google APIs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import dashboard
import db_manager
import youtube_uploader


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "debug")
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def _seed(n_approved=2, n_pending=1):
    for i in range(n_approved):
        db_manager.save_video(niche="finance", script_hook=f"Hook {i}",
                              scene_desc="s", video_file=f"v{i}.mp4",
                              score=9, youtube_id=f"yt{i}", run_id=f"run{i}",
                              upload_status="approved")
    for i in range(n_pending):
        db_manager.save_video(niche="finance", script_hook=f"PendingHook {i}",
                              scene_desc="s", video_file=f"h{i}.mp4",
                              score=5, hold_reason="score 5/10 < 8/10 threshold",
                              run_id=f"pendingrun{i}")


# ── Index page ─────────────────────────────────────────────────────────────────

def test_index_empty_db_does_not_crash(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Nothing here" in r.data


def test_index_shows_stats_and_rows(client):
    _seed(n_approved=2, n_pending=1)
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Hook 0" in body
    assert "PendingHook 0" in body
    assert "pending" in body
    assert "approved" in body


def test_index_pending_section_lists_only_pending(client):
    approved_id = db_manager.save_video(niche="finance", script_hook="Hook A",
                                        scene_desc="s", video_file="v.mp4",
                                        score=9, youtube_id="yt0",
                                        upload_status="approved")
    pending_id = db_manager.save_video(niche="finance", script_hook="Hook B",
                                       scene_desc="s", video_file="h.mp4",
                                       score=5)
    r = client.get("/")
    body = r.data.decode()
    pending_section = body.split("Awaiting your review")[1].split("All recent videos")[0]
    assert f'/video/{pending_id}"' in pending_section
    assert f'/video/{approved_id}"' not in pending_section


def test_index_escapes_script_content_xss(client):
    db_manager.save_video(niche="finance", script_hook="<script>alert(1)</script>",
                          scene_desc="s", video_file="v.mp4", score=9)
    r = client.get("/")
    assert b"<script>alert(1)</script>" not in r.data
    assert b"&lt;script&gt;" in r.data


def test_index_channel_filter(client):
    db_manager.save_video(niche="finance", script_hook="ChA", scene_desc="s",
                          video_file="a.mp4", score=9, channel="chan_a")
    db_manager.save_video(niche="finance", script_hook="ChB", scene_desc="s",
                          video_file="b.mp4", score=9, channel="chan_b")
    r = client.get("/?channel=chan_a")
    body = r.data.decode()
    assert "ChA" in body
    assert "ChB" not in body


# ── Video detail page ────────────────────────────────────────────────────────

def test_video_detail_404_for_missing_id(client):
    r = client.get("/video/9999")
    assert r.status_code == 404


def test_video_detail_shows_score_breakdown(client):
    vid = db_manager.save_video(
        niche="finance", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=7, run_id="r1",
        criterion_scores={"specificity": 2, "hook": 2, "compression": 1,
                          "loop": 1, "human": 1},
        score_reasoning="TOTAL: 7/10 — decent but padded",
    )
    r = client.get(f"/video/{vid}")
    assert r.status_code == 200
    body = r.data.decode()
    assert "7/10" in body
    assert "decent but padded" in body


def test_video_detail_shows_pending_status_and_gate_note(client):
    vid = db_manager.save_video(
        niche="finance", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=5,
        hold_reason="factual integrity: invented figure",
    )
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert "pending review" in body
    assert "invented figure" in body


def test_video_detail_shows_approve_reject_buttons_when_pending(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9)
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert f'/video/{vid}/approve' in body
    assert f'/video/{vid}/reject' in body


def test_video_detail_no_actions_when_already_approved(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9, youtube_id="ytABC",
                                upload_status="approved")
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert f'/video/{vid}/approve' not in body
    assert "watch" in body   # link to the live video instead


def test_video_detail_edit_form_prefilled(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9,
                                title="My Title", description="My description here")
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert "My Title" in body
    assert "My description here" in body


def test_video_detail_lists_debug_assets_when_present(client, tmp_path):
    run_dir = dashboard.DEBUG_ROOT / "runX"
    run_dir.mkdir(parents=True)
    (run_dir / "script.txt").write_text("hello")
    (run_dir / "01.png").write_bytes(b"x" * 2000)

    vid = db_manager.save_video(niche="finance", script_hook="Hook",
                                scene_desc="s", video_file="v.mp4",
                                score=9, run_id="runX")
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert "script.txt" in body
    assert "01.png" in body


def test_video_detail_no_assets_message_when_absent(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook",
                                scene_desc="s", video_file="v.mp4",
                                score=9, run_id="run-not-on-disk")
    r = client.get(f"/video/{vid}")
    assert b"No debug artifacts" in r.data


# ── Approve (the ONLY path that actually uploads) ─────────────────────────────

def test_approve_uploads_and_marks_approved(client, tmp_path, monkeypatch):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake video bytes")
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file=str(video_file), score=9,
                                title="T", description="D", channel="main_en")

    captured = {}
    def fake_upload(path, script, thumbnail_path=None, metadata=None):
        captured["path"] = path
        captured["metadata"] = metadata
        return "https://youtube.com/shorts/abc123", "abc123"

    monkeypatch.setattr(youtube_uploader, "upload", fake_upload)
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)

    assert r.status_code == 200
    assert "Uploaded" in r.data.decode()
    assert captured["metadata"]["title"] == "T"
    assert captured["metadata"]["description"] == "D"

    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status, youtube_id FROM videos WHERE id=?",
                        (vid,)).fetchone()
    assert row == ("approved", "abc123")


def test_approve_refuses_when_already_uploaded(client, tmp_path):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9, youtube_id="already",
                                upload_status="approved")
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    assert "already uploaded" in r.data.decode()


def test_approve_fails_gracefully_when_video_file_missing(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="/nonexistent/v.mp4", score=9)
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    body = r.data.decode()
    assert "missing on disk" in body
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "pending"   # never flipped to approved


def test_approve_upload_exception_keeps_status_pending(client, tmp_path, monkeypatch):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file=str(video_file), score=9)

    def fake_upload(*a, **kw):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(youtube_uploader, "upload", fake_upload)
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    assert "quota exceeded" in r.data.decode()
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "pending"


def test_approve_404_for_missing_video(client):
    r = client.post("/video/9999/approve")
    assert r.status_code == 404


# ── Reject / un-reject ─────────────────────────────────────────────────────────

def test_reject_marks_rejected(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9)
    r = client.post(f"/video/{vid}/reject", follow_redirects=True)
    assert "rejected" in r.data.decode()
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "rejected"


def test_reject_twice_toggles_back_to_pending(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9)
    client.post(f"/video/{vid}/reject")
    r = client.post(f"/video/{vid}/reject", follow_redirects=True)
    assert "pending" in r.data.decode()
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "pending"


def test_reject_refuses_when_already_uploaded(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9, youtube_id="live",
                                upload_status="approved")
    client.post(f"/video/{vid}/reject")
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "approved"   # unchanged


# ── Edit (title/description) ────────────────────────────────────────────────────

def test_edit_updates_title_and_description(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9,
                                title="Old", description="Old desc")
    client.post(f"/video/{vid}/edit", data={"title": "New Title",
                                            "description": "New description"})
    with db_manager._conn() as c:
        row = c.execute("SELECT title, description FROM videos WHERE id=?",
                        (vid,)).fetchone()
    assert row == ("New Title", "New description")


def test_edit_escapes_xss_on_redisplay(client):
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9)
    client.post(f"/video/{vid}/edit", data={"title": "<script>x</script>",
                                            "description": "fine"})
    r = client.get(f"/video/{vid}")
    assert b"<script>x</script>" not in r.data
    assert b"&lt;script&gt;" in r.data


# ── Debug file serving (path-traversal guard) ─────────────────────────────────

def test_debug_file_serves_existing_file(client, tmp_path):
    run_dir = dashboard.DEBUG_ROOT / "runY"
    run_dir.mkdir(parents=True)
    (run_dir / "voiceover.mp3").write_bytes(b"fake-mp3-bytes")
    r = client.get("/debug/runY/voiceover.mp3")
    assert r.status_code == 200
    assert r.data == b"fake-mp3-bytes"


def test_debug_file_404_for_missing_run(client):
    r = client.get("/debug/does-not-exist/voiceover.mp3")
    assert r.status_code == 404


def test_debug_file_blocks_path_traversal(client, tmp_path):
    run_dir = dashboard.DEBUG_ROOT / "runZ"
    run_dir.mkdir(parents=True)
    (dashboard.DEBUG_ROOT / "secret.txt").write_text("nope")
    r = client.get("/debug/runZ/../secret.txt")
    assert r.status_code in (404, 403)


# ── Pure helper functions ─────────────────────────────────────────────────────

def test_stats_empty_db(client):   # client fixture isolates DB_FILE
    assert dashboard._stats() == {"total": 0, "avg_score": 0.0, "hold_rate": 0.0,
                                  "uploaded": 0, "held": 0, "pending": 0, "rejected": 0}


def test_stats_counts_by_status(client):
    _seed(n_approved=2, n_pending=1)
    db_manager.save_video(niche="finance", script_hook="R", scene_desc="s",
                          video_file="r.mp4", score=4, upload_status="rejected")
    stats = dashboard._stats()
    assert stats["pending"] == 1
    assert stats["uploaded"] == 2
    assert stats["rejected"] == 1
    assert stats["total"] == 4


def test_sparkline_svg_empty_returns_message():
    assert "No scored" in dashboard._sparkline_svg([])


def test_sparkline_svg_contains_points_for_each_score():
    svg = dashboard._sparkline_svg([5, 8, 10])
    assert svg.count("<circle") == 3


def test_score_color_thresholds():
    assert dashboard._score_color(9) == "#22c55e"
    assert dashboard._score_color(7) == "#eab308"
    assert dashboard._score_color(3) == "#ef4444"
    assert dashboard._score_color(None) == "#888"


def test_status_badge_variants():
    assert "approved" in dashboard._status_badge("approved")
    assert "rejected" in dashboard._status_badge("rejected")
    assert "pending" in dashboard._status_badge("pending")


def test_approve_restores_env_vars_after_upload(client, tmp_path, monkeypatch):
    """Real bug caught live: approve_video() used to mutate os.environ
    permanently, which leaked RUFUS_NICHE_OVERRIDE/RUFUS_CHANNEL into every
    later request in this same long-lived process (and broke an unrelated
    test suite run in the same pytest session)."""
    import os as _os
    monkeypatch.delenv("RUFUS_CHANNEL", raising=False)
    monkeypatch.delenv("RUFUS_NICHE_OVERRIDE", raising=False)

    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    vid = db_manager.save_video(niche="money_history", script_hook="Hook",
                                scene_desc="s", video_file=str(video_file),
                                score=9, channel="side_channel")

    captured = {}
    def fake_upload(path, script, thumbnail_path=None, metadata=None):
        captured["channel"] = _os.environ.get("RUFUS_CHANNEL")
        captured["niche"] = _os.environ.get("RUFUS_NICHE_OVERRIDE")
        return "https://youtube.com/shorts/x", "x"

    monkeypatch.setattr(youtube_uploader, "upload", fake_upload)
    client.post(f"/video/{vid}/approve")

    # correct values were used DURING the call...
    assert captured["channel"] == "side_channel"
    assert captured["niche"] == "money_history"
    # ...but nothing leaked into the process afterward
    assert "RUFUS_CHANNEL" not in _os.environ
    assert "RUFUS_NICHE_OVERRIDE" not in _os.environ


def test_approve_restores_env_vars_even_on_failure(client, tmp_path, monkeypatch):
    import os as _os
    monkeypatch.delenv("RUFUS_NICHE_OVERRIDE", raising=False)
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    vid = db_manager.save_video(niche="money_history", script_hook="Hook",
                                scene_desc="s", video_file=str(video_file), score=9)

    def fake_upload(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(youtube_uploader, "upload", fake_upload)
    client.post(f"/video/{vid}/approve")
    assert "RUFUS_NICHE_OVERRIDE" not in _os.environ
