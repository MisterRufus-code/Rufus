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


def test_approve_blocked_below_hard_score_floor(client, tmp_path, monkeypatch):
    """No video below the 7/10 hard floor can be approved, even by a human
    clicking the button — a misclick or a reviewer other than the owner
    (Tailscale-shared access) must not be able to publish a weak script."""
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file=str(video_file), score=6)

    called = []
    monkeypatch.setattr(youtube_uploader, "upload",
                        lambda *a, **k: called.append(1) or ("u", "i"))
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)

    assert "below the 7/10 minimum" in r.data.decode()
    assert not called   # upload() was never even attempted
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status, youtube_id FROM videos WHERE id=?",
                        (vid,)).fetchone()
    assert row == ("pending", None)


def test_approve_blocked_when_unscored(client, tmp_path):
    """A video with no score at all (scoring itself failed/skipped) is
    treated as unscored, not as passing — refuse rather than guess."""
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"x")
    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file=str(video_file), score=None)
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    assert "unscored" in r.data.decode()


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


# ── Failures page: crashed runs + rejected attempts browser ───────────────────

def test_failures_page_loads_empty(client):
    r = client.get("/failures")
    assert r.status_code == 200
    assert b"No crashed" in r.data
    assert b"No rejected attempts" in r.data


def test_failures_lists_orphaned_debug_run_not_in_db(client, tmp_path):
    """A run that started (RUFUS_DEBUG wrote files) but crashed before Step 6
    has NO videos row — it must still show up here, unlike everywhere else."""
    run_dir = dashboard.DEBUG_ROOT / "crashed-run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "script.txt").write_text("This script never made it to render.")
    (run_dir / "01.png").write_bytes(b"x" * 1000)

    r = client.get("/failures")
    body = r.data.decode()
    assert "crashed-run-1" in body
    assert "This script never made it to render." in body
    assert "01.png" in body


def test_failures_excludes_runs_that_did_reach_db(client, tmp_path):
    """A debug folder WITH a matching videos.run_id is a normal completed
    run, not a failure — must not appear in the crashed-runs section."""
    run_dir = dashboard.DEBUG_ROOT / "completed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "script.txt").write_text("finished fine")
    db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                          video_file="v.mp4", score=9, run_id="completed-run")

    r = client.get("/failures")
    body = r.data.decode()
    section = body.split("Crashed / incomplete runs")[1].split("Rejected script attempts")[0]
    assert "completed-run" not in section


def test_failures_shows_rejected_script_attempts(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="hook_gen", attempt_n=1, hook="A bad hook",
                            rejected_reason="banned phrase: crucial",
                            accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=2, body="Some rejected body text",
                            rejected_reason="specificity too low",
                            accepted=False)
    r = client.get("/failures")
    body = r.data.decode()
    assert "banned phrase: crucial" in body
    assert "specificity too low" in body
    assert "Some rejected body text" in body


def test_failures_excludes_accepted_attempts(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=1, body="A fine script",
                            total_score=9, accepted=True)
    r = client.get("/failures")
    assert b"A fine script" not in r.data


def test_failures_niche_filter(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="hook_gen", attempt_n=1, hook="Finance hook",
                            rejected_reason="reason A", accepted=False)
    db_manager.save_attempt(run_id="r2", niche="money_history", seed_type="wisdom",
                            phase="hook_gen", attempt_n=1, hook="History hook",
                            rejected_reason="reason B", accepted=False)
    r = client.get("/failures?niche=finance")
    body = r.data.decode()
    assert "Finance hook" in body
    assert "History hook" not in body


def test_failures_phase_filter(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="hook_gen", attempt_n=1, hook="A hook",
                            rejected_reason="hook reason", accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=1, body="A body",
                            rejected_reason="body reason", accepted=False)
    r = client.get("/failures?phase=hook_gen")
    body = r.data.decode()
    assert "hook reason" in body
    assert "body reason" not in body


def test_failures_escapes_xss_in_script_preview(client, tmp_path):
    run_dir = dashboard.DEBUG_ROOT / "xss-run"
    run_dir.mkdir(parents=True)
    (run_dir / "script.txt").write_text("<script>alert(1)</script>")
    r = client.get("/failures")
    assert b"<script>alert(1)</script>" not in r.data
    assert b"&lt;script&gt;" in r.data


def test_orphaned_debug_runs_handles_missing_directory(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "does-not-exist")
    assert dashboard._orphaned_debug_runs() == []


def test_nav_link_to_failures_present_on_homepage(client):
    r = client.get("/")
    assert b"/failures" in r.data


# ── Manual topic request (backlog item #6) ─────────────────────────────────────

def test_request_topic_launches_background_process(client, monkeypatch, tmp_path):
    import subprocess
    captured = {}
    class FakeProc:
        pass
    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)

    r = client.post("/request-topic", data={"topic": "Bretton Woods"},
                    follow_redirects=True)
    assert r.status_code == 200
    assert "Bretton Woods" in r.data.decode()
    assert "--topic" in captured["cmd"]
    assert "Bretton Woods" in captured["cmd"]
    assert captured["cmd"][0] == sys.executable
    assert str(tmp_path / "scripts" / "main.py") in captured["cmd"]


def test_request_topic_passes_channel_when_given(client, monkeypatch, tmp_path):
    import subprocess
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        class P: pass
        return P()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)

    client.post("/request-topic", data={"topic": "Tulip mania", "channel": "side_channel"})
    assert "--channel" in captured["cmd"]
    assert "side_channel" in captured["cmd"]


def test_request_topic_rejects_empty_topic(client):
    r = client.post("/request-topic", data={"topic": "  "}, follow_redirects=True)
    assert "topic is required" in r.data.decode()


def test_request_topic_handles_popen_failure_gracefully(client, monkeypatch, tmp_path):
    import subprocess
    def fake_popen(cmd, **kwargs):
        raise OSError("no such file")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)

    r = client.post("/request-topic", data={"topic": "Bretton Woods"},
                    follow_redirects=True)
    assert "failed to start" in r.data.decode()


def test_index_has_topic_request_form(client):
    r = client.get("/")
    body = r.data.decode()
    assert 'action="/request-topic"' in body
    assert 'name="topic"' in body


# ── Root-cause attribution (backlog: bottleneck breakdown) ────────────────────

def test_categorize_rejection_safety():
    assert dashboard._categorize_rejection("banned phrase: 'crucial'") == "safety"
    assert dashboard._categorize_rejection("hedging word: 'maybe'") == "safety"


def test_categorize_rejection_accuracy():
    assert dashboard._categorize_rejection("low specificity (0.20/25w, need >=1.0)") == "accuracy"
    assert dashboard._categorize_rejection("DISQUALIFIERS: NO SENSORY DETAIL") == "accuracy"


def test_categorize_rejection_weak_hook():
    assert dashboard._categorize_rejection("forbidden opener: 'did you know'") == "weak_hook"
    assert dashboard._categorize_rejection("hook too short (2 words, need >=4)") == "weak_hook"


def test_categorize_rejection_loose_structure():
    assert dashboard._categorize_rejection("loop no echo (second-to-last line shares no content tokens with hook)") == "loose_structure"
    assert dashboard._categorize_rejection("cadence: missing a short, punchy sentence") == "loose_structure"
    assert dashboard._categorize_rejection("sentences too long (avg 20.0 words, cap 14)") == "loose_structure"


def test_categorize_rejection_boring():
    assert dashboard._categorize_rejection("BORING: reads like a neutral Wikipedia summary") == "boring"


def test_categorize_rejection_unknown_falls_to_other():
    assert dashboard._categorize_rejection("some completely novel reason never seen before") == "other"


def test_categorize_rejection_empty_string():
    assert dashboard._categorize_rejection("") == "other"
    assert dashboard._categorize_rejection(None) == "other"


def test_rejection_category_counts_aggregates_correctly(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=1, rejected_reason="low specificity",
                            accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=2, rejected_reason="banned phrase: 'crucial'",
                            accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=3, rejected_reason="banned phrase: 'vital'",
                            accepted=False)
    counts = dashboard._rejection_category_counts()
    by_cat = {c["category"]: c["count"] for c in counts}
    assert by_cat["safety"] == 2
    assert by_cat["accuracy"] == 1
    total_pct = sum(c["pct"] for c in counts)
    assert 99.0 <= total_pct <= 101.0   # rounding tolerance


def test_rejection_category_counts_empty():
    assert dashboard._rejection_category_counts() == []


def test_failures_page_shows_bottleneck_breakdown(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="hook_gen", attempt_n=1, rejected_reason="forbidden opener: 'imagine'",
                            accepted=False)
    r = client.get("/failures")
    body = r.data.decode()
    assert "Bottleneck breakdown" in body
    assert "weak_hook" in body


def test_failures_page_shows_category_per_row(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="body_gen", attempt_n=1, body="x",
                            rejected_reason="low specificity", accepted=False)
    r = client.get("/failures")
    body = r.data.decode()
    assert "accuracy" in body


# ── Supervisor-gate categorization (phase-driven, not keyword-driven) ─────────

def test_categorize_rejection_seed_gate_is_weak_seed():
    assert dashboard._categorize_rejection(
        "no counter-intuitive angle, knowledge gap test fails", "seed_gate") == "weak_seed"


def test_categorize_rejection_fact_check_is_accuracy_regardless_of_wording():
    """fact_check verdicts are free-form LLM prose, not the controlled
    vocabulary keyword-matching relies on — phase alone must decide."""
    assert dashboard._categorize_rejection(
        "the claim about Nixon's motives isn't supported", "fact_check") == "accuracy"


def test_categorize_rejection_footage_gate_is_footage_drift():
    assert dashboard._categorize_rejection(
        "near-duplicate prompts with no visual variety", "footage_gate") == "footage_drift"


def test_categorize_rejection_body_gen_keyword_matching_still_works():
    """Phase-driven override must not swallow the existing keyword-based
    path for script_writer's own phases."""
    assert dashboard._categorize_rejection("low specificity (0.2/25w)", "body_gen") == "accuracy"
    assert dashboard._categorize_rejection("cadence: missing a short sentence", "body_gen") == "loose_structure"


def test_rejection_category_counts_includes_supervisor_gates(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="seed_gate", attempt_n=1,
                            rejected_reason="no surprise", accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="fact_check", attempt_n=1,
                            rejected_reason="invented figure", accepted=False)
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="footage_gate", attempt_n=1,
                            rejected_reason="off-topic imagery", accepted=False)
    counts = dashboard._rejection_category_counts()
    by_cat = {c["category"]: c["count"] for c in counts}
    assert by_cat["weak_seed"] == 1
    assert by_cat["accuracy"] == 1
    assert by_cat["footage_drift"] == 1


def test_failures_page_shows_supervisor_gate_categories(client):
    db_manager.save_attempt(run_id="r1", niche="finance", seed_type="wisdom",
                            phase="fact_check", attempt_n=1,
                            rejected_reason="invented figure", accepted=False)
    r = client.get("/failures")
    body = r.data.decode()
    assert "accuracy" in body
    assert "fact_check" in body


def test_debug_route_blocks_run_id_traversal(client, tmp_path):
    """Audit C2: run_id=".." resolved to media_library/ itself, serving any
    rendered (incl. rejected/unapproved) video via /debug/../output/x.mp4."""
    outside = dashboard.DEBUG_ROOT.parent / "output"
    outside.mkdir(parents=True)
    (outside / "secret_video.mp4").write_bytes(b"unpublished")
    dashboard.DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

    r = client.get("/debug/../output/secret_video.mp4")
    assert r.status_code in (404, 403)


def test_dashboard_runs_single_threaded():
    """Audit H3: _scoped_env's env mutation is only safe when requests are
    serialized — Flask 3.x defaults threaded=True, so the explicit
    threaded=False in app.run() is load-bearing. Guard it textually."""
    src = (Path(__file__).parent.parent / "scripts" / "dashboard.py").read_text()
    assert "threaded=False" in src


# ── Audit M4: upload-success-then-DB-failure must NOT read as failure ──────────

def test_approve_db_failure_after_successful_upload_warns_do_not_retry(client, tmp_path, monkeypatch):
    """If upload() succeeds but the DB status update then fails, the message
    must NOT say "upload failed" (that tempts a re-approve → duplicate public
    video). It must say uploaded-OK-do-not-retry."""
    vf = tmp_path / "v.mp4"; vf.write_bytes(b"x")
    vid = db_manager.save_video(niche="finance", script_hook="H", scene_desc="s",
                                video_file=str(vf), score=9)
    monkeypatch.setattr(youtube_uploader, "upload",
                        lambda *a, **k: ("https://youtu.be/LIVE", "LIVE"))
    monkeypatch.setattr(db_manager, "update_youtube_id",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    body = r.data.decode()
    assert "UPLOADED OK" in body
    assert "Do NOT re-approve" in body


def test_approve_upload_failure_records_mark_upload_failed(client, tmp_path, monkeypatch):
    """A genuine upload failure should be recorded (report.py's FAILED count
    used to miss dashboard failures entirely) and say it's safe to retry."""
    vf = tmp_path / "v.mp4"; vf.write_bytes(b"x")
    vid = db_manager.save_video(niche="finance", script_hook="H", scene_desc="s",
                                video_file=str(vf), score=9)
    recorded = []
    monkeypatch.setattr(youtube_uploader, "upload",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quota")))
    monkeypatch.setattr(db_manager, "mark_upload_failed",
                        lambda vid_, err: recorded.append((vid_, err)))
    r = client.post(f"/video/{vid}/approve", follow_redirects=True)
    assert "safe to retry" in r.data.decode()
    assert recorded and recorded[0][0] == vid
    with db_manager._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "pending"                    # still retryable
