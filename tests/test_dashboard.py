"""Tests for dashboard.py — the approval-queue + status dashboard.

Uses Flask's test client (no network, no running server) against an
isolated temp DB so these run exactly like the rest of the suite. The real
upload path (youtube_uploader.upload) is monkeypatched everywhere — these
tests never touch the network or Google APIs.
"""

import sys
from pathlib import Path

import pytest
from filelock import FileLock

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


def test_video_detail_shows_image_prompts_inline(client, tmp_path, monkeypatch):
    """The per-beat image-generation prompts (script→images chain) must render
    inline on the review page, not just as downloadable files."""
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "debug")
    run_dir = tmp_path / "debug" / "run-xyz"
    run_dir.mkdir(parents=True)
    (run_dir / "01.txt").write_text("FLUX PROMPT:\nAn extreme close-up of a gold coin",
                                    encoding="utf-8")
    (run_dir / "01.png").write_bytes(b"fakepng")
    (run_dir / "02.txt").write_text("FLUX PROMPT:\nA wide shot of a trading floor",
                                    encoding="utf-8")

    vid = db_manager.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                video_file="v.mp4", score=9, run_id="run-xyz")
    body = client.get(f"/video/{vid}").data.decode()
    assert "close-up of a gold coin" in body       # prompt text rendered, label stripped
    assert "wide shot of a trading floor" in body
    assert "/debug/run-xyz/01.png" in body          # keyframe shown next to its prompt
    assert "FLUX PROMPT" not in body                # the label line is stripped, not shown


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
    def fake_upload(path, script, thumbnail_path=None, metadata=None, **kwargs):
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
    def fake_upload(path, script, thumbnail_path=None, metadata=None, **kwargs):
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


def test_rejection_category_counts_empty(client):
    """`client` for the DATABASE, not the HTTP client — it points DB_FILE at a
    tmp_path. Without it this asserted that the real rufus.db has never
    recorded a rejection, which is true exactly once: on a fresh checkout, on
    the first run of the suite. Any test elsewhere that writes an unisolated
    attempt row, or simply running pytest twice, turned it red for a reason
    that had nothing to do with the code under test."""
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


# ── /performance — real analytics, surfaced for the first time ──────────────

def test_performance_page_loads_empty(client):
    r = client.get("/performance")
    assert r.status_code == 200
    assert b"No uploaded videos" in r.data


def test_performance_page_shows_score_and_views(client):
    vid = db_manager.save_video(niche="finance", script_hook="Gold shock",
                                scene_desc="s", video_file="v.mp4",
                                score=9, youtube_id="yt0", upload_status="approved")
    db_manager.save_metrics(vid, views=12345, watch_pct=54.2, ctr=0.0, likes=99)
    r = client.get("/performance")
    body = r.data.decode()
    assert "Gold shock" in body
    assert "12345" in body
    assert "54%" in body
    assert "99" in body


def test_performance_page_handles_videos_with_no_metrics_yet(client):
    """LEFT JOIN must not drop a video that hasn't been analytics-fetched yet."""
    db_manager.save_video(niche="finance", script_hook="Brand new upload",
                          scene_desc="s", video_file="v.mp4",
                          score=8, youtube_id="yt1", upload_status="approved")
    r = client.get("/performance")
    body = r.data.decode()
    assert "Brand new upload" in body
    assert "—" in body   # blank views/watch%/likes rendered, not a crash


def test_performance_page_channel_filter(client):
    a = db_manager.save_video(niche="finance", script_hook="Chan A video",
                              scene_desc="s", video_file="a.mp4", score=8,
                              youtube_id="yt2", channel="chan_a", upload_status="approved")
    b = db_manager.save_video(niche="finance", script_hook="Chan B video",
                              scene_desc="s", video_file="b.mp4", score=8,
                              youtube_id="yt3", channel="chan_b", upload_status="approved")
    db_manager.save_metrics(a, views=100, watch_pct=50, ctr=0, likes=1)
    db_manager.save_metrics(b, views=200, watch_pct=50, ctr=0, likes=1)
    r = client.get("/performance?channel=chan_a")
    body = r.data.decode()
    assert "Chan A video" in body
    assert "Chan B video" not in body


def test_performance_correlation_needs_minimum_sample(client):
    vid = db_manager.save_video(niche="finance", script_hook="Only one",
                                scene_desc="s", video_file="v.mp4",
                                score=9, youtube_id="yt4", upload_status="approved")
    db_manager.save_metrics(vid, views=500, watch_pct=60, ctr=0, likes=5)
    r = client.get("/performance")
    assert b"Need" in r.data and b"to correlate" in r.data


def test_performance_correlation_shows_avg_views_once_enough_data(client):
    for i in range(5):
        vid = db_manager.save_video(niche="finance", script_hook=f"V{i}",
                                    scene_desc="s", video_file=f"v{i}.mp4",
                                    score=9, youtube_id=f"yt{i}", upload_status="approved")
        db_manager.save_metrics(vid, views=1000, watch_pct=50, ctr=0, likes=1)
    r = client.get("/performance")
    body = r.data.decode()
    assert "9/10" in body
    assert "avg views" in body


def test_score_vs_views_buckets_by_score():
    rows = [{"score": 9, "views": 100}, {"score": 9, "views": 300},
           {"score": 5, "views": 10}]
    out = dashboard._score_vs_views(rows)
    by_score = {b["score"]: b for b in out}
    assert by_score[9]["avg_views"] == 200
    assert by_score[9]["n"] == 2
    assert by_score[5]["avg_views"] == 10


# ── /system — process status + control ──────────────────────────────────────

def test_comfyui_reachable_true_on_200(monkeypatch):
    class R:
        status_code = 200
    monkeypatch.setattr(dashboard.requests, "get", lambda *a, **k: R())
    assert dashboard._comfyui_reachable() is True


def test_comfyui_reachable_false_on_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(dashboard.requests, "get", boom)
    assert dashboard._comfyui_reachable() is False


def test_run_in_progress_false_when_no_lock_held(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    assert dashboard._run_in_progress("main_en") is False


def test_run_in_progress_true_when_lock_held(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    lock_path = dashboard._lock_path("main_en")
    holder = FileLock(str(lock_path))
    holder.acquire(timeout=0)
    try:
        assert dashboard._run_in_progress("main_en") is True
    finally:
        holder.release()


def test_launch_run_builds_expected_command(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    dashboard._LAUNCHED.clear()
    captured = {}
    class FakeProc:
        def poll(self): return None
    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    proc, log_path = dashboard._launch_run(niche="finance", channel="main_en")
    assert captured["cmd"][0] == sys.executable
    assert str(tmp_path / "scripts" / "main.py") in captured["cmd"]
    assert "--niche" in captured["cmd"] and "finance" in captured["cmd"]
    assert "--channel" in captured["cmd"] and "main_en" in captured["cmd"]
    assert log_path.parent == tmp_path / "logs"
    assert dashboard._LAUNCHED["main_en"] is proc


def test_cancel_run_terminates_a_tracked_process():
    dashboard._LAUNCHED.clear()
    terminated = []
    class FakeProc:
        def poll(self): return None
        def terminate(self): terminated.append(True)
    dashboard._LAUNCHED["main_en"] = FakeProc()
    assert dashboard._cancel_run("main_en") is True
    assert terminated == [True]


def test_cancel_run_false_when_nothing_tracked():
    dashboard._LAUNCHED.clear()
    assert dashboard._cancel_run("nope") is False


def test_cancel_run_false_when_process_already_finished():
    dashboard._LAUNCHED.clear()
    class FakeProc:
        def poll(self): return 0   # already exited
    dashboard._LAUNCHED["main_en"] = FakeProc()
    assert dashboard._cancel_run("main_en") is False


def test_system_page_shows_comfy_and_channel_status(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_comfyui_reachable", lambda: True)
    monkeypatch.setattr(dashboard, "_channels", lambda: ["main_en"])
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    r = client.get("/system")
    body = r.data.decode()
    assert "reachable" in body
    assert "main_en" in body
    assert "idle" in body


def test_system_run_route_launches_and_redirects(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    dashboard._LAUNCHED.clear()
    calls = []
    def fake_launch(**kw):
        calls.append(kw)
        return None, tmp_path / "x.log"
    monkeypatch.setattr(dashboard, "_launch_run", fake_launch)
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: False)
    r = client.post("/system/run", data={"niche": "finance"}, follow_redirects=False)
    assert r.status_code in (301, 302)
    assert calls and calls[0]["niche"] == "finance"


def test_system_run_route_skips_when_already_running(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda cid: True)
    calls = []
    monkeypatch.setattr(dashboard, "_launch_run", lambda **kw: calls.append(kw))
    r = client.post("/system/run", data={"niche": "finance"}, follow_redirects=False)
    assert r.status_code in (301, 302)
    assert calls == []


def test_system_cancel_route_calls_cancel_run(client, monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard, "_cancel_run", lambda channel: calls.append(channel))
    r = client.post("/system/cancel", data={"channel": "main_en"}, follow_redirects=False)
    assert r.status_code in (301, 302)
    assert calls == ["main_en"]


def test_system_routes_block_non_localhost(client, monkeypatch):
    """Defense-in-depth: even if loopback binding is somehow bypassed, these
    routes must refuse a request that isn't from 127.0.0.1/::1. Flask's test
    client reports remote_addr as 127.0.0.1 by default, so force a non-local
    address via the environ override it supports.

    401, not the old 403: with no config/users.json these run in auth.py's
    legacy mode, where a non-loopback caller has no identity at all and is
    refused by the before_request hook before any route is entered. The
    refusal moved earlier and got broader — see the next test."""
    r = client.post("/system/run", data={"niche": "finance"},
                    environ_overrides={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 401
    r = client.post("/system/cancel", data={"channel": "main_en"},
                    environ_overrides={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 401


def test_non_localhost_cannot_read_anything_either(client):
    """The gap the old localhost-only guard left: it protected /system and
    /settings but nothing else, so anyone who could reach a dashboard bound to
    0.0.0.0 could read every script, score and rendered video. Now identity is
    required for all of it."""
    for path in ("/", "/performance", "/gallery", "/failures"):
        r = client.get(path, environ_overrides={"REMOTE_ADDR": "10.0.0.5"})
        assert r.status_code == 401, f"{path} was readable by a non-loopback client"


# ── /trending — browse rising queries, queue via the existing gate path ─────

def test_trending_page_lists_niche_links(client):
    r = client.get("/trending")
    assert r.status_code == 200
    body = r.data.decode()
    assert "finance" in body   # a real NICHE_TREND_SEEDS key


def test_trending_page_shows_queries_and_queue_buttons(client, monkeypatch):
    import research
    monkeypatch.setattr(research, "trending_queries_with_reason",
                        lambda niche: (["gold price surge"], ""))
    r = client.get("/trending?niche=finance")
    body = r.data.decode()
    assert "gold price surge" in body
    assert 'action="/request-topic"' in body
    assert 'value="gold price surge"' in body


def test_trending_page_handles_no_results(client, monkeypatch):
    import research
    monkeypatch.setattr(research, "trending_queries_with_reason",
                        lambda niche: ([], "Google Trends answered, and "
                                           "nothing is rising for these seeds "
                                           "this week. Nothing to fix."))
    r = client.get("/trending?niche=finance")
    assert b"No rising queries" in r.data


def test_the_empty_page_says_WHICH_of_the_four_it_was(client, monkeypatch):
    """"pytrends not installed, rate-limited, or nothing rising this week" was
    three guesses and a shrug on a page whose whole job is to tell you
    something: one of those needs a pip command, one clears by itself, one is
    not a problem at all — and they printed identically."""
    import research
    monkeypatch.setattr(research, "trending_queries_with_reason",
                        lambda niche: ([], "pytrends is not installed — "
                                           "`pip install pytrends`"))
    body = client.get("/trending?niche=finance").data.decode()
    assert "pip install pytrends" in body
    assert "rate-limited, or nothing rising" not in body, "the old shrug"


def test_trending_page_handles_lookup_failure(client, monkeypatch):
    import research
    def boom(niche):
        raise RuntimeError("pytrends rate-limited")
    monkeypatch.setattr(research, "trending_queries_with_reason", boom)
    r = client.get("/trending?niche=finance")
    assert b"Trend lookup failed" in r.data


def test_trending_queue_button_posts_to_request_topic(client, monkeypatch, tmp_path):
    """The whole point: queuing a trending query goes through the SAME gate
    path as manual topic entry, not a separate launch mechanism."""
    import subprocess
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        class P:
            def poll(self): return None
        return P()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    r = client.post("/request-topic", data={"topic": "gold price surge"},
                    follow_redirects=True)
    assert r.status_code == 200
    assert "--topic" in captured["cmd"] and "gold price surge" in captured["cmd"]


# ── /gallery — cross-run visual browsing ─────────────────────────────────────

def test_gallery_page_empty(client):
    r = client.get("/gallery")
    assert r.status_code == 200
    assert b"No keyframes saved yet" in r.data


def test_gallery_images_flattens_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path)
    for run_id in ("run_a", "run_b"):
        d = tmp_path / run_id
        d.mkdir()
        (d / "01.png").write_bytes(b"x")
        (d / "02.png").write_bytes(b"x")
    images = dashboard._gallery_images()
    assert len(images) == 4
    assert {i["run_id"] for i in images} == {"run_a", "run_b"}


def test_gallery_images_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path)
    d = tmp_path / "run_a"
    d.mkdir()
    for i in range(10):
        (d / f"{i:02d}.png").write_bytes(b"x")
    assert len(dashboard._gallery_images(limit=3)) == 3


def test_gallery_page_shows_thumbnails(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path)
    d = tmp_path / "run_a"
    d.mkdir()
    (d / "01.png").write_bytes(b"x")
    r = client.get("/gallery")
    body = r.data.decode()
    assert "/debug/run_a/01.png" in body


# ── /settings — tunables from a form, applied to dashboard-launched runs ────

def test_settings_page_loads_with_defaults(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    r = client.get("/settings")
    assert r.status_code == 200
    assert b"Stills only" in r.data
    assert b"Renderer" in r.data


def test_settings_save_writes_only_non_default_fields(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    client.post("/settings/save", data={"RUFUS_STILLS_ONLY": "1", "RUFUS_RENDERER": ""})
    saved = dashboard._load_settings()
    assert saved == {"RUFUS_STILLS_ONLY": "1"}   # blank ("default") never written


def test_settings_page_reflects_saved_values(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    dashboard._save_settings({"RUFUS_RENDERER": "remotion"})
    r = client.get("/settings")
    assert 'value="remotion" selected' in r.data.decode()


def test_settings_load_survives_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "nope.json")
    assert dashboard._load_settings() == {}


def test_settings_load_survives_corrupt_file(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    p.write_text("{ not json")
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", p)
    assert dashboard._load_settings() == {}


def test_settings_route_blocks_non_localhost(client):
    # 401 rather than 403 — see test_system_routes_block_non_localhost.
    r = client.post("/settings/save", data={},
                    environ_overrides={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 401


def test_settings_page_exposes_character_mode_toggle(client, tmp_path, monkeypatch):
    """RUFUS_CHARACTER_MODE — the global kill switch for character_engine.py
    — must be editable from the dashboard like every other feature toggle,
    not only by hand-editing niches.json."""
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    r = client.get("/settings")
    assert r.status_code == 200
    assert b"Recurring character" in r.data


def test_settings_save_persists_character_mode_off(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    client.post("/settings/save", data={"RUFUS_CHARACTER_MODE": "0"})
    assert dashboard._load_settings() == {"RUFUS_CHARACTER_MODE": "0"}


def test_launch_run_applies_saved_settings_as_env_overrides(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    dashboard._save_settings({"RUFUS_STILLS_ONLY": "1"})
    dashboard._LAUNCHED.clear()
    captured = {}
    class FakeProc:
        def poll(self): return None
    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return FakeProc()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    dashboard._launch_run(niche="finance")
    assert captured["env"]["RUFUS_STILLS_ONLY"] == "1"


# ── the workflow bench ───────────────────────────────────────────────────────

def test_bench_page_renders_with_nothing_benched_yet(client, monkeypatch):
    """The Style page answers "which look do I want". This one answers "which
    workflow draws it best", which is the question a gallery of pale beige
    frames raised — a style block that forbids a washed-out background twice
    and is obeyed by nobody is not a wording problem."""
    import workflow_bench as wb
    monkeypatch.setattr(wb, "latest", lambda: {})
    page = client.get("/bench").get_data(as_text=True)
    assert "Workflow bench" in page
    assert "config/workflows/" in page
    assert "measured against what ships today" in page


def test_bench_page_shows_the_grid_when_there_is_one(client, monkeypatch, tmp_path):
    """A GRID, not a list: comparing workflows means seeing the same probe
    across candidates at once."""
    import workflow_bench as wb
    d = tmp_path / "bench" / "20260817_000000"
    (d / "current").mkdir(parents=True)
    (d / "current" / "face.png").write_bytes(b"PNG")
    monkeypatch.setattr(wb, "latest", lambda: {
        "stamp": "20260817_000000", "dir": str(d), "width": 832, "height": 1472,
        "probes": ["face", "action"],
        "workflows": [{"label": "current", "mean_seconds": 12.5, "passed": 1,
                       "renders": {
                           "face": {"ok": True, "seconds": 12.5, "gate": "ok",
                                    "file": str(d / "current" / "face.png")},
                           "action": {"ok": True, "seconds": 12.5,
                                      "gate": "contact_sheet",
                                      "file": str(d / "current" / "face.png")}}}],
    })
    page = client.get("/bench").get_data(as_text=True)
    assert "face" in page and "action" in page
    assert "1/2 clean" in page
    assert "contact_sheet" in page, "a failed probe says why under the picture"


def test_the_bench_refuses_while_a_run_holds_the_gpu(client, monkeypatch):
    """Minutes of rendering behind a video run would look like a hung page."""
    monkeypatch.setattr(dashboard, "_run_in_progress", lambda c: True)
    r = client.post("/bench/run", follow_redirects=False)
    assert r.status_code in (301, 302)
    assert "using%20the%20GPU" in r.headers["Location"]


def test_the_bench_lists_what_is_wrong_with_an_export(client, monkeypatch):
    """A workflow sitting in the folder unvalidated is exactly the thing to
    know about before spending twenty-four renders on it."""
    import workflow_bench as wb
    monkeypatch.setattr(wb, "latest", lambda: {})
    monkeypatch.setattr(wb, "candidates",
                        lambda: [("broken", Path("/nope/broken.json"))])
    page = client.get("/bench").get_data(as_text=True)
    assert "unusable" in page
    assert "Export (API)" in page


def test_tracking_lists_what_is_waiting_to_go_live(client):
    """Uploaded private with a publish time, which YouTube acts on by itself.
    Until there was a column for it this was indistinguishable from a video
    that is private forever."""
    vid = db_manager.save_video(niche="money_history", script_hook="A hook",
                                scene_desc="s", video_file="v.mp4", score=9,
                                youtube_id="abcdefghijk", title="The panic")
    db_manager.set_publish_at(vid, "2099-01-01T12:00:00Z")
    page = client.get("/tracking").get_data(as_text=True)
    assert "Waiting to go live" in page
    assert "2099-01-01T12:00:00Z" in page
    assert "The panic" in page


def test_tracking_says_nothing_about_scheduling_when_nothing_is_scheduled(client):
    """An empty section that looks broken is worse than no section."""
    db_manager.save_video(niche="money_history", script_hook="A hook",
                          scene_desc="s", video_file="v.mp4", score=9)
    page = client.get("/tracking").get_data(as_text=True)
    assert "Waiting to go live" not in page


def test_the_privacy_setting_is_offered_and_explains_the_trade():
    """Scheduling is only possible on a private upload — YouTube's rule — so
    the two are one control, not two switches that can disagree."""
    kind = dashboard.SETTINGS_KINDS["RUFUS_PRIVACY"]
    assert kind == "select:public,private,unlisted"
    help_text = next(h for k, _l, _k, h in dashboard.SETTINGS_SCHEMA
                     if k == "RUFUS_PRIVACY")
    assert "next peak hour" in help_text
    assert "tzdata" in help_text, "the thing that silently breaks scheduling"


# ── the scout ────────────────────────────────────────────────────────────────

def test_scout_page_is_honest_about_an_empty_queue(client):
    """A quiet week is a real answer. An empty page that looks broken would
    make the owner go looking for a bug that is not there."""
    page = client.get("/scout").get_data(as_text=True)
    assert "Scout" in page
    assert "a quiet week is a real answer" in page


def test_a_proposal_shows_the_evidence_that_chose_it(client):
    """THE DIFFERENCE BETWEEN REVIEWING AN AGENT AND RUBBER-STAMPING IT. "Make
    a video about the Panic of 1893" is an instruction. "Neighbour published
    this, it did 9x their own median, and we have not covered it" is something
    a person can disagree with."""
    db_manager.save_proposal(
        channel="main_en", niche="money_history", topic="The Panic of 1893",
        evidence="Neighbour published X · 90,000 views — 9.0x that channel's "
                 "own median")
    page = client.get("/scout").get_data(as_text=True)
    assert "The Panic of 1893" in page
    assert "9.0x that channel" in page


def test_the_scout_card_no_longer_carries_a_script(client):
    """It used to show "the script it wrote" inside a <details>. That prose was
    bought for every proposal and thrown away even on approval, because
    approving launched an ordinary run with topic= alone — which writes its
    own. Six pending proposals meant six scripts paid for and six discarded."""
    db_manager.save_proposal(channel="c", niche="n", topic="Tulips",
                             evidence="e")
    page = client.get("/scout").get_data(as_text=True)
    assert "the script it wrote" not in page
    assert "Write 3 scripts" in page


def test_approving_a_topic_buys_scripts_and_not_a_render(client, monkeypatch):
    """THE EXPENSIVE STEP MOVED ONE PAGE FURTHER ON, and that is the point of
    the split. A render is hours of the 3090 and it used to be committed to
    from a topic card — with the only script anyone had seen already discarded.
    Approving a topic now buys three scripts, and the render waits behind a
    choice between them."""
    started, rendered = {}, []
    monkeypatch.setattr(dashboard, "_launch_candidates",
                        lambda **kw: started.update(kw) or Path("logs/c.log"))
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: rendered.append(kw) or (None, Path("r.log")))
    pid = db_manager.save_proposal(
        channel="main_en", niche="money_history", topic="The Panic of 1893",
        evidence="e")
    r = client.post(f"/scout/{pid}/approve", follow_redirects=False)
    assert r.status_code in (301, 302)
    assert started == {"topic": "The Panic of 1893", "proposal_id": pid,
                       "channel": "main_en"}
    assert rendered == [], "approving a topic must not start a render"
    assert db_manager.proposals(status="approved")[0]["id"] == pid


def test_a_proposal_cannot_be_approved_twice(client, monkeypatch):
    """Two clicks on a phone with a slow connection must not write two sets."""
    calls = []
    monkeypatch.setattr(dashboard, "_launch_candidates",
                        lambda **kw: calls.append(kw) or Path("l.log"))
    pid = db_manager.save_proposal(channel="c", niche="n", topic="t",
                                   evidence="e")
    client.post(f"/scout/{pid}/approve")
    client.post(f"/scout/{pid}/approve")
    assert len(calls) == 1


# ── choosing between three finished scripts ─────────────────────────────────

def _candidates(topic="The Panic of 1893", proposal_id=1):
    return [db_manager.save_candidate(
        proposal_id=proposal_id, channel="main_en", niche="money_history",
        topic=topic, hook_style=style, hook=f"Hook {i}",
        script=f"Hook {i}\nBody of script {i}.", score=7 + i, cost_usd=0.02)
        for i, style in enumerate(["counterintuitive", "shocking_stat",
                                   "warning"])]


def test_the_page_shows_every_candidate_and_its_style(client):
    """A flat list of three hooks is not a choice — what makes it one is seeing
    the whole script and which opening shape it is."""
    _candidates()
    page = client.get("/scripts").get_data(as_text=True)
    for style in ("counterintuitive", "shocking_stat", "warning"):
        assert style in page
    assert "Body of script 1." in page


def test_choosing_one_draws_galleries_for_that_script(client, monkeypatch):
    """--script, not --topic. main.py skips its writer for a run given a script
    file, so the script a person read is the script the video is built from;
    passing the topic would have the writer produce a fourth one nobody chose.

    And the render still does not start here — the same principle one stage on:
    the irreversible expensive step waits behind the LAST human judgement."""
    started, rendered = {}, []
    monkeypatch.setattr(dashboard, "_launch_galleries",
                        lambda **kw: started.update(kw) or Path("g.log"))
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: rendered.append(kw) or (None, Path("r.log")))
    ids = _candidates()
    client.post(f"/scripts/{ids[1]}/choose", follow_redirects=False)
    assert "script_file" in started and "topic" in started
    assert rendered == [], "choosing a script must not start a render"
    assert Path(started["script_file"]).read_text(
        encoding="utf-8").startswith("Hook 1")


def test_choosing_one_records_the_others_as_passed_over(client, monkeypatch):
    """The pair is the product. Nothing published here has view counts, so a
    click is the only labelled preference this channel can collect."""
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: (None, Path("r.log")))
    ids = _candidates()
    client.post(f"/scripts/{ids[0]}/choose")
    rows = {c["id"]: c["status"] for c in db_manager.candidates(proposal_id=1)}
    assert rows[ids[0]] == "chosen"
    assert rows[ids[1]] == rows[ids[2]] == "rejected"


def test_a_script_cannot_be_chosen_twice(client, monkeypatch):
    """A slow page invites a double click, and the second one would draw
    another forty minutes of galleries for the same script."""
    calls = []
    monkeypatch.setattr(dashboard, "_launch_galleries",
                        lambda **kw: calls.append(kw) or Path("g.log"))
    ids = _candidates()
    client.post(f"/scripts/{ids[0]}/choose")
    client.post(f"/scripts/{ids[0]}/choose")
    assert len(calls) == 1


# ── choosing the pictures ───────────────────────────────────────────────────

def _gallery(beats=3, variants=2, topic="Rome"):
    set_id = db_manager.save_gallery_set(
        candidate_id=1, channel="main_en", niche="money_history", topic=topic,
        script_file="logs/chosen_scripts/candidate_1.txt", n_variants=variants)
    for v in range(variants):
        for b in range(beats):
            db_manager.save_gallery_image(
                set_id=set_id, variant=v, beat_index=b,
                path=f"/nope/v{v}_{b}.png", prompt=f"shot {b}", seed=100 + b)
    return set_id


def test_taking_a_base_picks_every_shot_from_one_draw(client):
    """One click, then corrections. Sixteen separate judgements is not a
    workflow anybody finishes."""
    sid = _gallery()
    client.post(f"/galleries/{sid}/base/1")
    rows = db_manager.gallery_images(sid, status="chosen")
    assert len(rows) == 3
    assert {r["variant"] for r in rows} == {1}


def test_swapping_one_shot_leaves_the_rest_alone(client):
    """The good half of the other draw is exactly what a whole-bundle choice
    throws away."""
    sid = _gallery()
    client.post(f"/galleries/{sid}/base/0")
    client.post(f"/galleries/{sid}/swap/1/1")
    chosen = {r["beat_index"]: r["variant"]
              for r in db_manager.gallery_images(sid, status="chosen")}
    assert chosen == {0: 0, 1: 1, 2: 0}


def test_a_set_with_an_unpicked_shot_will_not_render(client, monkeypatch):
    """clip[i] belongs to beat[i] all the way downstream. A short list does not
    lose one picture — it slides every later one onto the wrong sentence."""
    rendered = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: rendered.append(kw) or (None, Path("r.log")))
    sid = _gallery()
    r = client.post(f"/galleries/{sid}/use", follow_redirects=False)
    assert r.status_code in (301, 302)
    assert rendered == []


def test_a_complete_set_renders_from_the_chosen_pictures(client, monkeypatch):
    """Nothing is redrawn — that is the whole point of having spent forty
    minutes drawing and choosing."""
    launched = {}
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: launched.update(kw) or (None, Path("r.log")))
    sid = _gallery()
    client.post(f"/galleries/{sid}/base/0")
    client.post(f"/galleries/{sid}/use")
    assert launched["gallery_id"] == sid
    assert launched["script_file"].endswith("candidate_1.txt")


def test_a_gallery_cannot_be_sent_to_render_twice(client, monkeypatch):
    calls = []
    monkeypatch.setattr(dashboard, "_launch_run",
                        lambda **kw: calls.append(kw) or (None, Path("r.log")))
    sid = _gallery()
    client.post(f"/galleries/{sid}/base/0")
    client.post(f"/galleries/{sid}/use")
    client.post(f"/galleries/{sid}/use")
    assert len(calls) == 1


def test_an_empty_gallery_page_says_where_pictures_come_from(client):
    page = client.get("/galleries").get_data(as_text=True)
    assert "Nothing waiting" in page and "/scripts" in page


def test_an_empty_choose_page_says_where_scripts_come_from(client):
    """A page that is empty because the work has not finished yet looks exactly
    like a page that is empty because it is broken."""
    page = client.get("/scripts").get_data(as_text=True)
    assert "Nothing waiting" in page and "/scout" in page


def test_rejecting_keeps_the_row(client):
    """Rejected proposals are what stop the scout proposing the same idea
    again next pass — deleting them would make it forget."""
    pid = db_manager.save_proposal(channel="c", niche="n", topic="Tulips",
                                   evidence="e")
    client.post(f"/scout/{pid}/reject")
    assert db_manager.proposals(status="rejected")[0]["topic"] == "Tulips"
    assert db_manager.proposals(status="pending") == []


def test_the_page_shows_what_it_is_watching(client):
    db_manager.record_observations([{
        "video_id": "v1", "channel_id": "c", "channel_title": "Neighbour",
        "title": "How Money Broke", "published_at": "", "views": 90_000,
        "channel_median": 10_000, "outperformance": 9.0}])
    page = client.get("/scout").get_data(as_text=True)
    assert "Neighbour" in page and "How Money Broke" in page
    assert "9.0x" in page


# ── the review page is in the order the review happens ──────────────────────
#
# Reviewing is: hear it, look at the beats, decide. The page was in the order
# it was BUILT in — id line, status, score, a five-row criteria table, then the
# buttons, the publish form and the title editor, and only then the voiceover
# and the contact sheet. On the phone this review is actually done from, that
# is several screens of numbers before the first thing being judged.


@pytest.fixture
def reviewable(client, tmp_path):
    """One pending video with a run folder that has a voiceover in it, so the
    preview block actually renders and its position can be asserted."""
    run = dashboard.DEBUG_ROOT / "run-review"
    run.mkdir(parents=True)
    (run / "voiceover.mp3").write_bytes(b"\xff\xfb" + b"\0" * 400)
    vid = db_manager.save_video(niche="money_history", script_hook="Hook",
                                scene_desc="s", video_file="v.mp4", score=9,
                                run_id="run-review", title="A title",
                                description="A description")
    return vid


def test_you_hear_it_before_you_are_asked_to_decide(client, reviewable):
    """The page used to open on the id line, the status, the score and a
    five-row criteria table — four screens of numbers on a phone before the
    voiceover, which is the thing the judgement is actually about."""
    page = client.get(f"/video/{reviewable}").get_data(as_text=True)
    assert page.index("voiceover.mp3") < page.index(f"/video/{reviewable}/approve")


def test_the_decision_comes_before_the_reading(client, reviewable):
    """Script, score breakdown, critic reasoning, seed and prompts all explain
    or amend the decision — they are read after it, not scrolled past to get
    to it."""
    page = client.get(f"/video/{reviewable}").get_data(as_text=True)
    approve = page.index(f"/video/{reviewable}/approve")
    for later in ("<h2>Script</h2>", "Why this score", "<h2>Seed / source</h2>",
                  "Image prompts", "Debug artifacts"):
        assert approve < page.index(later), f"{later} sits above Approve"


def test_approving_names_the_title_it_is_about_to_publish(client, reviewable):
    """The title editor moved below this button. A confirm that says "this
    video" would let a title nobody looked at go out on a tap."""
    page = client.get(f"/video/{reviewable}").get_data(as_text=True)
    assert "A title" in page.split("confirm(")[1].split(")")[0]


def test_a_title_with_an_apostrophe_does_not_break_the_button(client):
    """Hand-escaping the JS string and the attribute value differently is how
    one quote turns Approve into a button that does nothing."""
    vid = db_manager.save_video(niche="n", script_hook="h", scene_desc="s",
                                video_file="v.mp4", score=9,
                                title="Bretton Woods' end \"1971\"")
    page = client.get(f"/video/{vid}").get_data(as_text=True)
    attr = page.split('onsubmit="')[1].split('"')[0]
    assert attr.startswith("return confirm(")
    # Un-escape the attribute the way a browser would, then check what is left
    # is a single valid JS/JSON string literal rather than a broken one.
    import html as _html
    import json
    inner = _html.unescape(attr)[len("return confirm("):-len(");")]
    assert json.loads(inner).endswith("to YouTube now?")


def test_a_fact_gate_failure_is_visible_on_the_card(client):
    """The one warning that survives a person reading the script. They can
    judge the writing; they cannot check the figure against the source."""
    db_manager.save_candidate(
        proposal_id=7, channel="main_en", niche="money_history", topic="Rome",
        hook_style="shocking_stat", hook="Ninety per cent gone.",
        script="Ninety per cent gone.\nBody.", score=9, cost_usd=0.02,
        fact_ok=False, fact_reason="the source gives no silver percentage")
    page = client.get("/scripts").get_data(as_text=True)
    assert "the source does not support this" in page
    assert "no silver percentage" in page


def test_a_low_score_is_shown_and_not_hidden(client):
    """Nothing is withheld for missing a bar — that is the reviewer's call, and
    a threshold binning good scripts is the complaint this flow answers."""
    db_manager.save_candidate(
        proposal_id=8, channel="main_en", niche="money_history", topic="Tulips",
        hook_style="warning", hook="A weak one.", script="A weak one.\nBody.",
        score=4, cost_usd=0.02)
    page = client.get("/scripts").get_data(as_text=True)
    assert "4/10" in page
    assert "A weak one." in page
    assert "Make this one" in page
