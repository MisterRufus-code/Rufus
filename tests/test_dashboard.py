"""Tests for dashboard.py — the read-only status dashboard.

Uses Flask's test client (no network, no running server) against an
isolated temp DB so these run exactly like the rest of the suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import dashboard
import db_manager


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    monkeypatch.setattr(dashboard, "DEBUG_ROOT", tmp_path / "debug")
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def _seed(n_uploaded=2, n_held=1):
    for i in range(n_uploaded):
        db_manager.save_video(niche="finance", script_hook=f"Hook {i}",
                              scene_desc="s", video_file=f"v{i}.mp4",
                              score=9, youtube_id=f"yt{i}", run_id=f"run{i}")
    for i in range(n_held):
        db_manager.save_video(niche="finance", script_hook=f"HeldHook {i}",
                              scene_desc="s", video_file=f"h{i}.mp4",
                              score=5, hold_reason="score 5/10 < 8/10 threshold",
                              run_id=f"heldrun{i}")


# ── Index page ─────────────────────────────────────────────────────────────────

def test_index_empty_db_does_not_crash(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"No videos yet" in r.data


def test_index_shows_stats_and_rows(client):
    _seed(n_uploaded=2, n_held=1)
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Hook 0" in body
    assert "HeldHook 0" in body
    assert "held" in body
    assert "uploaded" in body


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


def test_video_detail_shows_held_status(client):
    vid = db_manager.save_video(
        niche="finance", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=5,
        hold_reason="factual integrity: invented figure",
    )
    r = client.get(f"/video/{vid}")
    body = r.data.decode()
    assert "HELD" in body
    assert "invented figure" in body


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
                                  "uploaded": 0, "held": 0}


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
