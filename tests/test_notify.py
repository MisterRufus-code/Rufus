"""Tests for notify.py — the phone ping that makes the approval queue usable.

Never touches the network: every backend's HTTP call is patched. The contract
under test is fail-open — a notification problem must never break a render.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import notify


def _clear(monkeypatch):
    for v in ("RUFUS_NTFY_TOPIC", "RUFUS_PUSHOVER_TOKEN", "RUFUS_PUSHOVER_USER",
              "RUFUS_TELEGRAM_TOKEN", "RUFUS_TELEGRAM_CHAT", "RUFUS_DASHBOARD_URL",
              "RUFUS_NOTIFY", "RUFUS_NTFY_SERVER"):
        monkeypatch.delenv(v, raising=False)


def _ok():
    r = MagicMock(); r.status_code = 200; return r


def test_no_backend_configured_is_a_noop(monkeypatch):
    _clear(monkeypatch)
    with patch.object(notify.requests, "post") as post:
        assert notify.send("t", "b") is False
    post.assert_not_called()          # nothing attempted, nothing raised


def test_disabled_via_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "secret-topic")
    monkeypatch.setenv("RUFUS_NOTIFY", "0")
    with patch.object(notify.requests, "post") as post:
        assert notify.send("t", "b") is False
    post.assert_not_called()


def test_ntfy_posts_to_the_topic(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "rufus-a7f3k9x2")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        assert notify.send("Title", "Body") is True
    assert post.call_args[0][0] == "https://ntfy.sh/rufus-a7f3k9x2"


def test_ntfy_headers_survive_a_non_ascii_title(monkeypatch):
    """Real trap: HTTP headers are latin-1; an em-dash or emoji in a video
    title would raise on encode and silently lose the notification."""
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        assert notify.send("Gold — the 1971 shock 🪙", "body") is True
    post.call_args[1]["headers"]["Title"].encode("latin-1")   # must not raise


def test_failure_is_swallowed(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    with patch.object(notify.requests, "post", side_effect=OSError("no network")):
        assert notify.send("t", "b") is False     # returns, never raises


def test_one_backend_failing_does_not_block_the_other(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    monkeypatch.setenv("RUFUS_PUSHOVER_TOKEN", "tok")
    monkeypatch.setenv("RUFUS_PUSHOVER_USER", "usr")

    def flaky(url, *a, **k):
        if "ntfy" in url:
            raise OSError("down")
        return _ok()

    with patch.object(notify.requests, "post", side_effect=flaky):
        assert notify.send("t", "b") is True      # pushover still delivered


def test_pending_review_deep_links_to_the_video(monkeypatch):
    """Two taps from the phone, not hunting for the row in a list."""
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "http://192.168.1.20:8765/")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_pending_review(title="Gold shock", score=9,
                                     niche="money_history", video_id=21)
    assert post.call_args[1]["headers"]["Click"] == "http://192.168.1.20:8765/video/21"


def test_pending_review_body_carries_score_and_hold_reason(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_pending_review(title="T", score=5, niche="money_history",
                                     video_id=3, hold_reason="score 5/10 < 8/10")
    body = post.call_args[1]["data"].decode("utf-8")
    assert "money_history" in body and "5/10" in body
