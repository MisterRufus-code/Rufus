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


def test_run_failed_sends_high_priority(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        assert notify.notify_run_failed("boom", niche="money_history",
                                        channel="main_en") is True
    assert post.call_args[1]["headers"]["Priority"] == "high"
    body = post.call_args[1]["data"].decode("utf-8")
    assert "money_history" in body and "main_en" in body and "boom" in body


def test_run_failed_links_to_failures_page(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    monkeypatch.setenv("RUFUS_DASHBOARD_URL", "http://192.168.1.20:8765")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_run_failed("boom")
    assert post.call_args[1]["headers"]["Click"] == "http://192.168.1.20:8765/failures"


def test_run_failed_truncates_a_long_reason(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "topic")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_run_failed("x" * 10_000)
    body = post.call_args[1]["data"].decode("utf-8")
    assert len(body) <= 500


def test_run_failed_no_backend_is_a_noop(monkeypatch):
    _clear(monkeypatch)
    assert notify.notify_run_failed("boom") is False


# ── the render that ended, whichever way it ended ────────────────────────────

def test_a_held_video_is_announced_like_an_approved_one(monkeypatch):
    """SIX ENDINGS, ONE OF THEM AUDIBLE. Only the review branch ever notified,
    so a video the QC held finished in complete silence and sat on disk until
    somebody thought to look — which is not something you find out about, it is
    something you eventually notice."""
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "rufus")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        assert notify.notify_finished(title="Croesus", outcome="qc",
                                      detail="silent for 4s") is True
    body = post.call_args.kwargs["data"].decode("utf-8")
    assert "failed QC" in body
    assert "silent for 4s" in body
    assert "was held" in post.call_args.kwargs["headers"]["Title"]


def test_an_upload_is_announced_with_the_youtube_link(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "rufus")
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_finished(title="Croesus", outcome="uploaded",
                               youtube_url="https://youtu.be/abc")
    assert post.call_args.kwargs["headers"]["Click"] == "https://youtu.be/abc"


def test_the_video_itself_goes_to_discord(monkeypatch, tmp_path):
    """The phone backends can carry a link and not a payload; Discord is the
    only one that can hand you the thing you are being asked to judge."""
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.test/hook")
    mp4 = tmp_path / "out.mp4"
    mp4.write_bytes(b"x" * 1024)
    with patch.object(notify.requests, "post", return_value=_ok()) as post:
        notify.notify_finished(title="Croesus", outcome="review",
                               video_path=mp4)
    assert any("files" in c.kwargs for c in post.call_args_list), (
        "the finished video was never attached")


def test_a_dead_webhook_does_not_unfinish_a_finished_render(monkeypatch,
                                                            tmp_path):
    """Fail-open, same contract as every other sender here."""
    _clear(monkeypatch)
    monkeypatch.setenv("RUFUS_NTFY_TOPIC", "rufus")
    monkeypatch.setenv("RUFUS_DISCORD_WEBHOOK", "https://discord.test/hook")
    mp4 = tmp_path / "out.mp4"
    mp4.write_bytes(b"x" * 1024)
    with patch.object(notify.requests, "post", side_effect=OSError("no route")):
        assert notify.notify_finished(title="C", outcome="review",
                                      video_path=mp4) is False


def test_every_ending_the_pipeline_can_reach_has_something_to_say():
    """A run that ends in a state this map has no entry for would announce the
    raw internal word. main.py sets exactly these."""
    import re
    from pathlib import Path
    src = (Path(notify.__file__).parent / "main.py").read_text(encoding="utf-8")
    used = set(re.findall(r'ending(?:, ending_detail)? = "([a-z_]+)"', src))
    assert used, "main.py stopped recording which ending it reached"
    assert used <= set(notify._OUTCOMES), used - set(notify._OUTCOMES)
