"""Tests for youtube_uploader.py – peak time scheduling and metadata building."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import metadata_writer
import youtube_uploader
from youtube_uploader import _next_peak_utc, build_metadata, PEAK_HOURS_ET


def test_next_peak_is_future():
    """The returned timestamp must always be strictly after `now`."""
    ts  = _next_peak_utc()
    dt  = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    assert dt > now


def test_next_peak_at_least_five_minutes_out():
    """Must be ≥5 min from now (YouTube requires future publishAt)."""
    ts  = _next_peak_utc()
    dt  = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    assert (dt - now) >= timedelta(minutes=5)


def test_next_peak_hour_is_in_peak_list():
    """The hour returned must be one of the configured ET peaks."""
    ts  = _next_peak_utc()
    dt  = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    et  = dt.astimezone(ZoneInfo("America/New_York"))
    assert et.hour in PEAK_HOURS_ET


def test_build_metadata_basic(monkeypatch):
    # Force the legacy (no-key) path so this test is hermetic regardless of
    # whether a real OpenAI key is configured in config/keys.json — otherwise
    # it silently makes a live GPT call and asserts on non-deterministic output.
    monkeypatch.setattr(metadata_writer, "_load_key", lambda: "")

    script = "You're broke\nHere's why\nStop saving\nBuy assets\nFollow for more"
    cfg    = {"cta": "Follow for daily tactics.", "youtube_category_id": "25"}
    meta   = build_metadata(script, "finance", cfg)

    assert meta["title"]      == "You're broke"
    assert meta["categoryId"] == "25"
    assert "finance" in meta["tags"]
    assert "Follow for daily tactics." in meta["description"]
    assert "#finance" in meta["description"]


def test_build_metadata_falls_back_to_default_category(monkeypatch):
    """When niche_cfg has no youtube_category_id, fall back to DEFAULT_CATEGORIES."""
    monkeypatch.setattr(metadata_writer, "_load_key", lambda: "")
    meta = build_metadata("Hook\nBody\nCTA", "motivation", {"cta": "x"})
    assert meta["categoryId"] == "22"  # People & Blogs


def test_build_metadata_truncates_long_first_line(monkeypatch):
    monkeypatch.setattr(metadata_writer, "_load_key", lambda: "")
    long_hook = "x" * 200
    meta = build_metadata(f"{long_hook}\nsecond line", "finance", {"cta": "x"})
    assert len(meta["title"]) <= 80


def test_build_metadata_tags_have_no_hash(monkeypatch):
    monkeypatch.setattr(metadata_writer, "_load_key", lambda: "")
    meta = build_metadata("Hook\n", "finance", {"cta": ""})
    # Tags should not include the leading #
    assert all(not t.startswith("#") for t in meta["tags"])


# ── Synthetic-media disclosure ────────────────────────────────────────────────
# Every Rufus video is 100% AI-generated (GPT script, FLUX/SVD imagery/motion,
# synthesized voice) — YouTube's altered/synthetic-content policy requires
# self-declaring status.containsSyntheticMedia=True on upload. Regression test
# for that disclosure actually reaching the API request body.

@dataclass
class _FakeChannel:
    id: str = "main_en"
    upload: dict = field(default_factory=lambda: {"privacy": "private", "peak_hours": [18], "timezone": "UTC"})
    niche_overrides: dict = field(default_factory=dict)


def test_upload_declares_synthetic_media(tmp_path, monkeypatch):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake mp4 bytes")

    captured = {}
    fake_youtube = MagicMock()

    def fake_insert(part, body, media_body):
        captured["body"] = body
        req = MagicMock()
        req.next_chunk.return_value = (None, {"id": "vid123"})
        return req

    fake_youtube.videos.return_value.insert.side_effect = fake_insert

    monkeypatch.setattr(youtube_uploader, "_channel", lambda: _FakeChannel())
    monkeypatch.setattr(youtube_uploader, "get_authenticated_service", lambda channel=None: fake_youtube)
    monkeypatch.setattr(youtube_uploader, "load_niche", lambda: ({"cta": "x"}, "money_history"))
    monkeypatch.setattr(youtube_uploader, "post_cta_comment", lambda *a, **k: None)

    meta = {"title": "t", "description": "d", "tags": ["a"], "categoryId": "22"}
    with patch("googleapiclient.http.MediaFileUpload", MagicMock()):
        youtube_uploader.upload(video, "script text", metadata=meta)

    assert captured["body"]["status"]["containsSyntheticMedia"] is True


# ── Source-citation comment ──────────────────────────────────────────────────
# Trust/differentiation lever: cite the real source (Wikipedia / Stack
# Exchange URL) a script was grounded in, same posting mechanism as the
# existing CTA comment. Pinning isn't possible via the public API (see the
# function's own docstring) — this only POSTS the comment.

def test_post_source_comment_noop_without_url():
    fake_youtube = MagicMock()
    youtube_uploader.post_source_comment(fake_youtube, "vid123", None)
    fake_youtube.commentThreads.assert_not_called()


def test_post_source_comment_noop_with_empty_string():
    fake_youtube = MagicMock()
    youtube_uploader.post_source_comment(fake_youtube, "vid123", "")
    fake_youtube.commentThreads.assert_not_called()


def test_post_source_comment_posts_url_and_label():
    captured = {}
    fake_youtube = MagicMock()

    def fake_insert(part, body):
        captured["body"] = body
        return MagicMock()

    fake_youtube.commentThreads.return_value.insert.side_effect = fake_insert

    youtube_uploader.post_source_comment(
        fake_youtube, "vid123",
        "https://en.wikipedia.org/wiki/Bretton_Woods_Conference",
        seed_source="Wikipedia",
    )

    text = captured["body"]["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
    assert "https://en.wikipedia.org/wiki/Bretton_Woods_Conference" in text
    assert "Wikipedia" in text
    assert captured["body"]["snippet"]["videoId"] == "vid123"


def test_post_source_comment_works_without_a_label():
    fake_youtube = MagicMock()
    fake_youtube.commentThreads.return_value.insert.return_value = MagicMock()
    # Must not raise just because seed_source wasn't passed.
    youtube_uploader.post_source_comment(fake_youtube, "vid123", "https://example.com/x")


def test_post_source_comment_never_raises_on_api_error():
    fake_youtube = MagicMock()
    fake_youtube.commentThreads.return_value.insert.side_effect = RuntimeError("quota exceeded")
    youtube_uploader.post_source_comment(fake_youtube, "vid123", "https://example.com/x")


def test_upload_posts_source_comment_when_url_provided(tmp_path, monkeypatch):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"fake mp4 bytes")

    fake_youtube = MagicMock()

    def fake_insert(part, body, media_body):
        req = MagicMock()
        req.next_chunk.return_value = (None, {"id": "vid123"})
        return req

    fake_youtube.videos.return_value.insert.side_effect = fake_insert

    monkeypatch.setattr(youtube_uploader, "_channel", lambda: _FakeChannel())
    monkeypatch.setattr(youtube_uploader, "get_authenticated_service", lambda channel=None: fake_youtube)
    monkeypatch.setattr(youtube_uploader, "load_niche", lambda: ({"cta": "x"}, "money_history"))
    monkeypatch.setattr(youtube_uploader, "post_cta_comment", lambda *a, **k: None)

    captured = {}
    monkeypatch.setattr(
        youtube_uploader, "post_source_comment",
        lambda youtube, video_id, source_url, seed_source=None: captured.update(
            video_id=video_id, source_url=source_url, seed_source=seed_source))

    meta = {"title": "t", "description": "d", "tags": ["a"], "categoryId": "22"}
    with patch("googleapiclient.http.MediaFileUpload", MagicMock()):
        youtube_uploader.upload(video, "script text", metadata=meta,
                                source_url="https://en.wikipedia.org/wiki/X",
                                seed_source="Wikipedia")

    assert captured == {"video_id": "vid123",
                        "source_url": "https://en.wikipedia.org/wiki/X",
                        "seed_source": "Wikipedia"}


# ── the timezone database Windows does not have ──────────────────────────────

def test_a_missing_timezone_database_does_not_lose_the_upload(monkeypatch, capsys):
    """Linux and macOS ship an IANA database and zoneinfo reads it; Windows
    ships none, so there `tzdata` IS the database — and it was in no
    requirements file. Live, on a finished render:

        Upload failed (not uploaded, safe to retry):
        'No time zone found with key America/New_York'

    A scheduling nicety refusing to publish work that was ready. The schedule
    is worth having; it is not worth the upload."""
    import zoneinfo

    def _boom(name):
        raise zoneinfo.ZoneInfoNotFoundError(f"'No time zone found with key {name}'")

    monkeypatch.setattr(zoneinfo, "ZoneInfo", _boom)
    assert youtube_uploader._next_peak_utc(tz_name="America/New_York") == ""
    out = capsys.readouterr().out
    assert "pip install tzdata" in out
    assert "WITHOUT a scheduled publish time" in out


def test_the_schedule_still_works_when_the_database_is_there():
    when = youtube_uploader._next_peak_utc(peak_hours=[8, 12, 17, 20],
                                           tz_name="America/New_York")
    assert when.endswith("Z") and len(when) == 20, when


def test_tzdata_is_declared():
    """It is not an optional extra on Windows — it is the only copy of the
    data the standard library needs."""
    from pathlib import Path as _P
    req = (_P(youtube_uploader.__file__).parent.parent / "requirements.txt")
    assert "tzdata" in req.read_text(encoding="utf-8")
