"""Tests for youtube_uploader.py – peak time scheduling and metadata building."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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


def test_build_metadata_basic():
    script = "You're broke\nHere's why\nStop saving\nBuy assets\nFollow for more"
    cfg    = {"cta": "Follow for daily tactics.", "youtube_category_id": "25"}
    meta   = build_metadata(script, "finance", cfg)

    assert meta["title"]      == "You're broke"
    assert meta["categoryId"] == "25"
    assert "finance" in meta["tags"]
    assert "Follow for daily tactics." in meta["description"]
    assert "#finance" in meta["description"]


def test_build_metadata_falls_back_to_default_category():
    """When niche_cfg has no youtube_category_id, fall back to DEFAULT_CATEGORIES."""
    meta = build_metadata("Hook\nBody\nCTA", "motivation", {"cta": "x"})
    assert meta["categoryId"] == "22"  # People & Blogs


def test_build_metadata_truncates_long_first_line():
    long_hook = "x" * 200
    meta = build_metadata(f"{long_hook}\nsecond line", "finance", {"cta": "x"})
    assert len(meta["title"]) <= 80


def test_build_metadata_tags_have_no_hash():
    meta = build_metadata("Hook\n", "finance", {"cta": ""})
    # Tags should not include the leading #
    assert all(not t.startswith("#") for t in meta["tags"])
