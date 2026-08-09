"""Small review artifacts: the mp4 proxy and the contact sheet.

Why these exist: reviewing a run away from the desk had two options and both
moved ~20MB. The dashboard serves the master with as_attachment=True (a
download, not even a preview), and the debug stills are 1.1-2.7MB EACH, so
opening them one at a time costs the same again. Meanwhile notify.send_file
could post into Discord but the attach limit is 8MB, so a 15-25MB Short never
fit: it always fell through to "File too large to attach" and posted a link —
the same dead end it was meant to replace.

The fix is to move a SMALL artifact instead of the master. These tests pin the
properties that make that safe: the master is never modified, the fallbacks are
silent, and nothing here can raise into the pipeline.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import review_proxy  # noqa: E402


def _png(path: Path, size=(1080, 1920), shade=90):
    from PIL import Image
    Image.new("RGB", size, (shade, shade // 2, 140)).save(path)


# ── Contact sheet ────────────────────────────────────────────────────────────

def test_contact_sheet_covers_every_still(tmp_path):
    for i in range(1, 11):
        _png(tmp_path / f"{i:02d}.png", shade=20 * i)
    sheet = review_proxy.contact_sheet(tmp_path)
    assert sheet is not None and sheet.exists()
    from PIL import Image
    with Image.open(sheet) as im:
        cols = review_proxy.SHEET_COLUMNS
        assert im.width == cols * review_proxy.SHEET_CELL_WIDTH
        assert im.height == (10 // cols) * (im.width // cols) * 1920 // 1080


def test_contact_sheet_is_far_smaller_than_the_stills(tmp_path):
    """The entire point. Ten real stills are 15-25MB; the sheet must be a
    fraction of one of them."""
    for i in range(1, 11):
        _png(tmp_path / f"{i:02d}.png", shade=20 * i)
    raw = sum(p.stat().st_size for p in tmp_path.glob("*.png"))
    sheet = review_proxy.contact_sheet(tmp_path)
    assert sheet.stat().st_size < raw / 4


def test_contact_sheet_keeps_beat_order(tmp_path):
    """Beat order IS the review — a fault gets reported as "beat 7". Lexical
    order of 01.png..10.png is narration order; mtime order is not, because
    the renders complete out of sequence."""
    import time
    from PIL import Image
    shades = {1: 250, 2: 10, 3: 130}
    for i in (3, 1, 2):                       # written out of order on purpose
        _png(tmp_path / f"{i:02d}.png", size=(100, 100), shade=shades[i])
        time.sleep(0.01)
    sheet = review_proxy.contact_sheet(tmp_path)
    with Image.open(sheet) as im:
        cw = review_proxy.SHEET_CELL_WIDTH
        reds = [im.getpixel((c * cw + cw // 2, im.height // 2))[0] for c in range(3)]
    assert reds[0] > reds[2] > reds[1], f"cells out of beat order: {reds}"


def test_contact_sheet_is_cached_until_a_still_changes(tmp_path):
    for i in range(1, 4):
        _png(tmp_path / f"{i:02d}.png")
    first = review_proxy.contact_sheet(tmp_path)
    stamp = first.stat().st_mtime_ns
    assert review_proxy.contact_sheet(tmp_path).stat().st_mtime_ns == stamp

    import os, time
    time.sleep(0.01)
    _png(tmp_path / "04.png")
    os.utime(tmp_path / "04.png", None)
    assert review_proxy.contact_sheet(tmp_path).stat().st_mtime_ns != stamp


def test_contact_sheet_ignores_non_beat_pngs(tmp_path):
    """A thumbnail or a character reference sitting in the folder is not a
    beat and must not appear in the strip."""
    _png(tmp_path / "01.png")
    _png(tmp_path / "thumbnail.png")
    _png(tmp_path / "character_reference.png")
    from PIL import Image
    with Image.open(review_proxy.contact_sheet(tmp_path)) as im:
        assert im.width == review_proxy.SHEET_CELL_WIDTH   # exactly one cell


def test_contact_sheet_returns_none_without_stills(tmp_path):
    assert review_proxy.contact_sheet(tmp_path) is None


def test_contact_sheet_returns_none_for_a_missing_folder(tmp_path):
    assert review_proxy.contact_sheet(tmp_path / "nope") is None


# ── mp4 proxy ────────────────────────────────────────────────────────────────

def test_a_small_master_is_its_own_proxy(tmp_path):
    """Re-encoding something already under the limit only loses quality."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * 1024)
    assert review_proxy.build(src) == src
    assert not review_proxy.proxy_path(src).exists()


def test_missing_source_returns_none(tmp_path):
    assert review_proxy.build(tmp_path / "gone.mp4") is None


def test_empty_source_returns_none(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"")
    assert review_proxy.build(src) is None


def test_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_REVIEW_PROXY", "0")
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * 1024)
    assert review_proxy.build(src) is None


def test_a_failed_encode_leaves_no_partial_file(monkeypatch, tmp_path):
    """A half-written proxy would be served to the phone as a broken video and
    then cached by mtime, so the next review would serve it again."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * (review_proxy.TARGET_BYTES + 1))
    out = review_proxy.proxy_path(src)

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, b"", b"boom")

    monkeypatch.setattr(review_proxy.subprocess, "run", fake_run)
    assert review_proxy.build(src) is None
    assert not out.exists()


def test_missing_ffmpeg_is_survivable(monkeypatch, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * (review_proxy.TARGET_BYTES + 1))

    def boom(cmd, **kw):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(review_proxy.subprocess, "run", boom)
    assert review_proxy.build(src) is None


def test_proxy_never_overwrites_the_master(tmp_path):
    """The master is what gets uploaded to YouTube. Nothing here may touch
    it."""
    src = tmp_path / "v.mp4"
    assert review_proxy.proxy_path(src) != src
    assert review_proxy.proxy_path(src).name == "v.review.mp4"


def test_an_oversized_proxy_is_reported_as_a_miss(monkeypatch, tmp_path):
    """Handing the caller a file Discord will 413 on is worse than admitting
    the encode didn't get small enough."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * (review_proxy.TARGET_BYTES + 1))
    out = review_proxy.proxy_path(src)

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"y" * (review_proxy.TARGET_BYTES + 10))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(review_proxy.subprocess, "run", fake_run)
    assert review_proxy.build(src) is None


def test_encode_targets_half_height_and_faststart(monkeypatch, tmp_path):
    """faststart lets a phone start playing before the file has fully arrived
    — the difference between "loads" and "spins"."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * (review_proxy.TARGET_BYTES + 1))
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"y" * 2048)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(review_proxy.subprocess, "run", fake_run)
    review_proxy.build(src)
    cmd = seen["cmd"]
    assert f"scale=-2:{review_proxy.PROXY_HEIGHT}" in cmd
    assert "+faststart" in cmd
    assert review_proxy.AUDIO_BITRATE in cmd, "the voice is what's being judged"


# ── notify integration ───────────────────────────────────────────────────────

def test_send_file_shrinks_an_oversized_mp4_before_giving_up(monkeypatch, tmp_path):
    """The bug this closes: a 15-25MB Short is ALWAYS over Discord's wall, so
    the video branch of send_file never once delivered a video."""
    import notify

    big = tmp_path / "short.mp4"
    big.write_bytes(b"x" * (notify.DISCORD_MAX_UPLOAD_BYTES + 1))
    small = tmp_path / "short.review.mp4"
    small.write_bytes(b"y" * 4096)

    monkeypatch.setattr(notify, "_discord_webhook", lambda: "https://hook.test")
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(review_proxy, "build", lambda p, **kw: small)

    posted = {}

    class R:
        status_code = 200

    def fake_post(url, data=None, files=None, timeout=None):
        posted["name"] = files["file"][0]
        return R()

    monkeypatch.setattr(notify.requests, "post", fake_post)
    assert notify.send_file(big, caption="review me") is True
    assert posted["name"] == "short.review.mp4"


def test_send_file_still_posts_a_link_when_no_proxy_is_possible(monkeypatch, tmp_path):
    import notify

    big = tmp_path / "short.mp4"
    big.write_bytes(b"x" * (notify.DISCORD_MAX_UPLOAD_BYTES + 1))
    monkeypatch.setattr(notify, "_discord_webhook", lambda: "https://hook.test")
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(review_proxy, "build", lambda p, **kw: None)

    calls = []
    monkeypatch.setattr(notify, "_send_discord",
                        lambda *a, **kw: calls.append(a) or True)
    assert notify.send_file(big) is True
    assert calls, "must fall back to a link rather than silently dropping it"


def test_a_broken_proxy_module_cannot_break_notification(monkeypatch, tmp_path):
    import notify

    big = tmp_path / "short.mp4"
    big.write_bytes(b"x" * (notify.DISCORD_MAX_UPLOAD_BYTES + 1))
    monkeypatch.setattr(notify, "_discord_webhook", lambda: "https://hook.test")
    monkeypatch.setattr(notify, "enabled", lambda: True)

    def boom(*a, **kw):
        raise RuntimeError("no ffmpeg, no pillow, no nothing")

    monkeypatch.setattr(review_proxy, "build", boom)
    monkeypatch.setattr(notify, "_send_discord", lambda *a, **kw: True)
    assert notify.send_file(big) is True


def test_pending_review_alert_survives_a_failed_attachment(monkeypatch):
    """The alert is the contract; the video is a convenience. A failed upload
    must never swallow "a video is waiting for you"."""
    import notify
    sent = {}
    monkeypatch.setattr(notify, "send",
                        lambda *a, **kw: sent.setdefault("alert", True))

    def boom(*a, **kw):
        raise RuntimeError("discord down")

    monkeypatch.setattr(notify, "send_file", boom)
    notify.notify_pending_review(title="T", score=8, niche="money_history",
                                 video_id=1, video_path="/nope/x.mp4")
    assert sent["alert"] is True


def test_pending_review_attaches_the_video_when_given_one(monkeypatch):
    import notify
    monkeypatch.setattr(notify, "send", lambda *a, **kw: True)
    grabbed = {}
    monkeypatch.setattr(notify, "send_file",
                        lambda p, **kw: grabbed.setdefault("path", p))
    notify.notify_pending_review(title="T", score=8, niche="money_history",
                                 video_id=1, video_path="/out/v.mp4")
    assert grabbed["path"] == "/out/v.mp4"


def test_pending_review_without_a_video_path_is_unchanged(monkeypatch):
    import notify
    monkeypatch.setattr(notify, "send", lambda *a, **kw: True)

    def unexpected(*a, **kw):
        pytest.fail("send_file must not be called without a video_path")

    monkeypatch.setattr(notify, "send_file", unexpected)
    notify.notify_pending_review(title="T", score=8, niche="money_history",
                                 video_id=1)
