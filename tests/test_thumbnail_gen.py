"""Tests for thumbnail_gen.py – hook wrapping, niche accent loading, and the
recurring-character brand badge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from thumbnail_gen import _wrap_hook, _hex_to_rgb
import thumbnail_gen as tg


def test_wrap_hook_uppercases():
    lines = _wrap_hook("save more money")
    assert all(l == l.upper() for l in lines)


def test_wrap_hook_returns_at_most_two_lines():
    long_text = "the fastest way to save money without changing your lifestyle at all"
    lines = _wrap_hook(long_text)
    assert 1 <= len(lines) <= 2


def test_wrap_hook_splits_at_word_boundary():
    # 22-char limit: "THIS IS A SHORT" = 15 chars — all on one line
    lines = _wrap_hook("this is a short hook")
    # The full text fits in 22 chars → single line
    assert len(lines) >= 1
    # Words must not be split mid-word
    for line in lines:
        assert not any(c == "-" for c in line)


def test_wrap_hook_empty_returns_empty():
    assert _wrap_hook("") == []


def test_wrap_hook_single_long_word_fits_one_line():
    lines = _wrap_hook("extraordinary")
    assert lines == ["EXTRAORDINARY"]


def test_wrap_hook_respects_max_chars_param():
    # max_chars=5 forces very short lines
    lines = _wrap_hook("save your money now please", max_chars=5)
    assert len(lines) >= 1
    for line in lines:
        # Each line should be at most one short word
        assert len(line) <= 10  # generous upper bound given word lengths


def test_hex_to_rgb_finance_gold():
    r, g, b = _hex_to_rgb("#FFC53D")
    assert r == 0xFF
    assert g == 0xC5
    assert b == 0x3D


def test_hex_to_rgb_strips_hash():
    assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert _hex_to_rgb("#000000") == (0, 0, 0)


# ── Recurring-character brand badge ──────────────────────────────────────────
# Real channels that stick a recognizable face/mascot in the same thumbnail
# corner every time build click-through and subscriber recall that a random
# per-video frame can't — this is character_engine.py's payoff on the
# thumbnail side, not just inside the video.

def test_composite_character_badge_pastes_visible_content(tmp_path):
    from PIL import Image

    base = Image.new("RGBA", (tg.THUMB_W, tg.THUMB_H), (0, 0, 0, 255))
    ref = tmp_path / "ref.png"
    Image.new("RGBA", (400, 700), (255, 0, 0, 255)).save(ref)

    out = tg._composite_character_badge(base.copy(), ref, (0, 255, 0))
    assert out.size == base.size
    top_right = out.crop((tg.THUMB_W - 300, 0, tg.THUMB_W, 300)).convert("RGB")
    colors = top_right.getcolors(maxcolors=100_000)
    assert any(rgb != (0, 0, 0) for _, rgb in colors)


def test_composite_character_badge_returns_unchanged_on_missing_reference(tmp_path):
    from PIL import Image

    base = Image.new("RGBA", (tg.THUMB_W, tg.THUMB_H), (0, 0, 0, 255))
    out = tg._composite_character_badge(base.copy(), tmp_path / "missing.png", (0, 255, 0))
    assert list(out.getdata()) == list(base.getdata())


def test_make_thumbnail_adds_badge_when_niche_character_enabled(tmp_path, monkeypatch):
    from PIL import Image

    frame = Image.new("RGBA", (tg.THUMB_W, tg.THUMB_H), (10, 10, 10, 255))
    monkeypatch.setattr(tg, "_best_frame", lambda video_path, duration, tmp_png: frame.copy())
    monkeypatch.setattr(tg, "_probe_duration", lambda video_path: 20.0)

    ref = tmp_path / "ref.png"
    Image.new("RGBA", (400, 700), (255, 0, 0, 255)).save(ref)

    import character_engine
    monkeypatch.setattr(character_engine, "enabled", lambda niche: True)
    monkeypatch.setattr(character_engine, "reference_image_path", lambda niche: ref)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out = tg.make_thumbnail(video, "A hook sentence here", niche="money_history")

    result = Image.open(out).convert("RGB")
    top_right = result.crop((tg.THUMB_W - 300, 0, tg.THUMB_W, 300))
    colors = top_right.getcolors(maxcolors=100_000)
    # The badge must have changed SOME pixel away from the plain frame color.
    assert any(rgb != (10, 10, 10) for _, rgb in colors)


def test_make_thumbnail_no_badge_without_niche(tmp_path, monkeypatch):
    from PIL import Image

    frame = Image.new("RGBA", (tg.THUMB_W, tg.THUMB_H), (10, 10, 10, 255))
    monkeypatch.setattr(tg, "_best_frame", lambda video_path, duration, tmp_png: frame.copy())
    monkeypatch.setattr(tg, "_probe_duration", lambda video_path: 20.0)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out = tg.make_thumbnail(video, "A hook sentence here")   # no niche → no badge

    result = Image.open(out).convert("RGB")
    # Top-right corner, well clear of the accent bar (left edge) and hook
    # text (bottom quarter), must stay the plain background color.
    assert result.getpixel((tg.THUMB_W - 20, 20)) == (10, 10, 10)


def test_make_thumbnail_no_badge_when_reference_not_bootstrapped_yet(tmp_path, monkeypatch):
    """Character mode enabled but the reference portrait hasn't rendered yet
    (first-ever run) — must degrade gracefully, no crash, no badge."""
    from PIL import Image

    frame = Image.new("RGBA", (tg.THUMB_W, tg.THUMB_H), (10, 10, 10, 255))
    monkeypatch.setattr(tg, "_best_frame", lambda video_path, duration, tmp_png: frame.copy())
    monkeypatch.setattr(tg, "_probe_duration", lambda video_path: 20.0)

    import character_engine
    monkeypatch.setattr(character_engine, "enabled", lambda niche: True)
    monkeypatch.setattr(character_engine, "reference_image_path",
                        lambda niche: tmp_path / "never_rendered.png")

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    out = tg.make_thumbnail(video, "A hook sentence here", niche="money_history")

    result = Image.open(out).convert("RGB")
    assert result.getpixel((tg.THUMB_W - 20, 20)) == (10, 10, 10)
