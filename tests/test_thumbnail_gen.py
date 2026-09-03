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


# ── composing on a supplied background, at whatever size it is ─────────────
#
# Every drawing helper used to read the module-level THUMB_W/THUMB_H, which
# come from video_format.dimensions() — the VIDEO's shape, portrait for Shorts.
# Correct for the one thing this file could do, and wrong for a 1280x720
# YouTube thumbnail: the gradient lands off the bottom, the accent bar is drawn
# two and a half times too long, and the badge falls outside the frame.

LANDSCAPE = (1280, 720)


def _bg(size=LANDSCAPE, colour=(30, 44, 72)):
    from PIL import Image
    return Image.new("RGB", size, colour)


def test_compose_returns_an_image_the_size_of_its_background(tmp_path):
    from PIL import Image
    out = tg.compose(_bg(), "Gold", tmp_path / "t.jpg")
    assert Image.open(out).size == LANDSCAPE


def test_the_gradient_leaves_the_top_of_the_image_alone(tmp_path):
    """THE ONE THAT WAS ACTUALLY BROKEN. band_h was int(THUMB_H * 0.45) — 864
    rows — drawn onto a 720-tall thumbnail, so the band started at row -144 and
    covered the WHOLE image: alpha 13 at the very top rising to 99 by 55% of
    the way down. The subject of the picture was being dimmed to make room for
    text that sits in the bottom third.

    Nothing errors when you draw past an edge, which is why this survived.
    """
    from PIL import Image
    colour = (30, 44, 72)
    out = tg.compose(_bg(colour=colour), "Gold", tmp_path / "t.jpg")
    im = Image.open(out).convert("RGB")
    # Right of the accent bar, above the band, away from the badge corner.
    got = im.getpixel((int(LANDSCAPE[0] * 0.45), int(LANDSCAPE[1] * 0.20)))
    assert sum(abs(a - b) for a, b in zip(got, colour)) < 12, got


def test_the_accent_bar_scales_with_the_image(tmp_path):
    """A flat 12px is a confident stripe on a 1080-wide portrait frame and a
    hairline on a landscape one. A brand mark that disappears at one of the two
    sizes it is used at is not a brand mark."""
    from PIL import Image
    out = tg.compose(_bg(), "Gold", tmp_path / "t.jpg")
    im = Image.open(out).convert("RGB")
    accent = tg._hex_to_rgb(tg._load_niche_accent())
    row = LANDSCAPE[1] // 2
    width = 0
    for x in range(40):
        px = im.getpixel((x, row))
        if sum(abs(a - b) for a, b in zip(px, accent)) < 90:
            width = x + 1
        else:
            break
    assert width >= 12, f"accent bar is only {width}px wide"


def test_a_short_headline_is_drawn_large(tmp_path):
    """Auto-fit shrank only, on the theory that inflating one word would look
    like an accident. Side by side at feed size that was simply wrong: the
    thumbnail's whole job is to be readable as a postage stamp."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(_bg())
    font = tg._fit_font(draw, ["GOLD"], tg._find_font(),
                                   900, 300, 187)
    assert getattr(font, "size", 0) > tg.FONT_SIZE


def test_a_long_headline_is_shrunk_rather_than_overflowing(tmp_path):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(_bg())
    lines = ["NOBODY TOLD THEM THE MONEY WAS ALREADY WORTHLESS"]
    font = tg._fit_font(draw, lines, tg._find_font(),
                                   900, 300, 187)
    assert getattr(font, "size", 999) < 187


# ── a typed headline must never lose a word ───────────────────────────────

def test_the_wrap_used_for_a_typed_headline_keeps_every_word():
    """_wrap_hook's 2-line cap DROPS the rest, which is right for a spoken
    hook pulled off a script and wrong for words a person typed. Quietly
    deleting somebody's last word is only noticed after it has gone out."""
    text = "Nobody told them the money was already worthless"
    assert tg._keeps_every_word(text, tg._best_wrap(text))


def test_the_lossy_default_is_still_the_default():
    """The video path relies on it: two lines of a spoken sentence is a
    thumbnail, the rest is narration nobody reads at 168x94."""
    text = "one two three four five six seven eight nine ten eleven twelve"
    assert len(tg._wrap_hook(text)) == 2
    assert not tg._keeps_every_word(text, tg._wrap_hook(text))


def test_fewest_lines_is_not_what_wins():
    """"THE BANK THAT PRINTED ITSELF" fits on one line and is therefore tiny —
    a 28-character line is limited by the box WIDTH long before its height.
    Broken over two it renders half again as large."""
    lines = tg._best_wrap("The bank that printed itself")
    assert len(lines) == 2, lines


def test_an_unbreakable_headline_still_produces_a_thumbnail(tmp_path):
    """One forty-letter word has no wrap that works. Half of it on screen
    beats a blank thumbnail."""
    from PIL import Image
    out = tg.compose(_bg(), "A" * 40, tmp_path / "t.jpg")
    assert Image.open(out).size == LANDSCAPE


def test_an_empty_headline_composes_the_branding_and_no_text(tmp_path):
    from PIL import Image
    out = tg.compose(_bg(), "", tmp_path / "t.jpg")
    assert Image.open(out).size == LANDSCAPE


# ── what YouTube draws on top ─────────────────────────────────────────────

def test_the_text_clears_the_corner_youtube_stamps(tmp_path):
    """The duration badge sits bottom-right and the progress bar runs across
    the bottom on hover. The text was centred 80px from the bottom — under
    both."""
    from PIL import Image
    out = tg.compose(_bg(colour=(10, 10, 10)),
                                "Nobody told them the money was already worthless",
                                tmp_path / "t.jpg")
    im = Image.open(out).convert("L")
    w, h = im.size
    corner = im.crop((int(w * 0.86), int(h * 0.90), w, h))
    assert max(corner.getdata()) < 110, "something is drawn where the badge goes"


# ── one composer, not two ─────────────────────────────────────────────────

def test_make_thumbnail_goes_through_compose():
    """Two implementations of "what a Rufus thumbnail looks like" drift, and
    the one that drifts is the one nobody notices is wrong."""
    src = Path(tg.__file__).read_text(encoding="utf-8")
    body = src.split("def make_thumbnail", 1)[1]
    assert "return compose(" in body
    assert "_draw_gradient_overlay" not in body, "make_thumbnail composes its own"
