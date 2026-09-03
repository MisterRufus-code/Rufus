"""Tests for the SD perceptual-image de-duplication helpers. Pure-function
level — no Automatic1111 needed (we synthesize tiny PNGs with PIL)."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import sd_client as sd


def _solid_png(tmp_path: Path, name: str, color) -> Path:
    p = tmp_path / name
    Image.new("RGB", (64, 128), color).save(str(p))
    return p


def _gradient_png(tmp_path: Path, name: str, flip: bool = False) -> Path:
    p   = tmp_path / name
    img = Image.new("L", (64, 128))
    for y in range(128):
        v = int(255 * (y / 127)) if not flip else int(255 * (1 - y / 127))
        for x in range(64):
            img.putpixel((x, y), v)
    img.convert("RGB").save(str(p))
    return p


# ── _avg_hash ────────────────────────────────────────────────────────────────────

def test_avg_hash_returns_64bit_int(tmp_path):
    h = sd._avg_hash(_solid_png(tmp_path, "g.png", (120, 120, 120)))
    assert h is not None
    assert 0 <= h < (1 << 64)          # fits in 64 bits (8×8 grid)


def test_avg_hash_identical_images_same_hash(tmp_path):
    a = _gradient_png(tmp_path, "a.png")
    b = _gradient_png(tmp_path, "b.png")          # byte-identical content
    ha, hb = sd._avg_hash(a), sd._avg_hash(b)
    assert ha is not None and hb is not None
    assert ha == hb
    assert sd._hamming(ha, hb) == 0


def test_avg_hash_different_images_differ(tmp_path):
    top = _gradient_png(tmp_path, "top.png", flip=False)
    bot = _gradient_png(tmp_path, "bot.png", flip=True)   # inverted gradient
    ht, hb = sd._avg_hash(top), sd._avg_hash(bot)
    assert ht is not None and hb is not None
    # An inverted gradient should be far apart in hash space.
    assert sd._hamming(ht, hb) >= sd.DUP_THRESHOLD


def test_avg_hash_missing_file_returns_none(tmp_path):
    assert sd._avg_hash(tmp_path / "does_not_exist.png") is None


# ── _hamming ───────────────────────────────────────────────────────────────────

def test_hamming_known_values():
    assert sd._hamming(0b0000, 0b0000) == 0
    assert sd._hamming(0b1010, 0b0101) == 4
    assert sd._hamming(0b1111, 0b0000) == 4
    assert sd._hamming(0xFF, 0x0F) == 4


# ── duplicate detection (the guarantee that an image "doesn't come back") ────────

def test_identical_images_flagged_as_duplicate(tmp_path):
    a = _gradient_png(tmp_path, "a.png")
    b = _gradient_png(tmp_path, "b.png")
    ha, hb = sd._avg_hash(a), sd._avg_hash(b)
    dist = sd._hamming(ha, hb)
    assert dist < sd.DUP_THRESHOLD          # would trigger a regen in generate_clips


def test_distinct_images_not_flagged(tmp_path):
    a = _gradient_png(tmp_path, "a.png", flip=False)
    b = _gradient_png(tmp_path, "b.png", flip=True)
    dist = sd._hamming(sd._avg_hash(a), sd._avg_hash(b))
    assert dist >= sd.DUP_THRESHOLD         # accepted as a new, distinct image


def test_dedup_constants_sane():
    assert 0 < sd.DUP_THRESHOLD <= 64
    assert sd.MAX_DUP_RETRIES >= 1
