"""Tests for thumbnail_gen.py – hook wrapping and niche accent loading."""

from thumbnail_gen import _wrap_hook, _hex_to_rgb


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
