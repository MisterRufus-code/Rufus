"""Tests for thumbnail_gen.py – hook text extraction."""

from thumbnail_gen import _hook_text


def test_hook_text_uppercases_first_line():
    # Apostrophes are stripped (FFmpeg drawtext can't escape them safely)
    assert _hook_text("you're broke") == "YOURE BROKE"


def test_hook_text_limits_to_three_words():
    assert _hook_text("one two three four five") == "ONE TWO THREE"


def test_hook_text_uses_only_first_line():
    assert _hook_text("Hook line\nSecond line ignored") == "HOOK LINE"


def test_hook_text_strips_punctuation():
    # Commas and periods stripped (kept readable as a thumbnail header)
    assert _hook_text("stop, scrolling.") == "STOP SCROLLING"


def test_hook_text_drops_unsafe_chars():
    # Single quotes and colons removed (would break ffmpeg drawtext)
    result = _hook_text("don't: skip")
    assert "'" not in result
    assert ":" not in result


def test_hook_text_handles_empty_script():
    assert _hook_text("") == ""


def test_hook_text_respects_max_words_param():
    assert _hook_text("alpha bravo charlie delta", max_words=2) == "ALPHA BRAVO"
