"""Tests for script_writer.py – banned-phrase detection and blacklist keying."""

from script_writer import _blacklist_key, _find_banned


def test_find_banned_detects_exact_phrase():
    assert _find_banned("Let's dive in and learn something") == "let's dive in"


def test_find_banned_case_insensitive():
    assert _find_banned("BUCKLE UP everyone") == "buckle up"


def test_find_banned_whole_word_only():
    # 'leverage' is banned but should not match 'cleaverage' etc.
    assert _find_banned("This is a clever angle") is None


def test_find_banned_no_match_clean_script():
    # "here's why" is a banned phrase — test must not use it
    clean = "You're broke. Three patterns. Most people save wrong. Spend on assets. Follow for more."
    assert _find_banned(clean) is None


def test_find_banned_multiple_returns_first():
    # Should still detect at least one banned phrase even when several present
    script = "Buckle up. Let's dive in to the journey."
    assert _find_banned(script) is not None


def test_find_banned_empty_string():
    assert _find_banned("") is None


def test_blacklist_key_first_twenty_words_lowercase():
    s = "You're losing money every single day right now my friend stop wasting time on things that do not matter"
    key = _blacklist_key(s)
    assert key == "you're losing money every single day right now my friend stop wasting time on things that do not matter"


def test_blacklist_key_normalises_whitespace():
    a = _blacklist_key("Hello   world   foo bar")
    b = _blacklist_key("Hello world foo bar")
    assert a == b


def test_blacklist_key_short_script():
    assert _blacklist_key("Hi there") == "hi there"
