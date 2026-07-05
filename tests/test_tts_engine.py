"""Tests for tts_engine.py — punctuation-driven pause shaping for Kokoro."""

from tts_engine import _pause_seconds


def test_pause_after_full_stop():
    assert _pause_seconds("The bank called.") == 0.26


def test_pause_after_question_or_exclaim():
    assert _pause_seconds("Then what happened?") == 0.30
    assert _pause_seconds("She lost everything!") == 0.30


def test_pause_after_dramatic_beat():
    assert _pause_seconds("Until the day it stopped—") == 0.32
    assert _pause_seconds("He waited...") == 0.32


def test_pause_after_comma_is_shortest():
    assert _pause_seconds("First, the market crashed,") == 0.14


def test_pause_trims_trailing_whitespace_before_checking_punctuation():
    assert _pause_seconds("The bank called.   \n") == 0.26


def test_pause_empty_chunk_defaults_light():
    assert _pause_seconds("") == 0.15


def test_pause_no_recognized_punctuation_defaults_light():
    assert _pause_seconds("no punctuation here") == 0.15
