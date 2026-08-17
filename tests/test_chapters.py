"""A nine-minute video without chapters is one a viewer commits to blind.

The temptation is to divide the runtime by the number of sections, which would
be worse than having none: a mark thirty seconds off is a promise the video
breaks four times, and YouTube shows nothing at all for a list that breaks its
rules. These tests hold the line on both — the timestamps come from the audio
that actually shipped, and an invalid list is returned as no list.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import chapters  # noqa: E402


def _stream(text: str, start: float = 0.0, rate: float = 0.4):
    """A Whisper-style word stream: (start_seconds, word), one word per rate."""
    return [(start + i * rate, w) for i, w in enumerate(text.split())]


def test_a_stamp_grows_an_hour_field_only_when_there_is_one():
    assert chapters._stamp(0) == "0:00"
    assert chapters._stamp(7) == "0:07"
    assert chapters._stamp(247) == "4:07"
    assert chapters._stamp(3725) == "1:02:05"


def test_a_section_is_found_where_it_is_actually_spoken():
    opening = "Imagine you have not eaten in two days and it starts raining."
    one = "The rain was not an inconvenience at all it was a crisis."
    two = "By March the receivers had taken over the railroad office."
    script = f"{opening}\n\n{one}\n\n{two}"
    words = _stream(f"{opening} {one} {two}")

    starts = chapters.locate([one, two], words)
    assert starts[0] == pytest.approx(len(opening.split()) * 0.4)
    assert starts[1] == pytest.approx(
        (len(opening.split()) + len(one.split())) * 0.4)
    assert script  # the paragraphs are the ones the script is made of


def test_punctuation_and_casing_do_not_stop_a_match():
    """Whisper writes "rain," and the script says "rain" — matching on the
    raw strings would find nothing and silently drop every chapter."""
    para = "The rain, it turned out, was the crisis."
    words = _stream("the RAIN it turned out was the crisis")
    assert chapters.locate([para], words)[0] == 0.0


def test_a_section_that_was_never_spoken_is_dropped_not_estimated():
    words = _stream("one two three four five six seven eight nine ten")
    starts = chapters.locate(["a sentence that appears nowhere in the audio"],
                             words)
    assert starts == [-1.0]


def test_a_misheard_first_word_still_finds_the_section():
    """The probe shortens rather than giving up: one bad word at the head of a
    section should cost precision, never the whole chapter."""
    para = "Ledgers were burning in the counting house that night"
    words = _stream("something else entirely were burning in the counting house that night")
    assert chapters.locate([para], words)[0] > 0


def test_a_valid_list_starts_at_zero_and_names_the_sections():
    opening = " ".join(["opening"] * 30)
    secs = [" ".join([f"section{i}word{j}" for j in range(30)]) for i in range(3)]
    script = "\n\n".join([opening] + secs)
    words = _stream(" ".join([opening] + secs))
    out = chapters.build(script, words, ["The panic", "The rescue", "The bill"])

    assert out[0] == (0.0, "Intro")
    assert [t for _s, t in out[1:]] == ["The panic", "The rescue", "The bill"]
    assert [s for s, _t in out] == sorted(s for s, _t in out)


def test_two_sections_that_land_together_do_not_both_become_chapters():
    """YouTube requires ten seconds per chapter and shows NOTHING for a list
    that breaks the rule — so a mark that arrives too soon is dropped rather
    than shipped."""
    opening = " ".join(["opening"] * 30)
    near = "one two three four five six"
    secs = [near, " ".join(["later"] * 30), " ".join(["last"] * 30)]
    script = "\n\n".join([opening] + secs)
    words = _stream(" ".join([opening] + secs), rate=0.05)   # everything is close
    out = chapters.build(script, words, ["a", "b", "c"])
    assert out == []


def test_fewer_than_three_chapters_ships_nothing():
    opening = " ".join(["opening"] * 30)
    one = " ".join(["only"] * 30)
    script = f"{opening}\n\n{one}"
    words = _stream(f"{opening} {one}")
    assert chapters.build(script, words, ["the only one"]) == []


def test_a_chapter_in_the_last_ten_seconds_is_dropped():
    opening = " ".join(["opening"] * 30)
    secs = [" ".join([f"s{i}w{j}" for j in range(30)]) for i in range(3)]
    script = "\n\n".join([opening] + secs)
    words = _stream(" ".join([opening] + secs))
    duration = words[-1][0] + 1.0
    out = chapters.build(script, words, ["a", "b", "c"], duration=duration)
    assert all(duration - s >= chapters.MIN_CHAPTER_S for s, _t in out) or out == []


def test_no_timings_no_titles_no_chapters():
    assert chapters.build("a\n\nb", [], ["x"]) == []
    assert chapters.build("a\n\nb", _stream("a b"), []) == []
    assert chapters.build("", _stream("a b"), ["x"]) == []


def test_the_lines_are_the_shape_youtube_parses():
    out = chapters.as_lines([(0.0, "Intro"), (94.0, "The panic"),
                             (247.5, "The bill")])
    assert out == "0:00 Intro\n1:34 The panic\n4:07 The bill"


def test_chapters_are_a_long_form_thing(monkeypatch):
    """Three chapters inside forty seconds would each be shorter than the
    platform's own ten-second floor."""
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    assert chapters.enabled() is False
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    assert chapters.enabled() is True


# ── the seams, which is where this could quietly do nothing ──────────────────

def test_the_titles_come_from_the_sections_that_survived():
    """A section that came back too short is skipped, and a chapter list built
    from the PLAN would name the skipped one and put every timestamp after it
    against the wrong paragraph."""
    src = (Path(__file__).parent.parent / "scripts" / "longform_writer.py").read_text(encoding="utf-8")
    assert "kept_titles.append" in src
    assert '"section_titles": kept_titles' in src


def test_both_renderers_publish_the_word_stream():
    """A description whose timestamps depend on which engine drew the frames
    is a bug that only shows up on half the runs."""
    scripts = Path(__file__).parent.parent / "scripts"
    for name in ("audio_gen.py", "remotion_renderer.py"):
        assert "LAST_WORDS" in (scripts / name).read_text(encoding="utf-8"), name


def test_the_block_goes_under_the_opening_paragraph_not_on_top_of_it():
    """The first two lines of a description are what search shows. Spending
    them on "0:00 Intro" throws away the copy that earns the click."""
    import youtube_uploader as yu

    def fake_meta(script, niche, cfg, hashtags=None):
        return {"title": "t", "description": "The panic of 1893, in nine "
                                             "minutes.\n\n#history",
                "tags": ["x"]}

    import metadata_writer
    orig = metadata_writer.generate_metadata
    metadata_writer.generate_metadata = fake_meta
    try:
        meta = yu.build_metadata("script", "money_history", {},
                                 chapters="0:00 Intro\n1:34 The panic")
    finally:
        metadata_writer.generate_metadata = orig

    lines = meta["description"].splitlines()
    assert lines[0].startswith("The panic of 1893")
    assert "0:00 Intro" in meta["description"]
    assert meta["description"].index("0:00 Intro") < meta["description"].index("#history")


def test_a_second_pass_does_not_stack_two_lists():
    import youtube_uploader as yu

    def fake_meta(script, niche, cfg, hashtags=None):
        return {"title": "t", "description": "0:00 Intro\n1:34 The panic",
                "tags": ["x"]}

    import metadata_writer
    orig = metadata_writer.generate_metadata
    metadata_writer.generate_metadata = fake_meta
    try:
        meta = yu.build_metadata("script", "money_history", {},
                                 chapters="0:00 Intro\n1:34 The panic")
    finally:
        metadata_writer.generate_metadata = orig
    assert meta["description"].count("0:00 Intro") == 1
