"""Three defects from run 79 — the first run to reach 24 beats.

That run was the best the pipeline has produced (24 distinct prompts, 72
images, QC clean for the first time) and it exposed three things that only
show up at that density.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main  # noqa: E402
import storyboard  # noqa: E402


# ── the setting clause on every other frame ─────────────────────────────────

def test_the_setting_is_capped_like_the_thread():
    """At ten shots the pin fired three or four times and read as a reminder.
    At twenty-four it fired THIRTEEN, and "Same place as the rest of the
    sequence: market, cobblestone, bright" on half a sequence is a second
    description competing with each shot's own."""
    assert storyboard.SETTING_SHARE < storyboard.THREAD_SHARE
    assert max(2, round(24 * storyboard.SETTING_SHARE)) <= 6


def test_the_cap_never_falls_below_two():
    """A short sequence still gets its place established."""
    assert max(2, round(4 * storyboard.SETTING_SHARE)) == 2


# ── a shot that quotes words is asking for lettering ────────────────────────

def test_a_signpost_reading_something_is_caught():
    """`\\bsign\\b` does not match "signpost", so this shipped as-is — and the
    giveaway was never the noun anyway."""
    shot = "A signpost reading 'Marshalltown, Iowa' amidst the market hustle."
    assert main._defuse_readable_text(shot) != shot


def test_the_quoting_construction_is_caught_whatever_it_hangs_on():
    for shot in ("A book titled 'How Money Really Began' on a table.",
                 "A banner reading 'Part of a series.'",
                 "A stone that says nothing at all",
                 "A ribbon emblazoned with the words of the act"):
        assert main._defuse_readable_text(shot) != shot, shot


def test_a_clean_shot_is_left_alone():
    """The clause is appended only when triggered — every prompt carrying it
    would dilute all of them and cost tokens on all of them."""
    for shot in ("A worn silver coin lies alone on a bare wooden counter.",
                 "Two figures shake hands beside a market stall, brows raised."):
        assert main._defuse_readable_text(shot) == shot, shot


# ── the sign-off is not a scene ─────────────────────────────────────────────

def test_the_prompt_says_the_last_line_is_a_sign_off():
    """Run 79 closed on "a banner reading 'Part of a series'" and "a book
    titled 'How Money Really Began'" — two frames of garbled lettering, from
    taking the channel's fixed CTA as a scene to illustrate."""
    p = storyboard._prompt("script", ["a", "b"], [], scene="")
    assert "THE LAST LINE IS A SIGN-OFF, NOT A SCENE" in p
    assert "Never a banner, a book cover, a title card or a logo." in p
