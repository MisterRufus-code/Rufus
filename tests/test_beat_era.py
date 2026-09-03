"""Per-beat era tagging, and reconciling a prompt's own camera spec with the
channel's style.

Both were diagnosed from live runs whose prompts contradicted themselves.

ERA — a money_history script is not set in one period. It opens in history and
lands in the viewer's present; that pivot IS the format. A run-wide period rule
therefore fights every beat written to be modern:

  run #58 beat 08  "economists and policymakers in a CONTEMPORARY setting"
                   + "Period setting: 1791 ... no business suits"
  run #58 beat 10  the Chronicler in a "timeless green field" + the same 1791
                   clause — which is why he came back in modern dress
  run #53 beat 10  "a CONTEMPORARY classroom" + "Period setting: 1791", on a
                   script about the 1923 Rentenmark: the year was wrong for the
                   entire video, not just that beat

CAMERA SPEC — the prompt-writer emits "Shot on a Canon EOS 5D Mark IV, 50mm
f/1.8 lens" out of habit. The old guard read that as "this prompt brought its
own style" and skipped the flat-2D suffix, rendering that ONE beat photoreal
among nine flat-vector ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main            # noqa: E402
import comfy_client    # noqa: E402


# ── Deriving the period from THIS run's script ───────────────────────────────

def test_period_comes_from_the_earliest_year_in_the_script():
    """The format opens in the past and moves toward the present, so the
    opening year is the setting and a later one is usually the payoff."""
    script = ("The First Bank of the United States was chartered in 1791. "
              "Debates still echo in 2024.")
    assert main._script_period(script) == "1791"


def test_period_handles_bc_dates():
    script = ("Around 2150 BC, the shekel emerged as a unit of weight. "
              "By the 5th century BCE, Philistia minted silver coins.")
    assert "2150" in main._script_period(script)


def test_no_year_means_no_period():
    """Without a year there is nothing to anchor to, and inventing one is how
    the 1923 Rentenmark script ended up labelled 1791."""
    assert main._script_period("Trust is what money has always run on.") == ""


def test_a_quantity_is_not_mistaken_for_a_year():
    """"under five percent" and "14.32 grams" must not become the setting."""
    assert main._script_period("It fell to under 5 percent, from 950 grams.") == ""


# ── Which beats are present-day ──────────────────────────────────────────────

def test_present_day_beats_are_detected():
    for beat in ("Today, this paradox is alive.",
                 "Your money looks fine too.",
                 "Debates about modern central banks still run.",
                 "You do it now, with the worn note and the crisp one.",
                 "A contemporary classroom, a map on the wall."):
        assert main._beat_is_present_day(beat), beat


def test_historical_beats_are_not_flagged():
    for beat in ("The First Bank of the United States was chartered in 1791.",
                 "Hamilton's vision was bold but incomplete.",
                 "People flooded the banks, demanding metal for their notes."):
        assert not main._beat_is_present_day(beat), beat


def test_era_tag_is_per_beat_not_per_run():
    """The exact failure: one script, two eras, and every beat must get the
    tag that matches ITS OWN sentence."""
    script = ("The First Bank was chartered in 1791. "
              "Today, this paradox is alive.")
    period = main._script_period(script)
    assert main._beat_era_tag("The First Bank was chartered in 1791.", period) == "1791"
    assert main._beat_era_tag("Today, this paradox is alive.", period) == "present day"


def test_a_present_day_beat_never_gets_a_year_even_when_it_names_one():
    """"Debates about what a central bank should do still echo" sat next to
    1791 in run #58. A beat that speaks about now is present-day regardless of
    what the rest of the script is set in."""
    assert main._beat_era_tag("Today's banks still repeat it, 300 years on.",
                              "1791") == "present day"


def test_no_period_falls_back_to_present_day_not_to_a_guess():
    assert main._beat_era_tag("Money was never what you think.", "") == "present day"


# ── The instruction actually carries the tags ────────────────────────────────

def test_instruction_states_the_era_rule_per_beat():
    src = Path(main.__file__).read_text(encoding="utf-8")
    assert "[ERA=" in src
    assert "present day" in src
    assert "period costume on a present-day" in src.lower()


# ── Camera spec vs. style ────────────────────────────────────────────────────

def test_photo_direction_is_stripped_clause_by_clause():
    """Token-level deletion left debris (".8 lens, with warm sepia tones and
    fine .") that was worse than the contradiction."""
    out = comfy_client._strip_photo_direction(
        "A close-up of several crumpled notes on a bank counter, their edges "
        "frayed. Shot on a Canon EOS 5D Mark IV, 50mm f/1.8 lens, with warm "
        "sepia tones and fine film grain.")
    assert out == ("A close-up of several crumpled notes on a bank counter, "
                   "their edges frayed.")
    for debris in ("mm", "f/", "Canon", "grain", "..", " ,"):
        assert debris not in out


def test_scene_content_survives_the_strip():
    out = comfy_client._strip_photo_direction(
        "A wide shot of a bank entrance, with a crowd of anxious people. The "
        "golden evening light filters through. Shot on a Sony A7R III, 24mm "
        "f/1.4 lens.")
    assert "crowd of anxious people" in out
    assert "golden evening light" in out
    assert "Sony" not in out


def test_a_clean_prompt_is_untouched():
    clean = "A medium portrait of a hooded figure carrying a bronze lantern."
    assert comfy_client._strip_photo_direction(clean) == clean


def test_flat_style_always_wins_over_a_stray_camera_spec():
    """The style is a channel-wide decision. One stray line must never opt a
    single beat out of it — mixed looks inside one Short are more obvious than
    either look on its own."""
    out = comfy_client._with_detail(
        "A worn gold coin on a counter. Shot on a Leica M10, 28mm f/2 lens.")
    assert "Flat 2D vector illustration" in out
    assert "Leica" not in out and "28mm" not in out


def test_photographic_style_defers_to_the_prompts_own_camera(monkeypatch):
    """Under a photographic style the prompt's own spec is a legitimate, more
    specific choice — the reason the original guard existed."""
    monkeypatch.setenv("RUFUS_STILLS_DETAIL",
                       "photorealistic, shot on a real camera, fine film grain")
    prompt = "A worn gold coin. Shot on a Leica M10, 28mm f/2 lens."
    assert comfy_client._with_detail(prompt) == prompt


def test_disabled_style_still_leaves_the_prompt_alone(monkeypatch):
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "")
    prompt = "A worn gold coin. Shot on a Leica M10, 28mm f/2 lens."
    assert comfy_client._with_detail(prompt) == prompt


def test_default_style_is_recognised_as_non_photographic():
    """The flat-2D default says "not a photograph" while still using the words
    photographic/photo-real to say what it is NOT — the check must read the
    negation, or the shipped default would classify itself as photographic and
    the strip would never run."""
    assert not comfy_client._is_photographic(comfy_client.DEFAULT_DETAIL_SUFFIX)
    assert comfy_client._is_photographic(
        "photorealistic, documentary photojournalism captured on a real camera")


# ── Second person is NOT a present-day marker on its own ─────────────────────
# Two rules this pipeline sets itself collided. The SOUND section requires
# every script to address the viewer ("A script with no 'you' in it is a
# lecture"), so "you" turns up in HISTORICAL beats as a rhetorical device.
# Treating it as present-day tagged one of those modern, and the prompt-writer
# obeyed: run #61's beat 6 came back as "a wide establishing shot of a MODERN
# BANK ... sleek architecture and digital displays" inside an 1865 story about
# the Latin Monetary Union.

def test_rhetorical_you_in_a_past_tense_beat_stays_historical():
    """The exact live beat: 'You could swap cheaper silver for premium gold at
    the fixed rate' is 1873, not now."""
    for beat in ("You could swap cheaper silver for premium gold at the fixed rate.",
                 "You would have paid in silver.",
                 "You had no way to check the coin's weight.",
                 "Your grandfather was paid in these."):
        assert not main._beat_is_present_day(beat), beat


def test_second_person_in_the_present_still_reads_as_now():
    for beat in ("Your money looks fine too.",
                 "You spend the worst coin first.",
                 "You do it now, with the worn note and the crisp one."):
        assert main._beat_is_present_day(beat), beat


def test_an_explicit_marker_beats_a_past_tense_verb():
    """"Today's banks still repeated the mistake" is about now, whatever tense
    the verb is in — an explicit marker must not be cancelled by one."""
    assert main._beat_is_present_day("Today's banks repeated exactly that.")
    assert main._beat_is_present_day("Modern central banks had the same gap.")


def test_a_historical_beat_with_no_second_person_is_unaffected():
    for beat in ("Traders exploited it — a quick way to profit.",
                 "The union was disbanded quietly.",
                 "Silver's value plunged in 1873."):
        assert not main._beat_is_present_day(beat), beat
