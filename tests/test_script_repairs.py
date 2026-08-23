"""Mechanical violations are repaired, not bought with a whole generation.

From a live run, one script, three attempts:

    attempt 1/3 – rejected (too long (120 words, cap 115))
    attempt 2/3 – rejected (cadence: missing a longer, flowing sentence)
    attempt 3/3 – score: 5/10
    ⚠ best score was 6/10 (target ≥7) — using best attempt
    cycle 1/3 not good enough — retrying with a DIFFERENT angle

Two of the three attempts were spent on edits a human editor makes in seconds:
a five-word overage and a missing long sentence. The model was never scored on
its best work. That is the wasted-generation rejection ladder AGENTS.md warns
about, and banned phrases were already exempted from it for exactly this
reason ("repaired banned phrase in place (no retry spent)").

The cadence rejection had a second cause worth recording: the prompt's DELIVERY
section told the model to "split it into short sentences instead" while the
cadence gate required a sentence of 15+ words. The prompt and the gate
contradicted each other, so the model was being punished for obeying it.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer as sw


HOOK = "Why didn't four hundred billion convert the Fuggers?"
LOOP = "So why did the richest man in Europe stay Catholic?"
CTA  = "Save this for later."


def _script(*body: str) -> str:
    return "\n".join([HOOK, *body, LOOP, CTA])


# ------------------------------------------------------------ length repair

def test_a_small_overage_is_trimmed_instead_of_rejected():
    long_body = " ".join(f"Sentence number {i} carries some real weight here."
                         for i in range(30))
    script = _script(long_body)
    assert len(script.split()) > 115

    out = sw._repair_length(script, 115)
    assert len(out.split()) <= 115


def test_the_hook_and_the_cta_survive_a_trim():
    """Line 1 is what the whole video is scored on; the last line is dictated
    verbatim by the niche; the loop must keep echoing the hook."""
    long_body = " ".join(f"Filler sentence {i} about money and banks." for i in range(40))
    out = sw._repair_length(_script(long_body), 115)
    lines = [l for l in out.split("\n") if l.strip()]
    assert lines[0] == HOOK
    assert lines[-1] == CTA
    assert lines[-2] == LOOP


def test_a_script_already_under_cap_is_untouched():
    script = _script("Jakob Fugger lent the crown eight tonnes of silver.")
    assert sw._repair_length(script, 115) == script


def test_a_script_with_no_middle_is_returned_unchanged():
    tiny = "\n".join([HOOK, CTA])
    assert sw._repair_length(tiny, 115) == tiny


def test_the_trim_terminates_on_an_unsplittable_script():
    """One enormous unpunctuated middle line must not loop forever."""
    script = _script("word " * 400)
    out = sw._repair_length(script, 115)
    assert isinstance(out, str) and out.strip()


# ----------------------------------------------------------- cadence repair

def test_two_short_sentences_are_joined_into_a_long_one():
    script = _script(
        "He lent the crown eight tonnes of silver.",
        "The emperor repaid him with copper mines instead.")
    assert sw._cadence_violation(script) is not None

    out = sw._repair_cadence(script)
    lengths = [len(s.split()) for s in sw._SENTENCE_RE.findall(out)]
    assert any(n >= 15 for n in lengths)


def test_the_join_reads_as_one_sentence():
    script = _script("He lent the crown eight tonnes of silver.",
                     "The emperor repaid him with copper mines instead.")
    out = sw._repair_cadence(script)
    assert ", the emperor" in out


def test_cadence_repair_protects_the_hook_and_cta():
    script = _script("He lent the crown eight tonnes of silver.",
                     "The emperor repaid him with copper mines instead.")
    out = sw._repair_cadence(script)
    lines = [l for l in out.split("\n") if l.strip()]
    assert lines[0] == HOOK
    assert lines[-1] == CTA


def test_a_script_with_nothing_joinable_is_returned_unchanged():
    script = _script("A coin.")
    assert sw._repair_cadence(script) == script


def test_a_join_that_would_be_absurdly_long_is_skipped():
    """Joining two 20-word sentences makes a 40-word monster — worse than the
    violation it fixes."""
    a = " ".join(["word"] * 20) + "."
    b = " ".join(["other"] * 20) + "."
    script = _script(f"{a} {b}")
    assert sw._repair_cadence(script) == script


# -------------------------------------------------------- the prompt itself

def test_the_prompt_states_a_target_not_just_a_range():
    """A model given '80-115 words' lands at 115. It needs a number to aim at."""
    std = sw._standards()["body"]
    target = std["min_words"] + (std["max_words"] - std["min_words"]) * 2 // 3
    assert std["min_words"] < target < std["max_words"]


def test_the_prompt_and_the_cadence_gate_no_longer_contradict():
    """The gate requires a 15+ word sentence; the prompt used to demand only
    short ones. Whatever the prompt says now, it must state the requirement."""
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "RHYTHM (required" in src
    assert "15 words or more" in src
    assert "6 words or fewer" in src


def test_the_repairs_are_wired_into_the_generation_loop():
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "_repair_length" in src and "_repair_cadence" in src
    assert "no retry spent" in src


# ── the repair declined on the exact shape the prompt produces ───────────────
#
# _repair_cadence joined EXACTLY TWO adjacent sentences. Measured against the
# shapes a live body actually takes, that is the wrong number:
#
#     8, 9, 10, 9, 8   pairs to 17-19   repaired
#     6, 6, 6, 6, 6    pairs to 12      DECLINED — below the gate's 15
#     5, 5, 5, 5, 5    pairs to 10      DECLINED
#
# And all-short is not an unlucky shape, it is the shape DELIVERY asks for:
# "Never write a long comma-chained run-on when the moment deserves a hard
# stop. Split it into short sentences instead." The model complies, the gate
# rejects it for having only short sentences, and the repair written to absorb
# that rejection refuses because two of them are still not long enough. Three
# six-word sentences make eighteen.
#
# The other half of the gate — "missing a short, punchy sentence (≤6 words)" —
# had no repair at all. 21 live rejections, each one buying with a whole
# generation an edit that is one keystroke: promote a comma to a full stop.

def _body(lengths, hook="Are your coins secretly worthless?",
          cta="Follow for more."):
    mid = [" ".join(["word%d" % i] * (n - 1) + ["end."]).capitalize()
           for i, n in enumerate(lengths)]
    return "\n".join([hook] + mid
                     + ["Every coin is a promise someone can break.", cta])


@pytest.mark.parametrize("lengths", [[6, 6, 6, 6, 6, 6], [5, 5, 5, 5, 5, 5, 5],
                                     [4, 5, 4, 5, 4]])
def test_a_run_of_short_sentences_is_joined_until_it_is_long_enough(lengths):
    script = _body(lengths)
    assert "longer, flowing" in (sw._cadence_violation(script) or "")
    assert sw._cadence_violation(sw._repair_cadence(script)) is None


@pytest.mark.parametrize("lengths", [[8, 9, 10, 9, 8], [11, 12, 11], [12, 13, 12, 13]])
def test_the_shapes_that_already_worked_still_work(lengths):
    script = _body(lengths)
    assert sw._cadence_violation(sw._repair_cadence(script)) is None


def test_a_joined_run_reads_back_as_one_sentence():
    """The middle parts have to lose their full stops too. Left in place, the
    "long sentence" splits back into three the moment the gate re-reads it —
    the repair reports success and changes nothing that counts."""
    out = sw._repair_cadence(_body([6, 6, 6, 6, 6, 6]))
    lengths = [len(s.split()) for s in sw._SENTENCE_RE.findall(out) if s.strip()]
    assert any(n >= 15 for n in lengths), lengths


def test_two_long_sentences_are_still_not_joined_into_a_run_on():
    """26 words is the cap and it stays. DELIVERY forbids run-ons, so a repair
    that produces one has traded a rhythm complaint for a worse sentence."""
    script = _body([14, 13, 14, 13])
    assert sw._repair_cadence(script) == script


def test_a_missing_short_sentence_is_repaired_by_promoting_a_comma():
    script = ("How did Spain's own silver quietly destroy its economy?\n"
              "By 1600 the crown was bankrupt, despite owning the richest "
              "silver mine anywhere on the planet.\n"
              "Prices in Seville rose four hundred percent across the century "
              "that followed the first discovery.\n"
              "The fleets kept arriving and the money kept losing its meaning "
              "with every single voyage home.\n"
              "The silver that was meant to make Spain rich is what emptied "
              "it completely.\n"
              "Subscribe now for more stories about the history of money.")
    assert "punchy" in (sw._cadence_violation(script) or "")
    out = sw._repair_cadence(script)
    assert sw._cadence_violation(out) is None
    assert "bankrupt. Despite" in out


def test_the_short_repair_leaves_a_sentence_it_cannot_cut_cleanly_alone():
    """No comma at a place where one side is a plausible short sentence means
    no safe cut. A repair that hacked one in would put an ungrammatical line
    into the narration, which is what the join rule already warns about."""
    script = ("How did Spain's own silver quietly destroy its economy?\n"
              "In 1545 the mines at Potosi began sending enormous quantities "
              "of silver back across the Atlantic.\n"
              "Prices in Seville rose four hundred percent across the century "
              "that followed the discovery.\n"
              "The silver that was meant to make Spain rich is what emptied "
              "it completely.\n"
              "Subscribe now for more stories about the history of money.")
    assert "punchy" in (sw._cadence_violation(script) or "")
    assert sw._repair_cadence(script) == script


def test_the_repair_never_touches_the_hook_or_the_cta():
    for lengths in ([6, 6, 6, 6, 6, 6], [5, 5, 5, 5, 5, 5, 5]):
        script = _body(lengths)
        out = sw._repair_cadence(script)
        assert out.split("\n")[0] == script.split("\n")[0]
        assert out.split("\n")[-1] == script.split("\n")[-1]
