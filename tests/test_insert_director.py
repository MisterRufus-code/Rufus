"""Which words get a picture, and when.

The format: a picture per NOUN rather than per sentence — the narrator says
"palace" and a palace pops in on the beat. The hard part of that style is
knowing the exact second the word is spoken, and this pipeline already has it,
because remotion_renderer transcribes the finished voiceover with word-level
timestamps. This module is the other half: which words deserve a picture.

Every rejection rule below was added because a REAL script leaked something
undrawable past the previous version. The docstrings name what leaked, because
the rules look arbitrary otherwise and the next person will delete them.

No GPU, no ComfyUI, no network — this module plans, it never renders.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import insert_director as ins  # noqa: E402


def _words(text: str, step: float = 0.32) -> list[dict]:
    """Synthetic word timings in the shape Whisper produces."""
    import re
    return [{"text": w.upper(), "start": i * step, "end": i * step + 0.3}
            for i, w in enumerate(re.findall(r"[A-Za-z]+", text))]


_MEDIEVAL = ("In 1523 Jakob Fugger sent the emperor a letter demanding "
             "repayment. The family controlled the copper mines and a fleet "
             "of ships. Their palace held a vault of gold coins, a crown, and "
             "a sword taken in war.")

_BANGKOK = ("On July 2, 1997, traders in Bangkok rushed to exchange the Thai "
            "baht as the government announced it would float the currency. "
            "The baht had been pegged to the dollar, but a lack of foreign "
            "reserves forced the treasury to abandon this peg.")


# ── what gets drawn ──────────────────────────────────────────────────────────

def test_the_obviously_drawable_nouns_are_chosen():
    got = ins.insert_words(_MEDIEVAL)
    for want in ("emperor", "letter", "copper", "fleet", "ships", "palace",
                 "vault", "gold", "crown", "sword"):
        assert want in got, want


def test_a_place_name_is_not_an_object():
    """The first version lowercased the script before extracting, which
    defeated _content_words' proper-noun rule and chose "philadelphia",
    "bangkok" and "thai". A picture of Philadelphia is a picture of whatever
    the model free-associates."""
    got = ins.insert_words(_BANGKOK)
    for proper in ("bangkok", "thai", "jakob", "fugger"):
        assert proper not in got, proper


def test_a_verb_after_to_or_would_is_not_a_thing():
    """"rushed to exchange", "would float", "to abandon" — all three were
    chosen as things to draw before the verb-cue rule existed."""
    got = ins.insert_words(_BANGKOK)
    for verb in ("exchange", "float", "abandon"):
        assert verb not in got, verb


@pytest.mark.parametrize("word", ["worthless", "outside", "february", "runs"])
def test_the_four_classes_that_leaked_on_the_1893_script(word):
    """Verbatim from the first run of this planner: it chose february,
    outside, worthless and runs — a date, a relationship, a quality and a
    verb. None of them has a silhouette."""
    assert not ins._is_drawable(word)


@pytest.mark.parametrize("word", ["lack", "capital", "jobs", "foreign"])
def test_short_ordinary_abstractions_are_rejected(word):
    """These survive every morphological rule because they are short and
    common: "a lack of foreign reserves", "capital flight", "their jobs"."""
    assert not ins._is_drawable(word)


@pytest.mark.parametrize("word", ["palace", "sword", "crown", "coin", "ship",
                                  "bread", "lantern", "vault"])
def test_real_objects_survive_every_rule(word):
    """The rejection rules are aggressive; this is the guard that stops the
    next one from emptying the layer entirely."""
    assert ins._is_drawable(word)


# ── when it lands ────────────────────────────────────────────────────────────

def test_each_insert_lands_on_the_word_being_spoken():
    plan = ins.plan(_MEDIEVAL, _words(_MEDIEVAL))
    spoken = {w["text"].lower(): w["start"] for w in _words(_MEDIEVAL)}
    for item in plan:
        assert item["at"] == pytest.approx(spoken[item["word"]], abs=0.01)


def test_a_noun_the_narration_never_says_gets_no_picture():
    """Whisper transcribes the FINISHED audio, so a word the TTS swallowed has
    no timestamp. Planning a picture for it would put it at zero."""
    plan = ins.plan("The palace held a sword.", _words("The palace held a"))
    assert [p["word"] for p in plan] == ["palace"]


def test_two_inserts_never_collide():
    """Under half a second apart they read as one flicker and the viewer
    registers neither, so the second is dropped rather than squeezed."""
    plan = ins.plan(_MEDIEVAL, _words(_MEDIEVAL, step=0.05))
    times = [p["at"] for p in plan]
    assert all(b - a >= ins.DEFAULT_GAP for a, b in zip(times, times[1:]))


def test_the_plan_is_in_spoken_order():
    plan = ins.plan(_MEDIEVAL, _words(_MEDIEVAL))
    assert [p["at"] for p in plan] == sorted(p["at"] for p in plan)


def test_only_the_first_utterance_of_a_word_is_used():
    """"baht" twice in one script is one idea, not two — the second pop would
    land on a picture the viewer already has."""
    plan = ins.plan(_BANGKOK, _words(_BANGKOK))
    assert len([p for p in plan if p["word"] == "baht"]) <= 1


# ── the knobs ────────────────────────────────────────────────────────────────

def test_the_layer_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RUFUS_INSERTS", "0")
    assert ins.plan(_MEDIEVAL, _words(_MEDIEVAL)) == []


def test_the_count_is_capped(monkeypatch):
    monkeypatch.setenv("RUFUS_INSERT_MAX", "3")
    assert len(ins.plan(_MEDIEVAL, _words(_MEDIEVAL))) == 3


def test_a_junk_setting_falls_back_loudly(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_INSERT_GAP", "soon")
    ins.plan(_MEDIEVAL, _words(_MEDIEVAL))
    assert "not a number" in capsys.readouterr().out


# ── the shape the rest of the pipeline consumes ──────────────────────────────

def test_the_prompt_asks_for_one_object_and_no_text():
    """An insert is on screen under a second at a fraction of frame size, so
    what has to read is the silhouette — and a model that writes letters into
    a 0.7s insert produces garbage nobody can read anyway."""
    p = ins.insert_prompt("sword")
    assert "single sword" in p
    assert "no text" in p


def test_the_channel_style_is_carried_into_every_insert():
    """An insert in a different style from the beat behind it reads as a bug,
    which is exactly what text-to-video did to the flat-vector look."""
    p = ins.insert_prompt("sword", "Flat 2D vector illustration style")
    assert "Flat 2D vector illustration style" in p


def test_sfx_events_match_the_mixer_signature():
    """audio_gen's ffmpeg path already delays (time, gain) events into place,
    so a pop per insert needs no new mixing code — only the list."""
    plan = ins.plan(_MEDIEVAL, _words(_MEDIEVAL))
    events = ins.sfx_events(plan, 0.3)
    assert events and all(isinstance(t, float) and g == 0.3 for t, g in events)
    assert [t for t, _ in events] == [p["at"] for p in plan]


def test_empty_and_junk_inputs_are_safe():
    assert ins.plan("", []) == []
    assert ins.plan("The palace.", []) == []
    assert ins.insert_words("") == []


def test_describe_names_the_words_so_a_bad_plan_is_obvious():
    line = ins.describe(ins.plan(_MEDIEVAL, _words(_MEDIEVAL)))
    assert "palace" in line and "inserts]" in line


def test_describe_handles_an_empty_plan():
    assert "none planned" in ins.describe([])


# ── density, and what actually limits it ─────────────────────────────────────

@pytest.mark.parametrize("word", ["seventy", "percent", "million", "keep",
                                  "commercial", "universal", "financial"])
def test_the_classes_that_leaked_on_the_investment_banking_script(word):
    """That run planned office, brokers, keep, commissions, seventy, revenue,
    banks, commercial, universal, lines. A number, a bare verb and two
    adjectives — "brokers earned their keep", "seventy percent"."""
    assert not ins._is_drawable(word)


@pytest.mark.parametrize("word", ["metal", "signal", "canal", "medal"])
def test_the_al_ending_is_not_treated_as_adjectival(word):
    """"ial"/"ional" are safe to reject; bare "al" is not, and these four are
    all things a picture can be."""
    assert ins._is_drawable(word)


def test_a_noun_can_be_shown_again_when_it_is_said_again(monkeypatch):
    """One picture per noun caps a 40-second video at however many distinct
    drawable nouns the script happens to hold — measured at seven out of
    fifty-one words. Showing the same coin again when the narration says "coin"
    again is the through-line, not a duplicate."""
    script = "The coin fell. Later the coin rose. At last the coin vanished."
    words = _words(script)
    monkeypatch.setenv("RUFUS_INSERT_GAP", "0.1")

    monkeypatch.setenv("RUFUS_INSERT_REPEAT", "1")
    once = [p for p in ins.plan(script, words) if p["word"] == "coin"]
    monkeypatch.setenv("RUFUS_INSERT_REPEAT", "3")
    thrice = [p for p in ins.plan(script, words) if p["word"] == "coin"]

    assert len(once) == 1
    assert len(thrice) > len(once)


def test_repeats_still_respect_the_spacing_rule(monkeypatch):
    monkeypatch.setenv("RUFUS_INSERT_REPEAT", "5")
    plan = ins.plan(_MEDIEVAL, _words(_MEDIEVAL, step=0.05))
    times = [p["at"] for p in plan]
    assert all(b - a >= ins.DEFAULT_GAP for a, b in zip(times, times[1:]))


def test_the_docstring_says_where_density_really_comes_from():
    """RUFUS_INSERT_MAX is not the cap in normal use and someone will raise it
    expecting more pictures. The module has to say so."""
    assert "RUFUS_FRAMES_PER_BEAT" in ins.__doc__
    assert "NOT the cap" in ins.__doc__
