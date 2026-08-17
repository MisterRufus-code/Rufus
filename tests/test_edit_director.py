"""The edit director: how each beat MOVES, decided from the script.

Why it exists: every Rufus Short was edited identically. Short.tsx picked its
camera move with `KB_PATTERNS[index % KB_PATTERNS.length]` — a fixed cycle of
six, in the same order, for every video ever rendered. A Short about
hyperinflation and one about the Bank of France got the same push-in on beat 1
and the same pull-back on beat 2. The images changed; the edit never did.

Timing is deliberately NOT the director's to set: cut points come from the
voiceover's own word timestamps, and a model stretching a beat would desync
picture from speech. It controls what happens INSIDE the slot it is given.

Everything here is about the fail-open contract. An edit plan improves a
working render; it must never be able to prevent one.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import edit_director  # noqa: E402


def _plan(n=4, motion="push_in", **overrides):
    plan = {"peak_beat": 2,
            "beats": [{"n": i + 1, "motion": motion, "intensity": "normal",
                       "emphasis": []} for i in range(n)]}
    plan.update(overrides)
    return plan


# ── Validation: a half-applied plan is worse than none ───────────────────────

def test_a_good_plan_survives_cleaning():
    out = edit_director._clean(_plan(3), 3)
    assert out is not None
    assert [b["n"] for b in out["beats"]] == [1, 2, 3]
    assert out["peak_beat"] == 2


def test_the_wrong_number_of_beats_is_rejected():
    """Beat i must line up with clip i — a plan of the wrong length would
    direct the wrong pictures."""
    assert edit_director._clean(_plan(3), 5) is None
    assert edit_director._clean(_plan(7), 5) is None


def test_an_unknown_motion_rejects_the_whole_plan():
    """Short.tsx would fall back to the cycle for that ONE beat, which looks
    like the director working on some beats and not others — harder to
    diagnose than it plainly not running."""
    assert edit_director._clean(_plan(3, motion="zoom_bounce"), 3) is None


def test_every_motion_the_director_may_pick_is_accepted():
    for motion in edit_director.MOTIONS:
        assert edit_director._clean(_plan(2, motion=motion), 2) is not None


def test_a_bad_intensity_falls_back_rather_than_failing():
    """Unlike motion, intensity has a safe default — losing a whole plan over
    it would be a poor trade."""
    plan = _plan(2)
    plan["beats"][0]["intensity"] = "ludicrous"
    out = edit_director._clean(plan, 2)
    assert out["beats"][0]["intensity"] == "normal"


def test_emphasis_is_capped_and_cleaned():
    plan = _plan(1)
    plan["beats"][0]["emphasis"] = ["  four  ", "", "trillion", "marks", "a", "b"]
    out = edit_director._clean(plan, 1)
    assert out["beats"][0]["emphasis"] == ["four", "trillion", "marks", "a"]


def test_a_missing_or_silly_peak_is_replaced_not_rejected():
    for bad in (None, 0, 99, "middle"):
        out = edit_director._clean(_plan(6, peak_beat=bad), 6)
        assert out is not None and 1 <= out["peak_beat"] <= 6


def test_junk_is_rejected():
    for junk in (None, [], "PASS", {}, {"beats": "nope"}):
        assert edit_director._clean(junk, 3) is None


# ── The instruction ──────────────────────────────────────────────────────────

def test_prompt_lists_only_motions_the_renderer_can_perform():
    """A name Short.tsx doesn't know renders as the mechanical cycle — the
    feature silently doing nothing."""
    p = edit_director._prompt(["a", "b"])
    for motion in edit_director.MOTIONS:
        assert motion in p


def test_prompt_teaches_that_stillness_is_a_choice():
    """The whole point. The old cycle moved on every beat, so nothing was
    emphasised; a number lands hardest on a frame that does not move."""
    p = edit_director._prompt(["a", "b"]).lower()
    assert "hold_still is the strongest move" in p
    assert "if every beat moves, nothing does" in p


def test_prompt_states_that_timing_is_not_the_directors_to_set():
    p = edit_director._prompt(["a"]).lower()
    assert "not setting timing" in p


def test_prompt_numbers_the_beats_and_asks_for_that_many():
    p = edit_director._prompt(["first line", "second line", "third line"])
    assert "1. first line" in p and "3. third line" in p
    assert "Exactly 3 entries" in p


# ── Fail-open ────────────────────────────────────────────────────────────────

def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RUFUS_EDIT_DIRECTOR", "0")
    assert edit_director.direct(["a", "b"]) is None


def test_no_beats_means_no_plan():
    assert edit_director.direct([]) is None


def test_a_missing_key_is_survivable(monkeypatch, tmp_path):
    monkeypatch.setattr(edit_director, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_EDIT_DIRECTOR", raising=False)
    assert edit_director.direct(["a", "b"]) is None


def test_an_api_failure_is_survivable(monkeypatch, tmp_path):
    (tmp_path / "keys.json").write_text(json.dumps({"openai": "sk-real"}))
    monkeypatch.setattr(edit_director, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_EDIT_DIRECTOR", raising=False)

    import openai
    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("network down")
    monkeypatch.setattr(openai, "OpenAI", Boom)
    assert edit_director.direct(["a", "b"]) is None


def test_a_valid_reply_comes_back_directed(monkeypatch, tmp_path):
    (tmp_path / "keys.json").write_text(json.dumps({"openai": "sk-real"}))
    monkeypatch.setattr(edit_director, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_EDIT_DIRECTOR", raising=False)

    body = json.dumps({"peak_beat": 2, "beats": [
        {"n": 1, "motion": "push_in",    "intensity": "strong", "emphasis": ["4.2 trillion"]},
        {"n": 2, "motion": "hold_still", "intensity": "normal", "emphasis": []},
        {"n": 3, "motion": "pull_back",  "intensity": "subtle", "emphasis": []},
    ]})

    class Resp:
        class C:
            class M:
                content = body
            message = M()
        choices = [C()]

    class Client:
        def __init__(self, **kw):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return Resp()

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: Client())
    out = edit_director.direct(["one", "two", "three"])
    assert [b["motion"] for b in out["beats"]] == ["push_in", "hold_still", "pull_back"]
    assert out["beats"][0]["emphasis"] == ["4.2 trillion"]


# ── The renderer honours it ──────────────────────────────────────────────────

def test_short_tsx_prefers_the_plan_over_the_index_cycle():
    src = (Path(__file__).parent.parent / "remotion" / "src" / "Short.tsx").read_text()
    assert "MOTION_PATTERNS[direction.motion]" in src
    assert "?? KB_PATTERNS[index % KB_PATTERNS.length]" in src, \
        "the mechanical cycle must remain the FALLBACK, not be deleted"


def test_short_tsx_knows_every_motion_the_director_can_send():
    src = (Path(__file__).parent.parent / "remotion" / "src" / "Short.tsx").read_text()
    for motion in edit_director.MOTIONS:
        assert f"{motion}:" in src, f"Short.tsx cannot perform {motion}"


def test_renderer_passes_the_plan_in_props():
    src = (Path(__file__).parent.parent / "scripts" / "remotion_renderer.py").read_text()
    assert '"edit":' in src
    assert "edit_director.direct(beats)" in src


# ── a hundred and fifty beats do not fit in one reply ────────────────────────

class _FakeClient:
    """Records every call and answers each batch with a valid plan."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Completions:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                asked = kw["messages"][0]["content"]
                # The prompt says "Exactly N entries" — answer with N.
                n = int(asked.rsplit("Exactly ", 1)[1].split(" ")[0])
                body = json.dumps({
                    "peak_beat": 2,
                    "beats": [{"n": i + 1, "motion": "hold_still",
                               "intensity": "normal", "tone": "neutral",
                               "emphasis": []} for i in range(n)],
                })
                msg = type("M", (), {"content": body})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions})()


def test_the_token_budget_grows_with_the_beat_count():
    """700 was the flat cap, and 700 tokens is about twenty entries. At
    fourteen beats it was comfortable; at a hundred and fifty it is a reply cut
    off mid-array, which fails the length check and throws the whole plan
    away — leaving a nine-minute video on the mechanical cycle with every beat
    graded neutral."""
    assert edit_director._budget(14) < edit_director._budget(150)
    assert edit_director._budget(24) > 700


def test_a_short_still_asks_once(monkeypatch):
    client = _FakeClient()
    plan = edit_director._ask(client, ["a", "b"], 0, 2, None)
    assert plan is not None and len(client.calls) == 1


def test_a_long_run_is_asked_for_in_batches():
    client = _FakeClient()
    beats = [f"beat {i}" for i in range(150)]
    merged = []
    peak = None
    for start in range(0, len(beats), edit_director.CHUNK_BEATS):
        batch = beats[start:start + edit_director.CHUNK_BEATS]
        got = edit_director._ask(client, batch, start, len(beats), peak)
        assert got is not None, start
        peak = peak or got["peak_beat"] + start
        merged.extend({**b, "n": start + i + 1} for i, b in enumerate(got["beats"]))
    assert len(merged) == 150
    assert [b["n"] for b in merged] == list(range(1, 151))


def test_a_batch_in_the_middle_knows_it_is_the_middle():
    """An editor handed beats 49-72 with no idea where they sit opens every
    batch like an opening and closes every batch like an ending."""
    p = edit_director._prompt(["x", "y"], offset=48, total=150, peak=75)
    assert "beats 49-50 of 150" in p
    assert "turns at beat 75" in p
    assert "Exactly 2 entries, n from 49 to 50" in p


def test_a_single_batch_run_says_nothing_about_batches():
    p = edit_director._prompt(["x", "y"])
    assert "of 2" not in p
    assert "Exactly 2 entries, n from 1 to 2" in p


def test_the_editor_is_told_which_video_it_is_cutting(monkeypatch):
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    assert "nine-minute" in edit_director._prompt(["x"])
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    assert "40-second vertical" in edit_director._prompt(["x"])


# ── the field it asked for and threw away ────────────────────────────────────

def test_the_emphasis_words_reach_the_captions():
    """The brief names this field exactly — "0-3 words per beat that the
    CAPTION should hit hardest, the figure, the name, the reversal" — and
    nothing read it back. Every run paid for the judgement and then coloured
    its captions from a regex that knows about digits."""
    import audio_gen
    plan = {"peak_beat": 1, "beats": [
        {"n": 1, "motion": "hold_still", "intensity": "normal",
         "tone": "revelation", "emphasis": ["vanished", "overnight"]},
        {"n": 2, "motion": "push_in", "intensity": "normal",
         "tone": "neutral", "emphasis": []},
    ]}
    words = audio_gen.emphasis_words(plan, 110)
    assert words == {"VANISHED", "OVERNIGHT"}
    assert audio_gen._is_highlight("VANISHED", words)
    assert audio_gen._is_highlight("their jobs vanished", words), "phrase caption"
    assert not audio_gen._is_highlight("QUIETLY", words)


def test_a_plan_that_marks_everything_marks_nothing(capsys):
    """The director's own brief says a beat where every word is emphasised has
    no emphasis. A model asked for up to four a beat will sometimes return four
    every time, and 96 accented words out of 110 is a green video."""
    import audio_gen
    # Distinct ALPHABETIC words: the tokeniser strips digits, so "word1a" and
    # "word2a" would collapse into one and the fixture would be testing
    # nothing.
    letters = "abcdefghijklmnopqrstuvwx"
    plan = {"peak_beat": 1, "beats": [
        {"n": i + 1, "motion": "push_in", "intensity": "normal",
         "tone": "neutral",
         "emphasis": [f"{c}{suffix}" for suffix in ("one", "two", "three", "four")]}
        for i, c in enumerate(letters)
    ]}
    assert audio_gen.emphasis_words(plan, 110) == set()
    assert "that is a colour, not an accent" in capsys.readouterr().out


def test_no_plan_leaves_the_captions_exactly_as_they_were():
    import audio_gen
    assert audio_gen.emphasis_words(None, 110) == set()
    assert audio_gen._is_highlight("ORDINARY", set()) is False
    assert audio_gen._is_highlight("$4 BILLION", set()) is True


def test_both_renderers_accent_the_same_words():
    """Captions accented on one renderer and not the other is a difference
    that only shows up to whoever watches both."""
    root = Path(__file__).parent.parent
    py = (root / "scripts" / "remotion_renderer.py").read_text(encoding="utf-8")
    tsx = (root / "remotion" / "src" / "Short.tsx").read_text(encoding="utf-8")
    assert "audio_gen.emphasis_words(" in py, "the share guard runs for both"
    assert "emphasis={emphasis}" in tsx
