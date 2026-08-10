"""Tests for the script-quality gates, driven by REAL rejections from the
live /failures log rather than invented examples.

The rejection data showed three distinct problems, and each has a section
here: a false-positive gate that rejected correct scripts, a retry loop that
learned one rule per wasted generation, and a hook prompt that asked for
grounded numbers without ever saying which numbers existed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer as sw


# ── _repeated_number: comma-grouped figures ──────────────────────────────────
# Live bug: r"\b\d{3,}\b" split "10,000,000" into three "000" tokens, so a
# script naming that figure ONCE was rejected for repeating it 3x. Three
# consecutive generations were burned on it (2026-07-31 10:43).

def test_comma_grouped_number_is_not_a_repeat():
    script = ("What happens when inflation hits 10,000,000%? In 2018, Venezuelan "
              "inflation reached an unimaginable level.")
    assert sw._repeated_number(script) is None


def test_several_distinct_comma_grouped_numbers_are_not_repeats():
    script = "Inflation hit 1,700,000% while debt reached $10,500,000 in 2018."
    assert sw._repeated_number(script) is None


def test_a_genuine_repeat_is_still_caught():
    """The check must keep doing its actual job — this is the '1873 three
    times' pattern it was written for."""
    script = "In 1873 Congress acted. By 1873 silver was gone. Again in 1873."
    result = sw._repeated_number(script)
    assert result is not None and "1873" in result


def test_a_genuine_repeat_of_a_comma_grouped_number_is_caught():
    script = "Losses hit 1,700,000 dollars. Later that year, 1,700,000 again."
    result = sw._repeated_number(script)
    assert result is not None and "1700000" in result.replace(",", "")


def test_small_numbers_may_repeat():
    """Two- and one-digit numbers recur legitimately ('3 banks', '3 years')
    and were never the padding pattern this gate targets."""
    assert sw._repeated_number("3 banks failed. 3 years later, 3 more did.") is None


def test_distinct_large_figures_pass():
    script = "Debt hit $10.5 trillion in 2008 after 8.7 million jobs vanished."
    assert sw._repeated_number(script) is None


# ── _body_violations: report everything at once ──────────────────────────────
# Live pattern: one script rejected 7 times in a row, each attempt told about
# only the single rule it had just tripped (2026-08-02 01:15).

def _clean_script() -> str:
    """A script that passes every body gate — the control for these tests.

    Worth noting how tight the target is: _cadence_violation demands a
    sentence of ≥15 words, while max_avg_sentence_words is 14, so a passing
    script must pay for its one long sentence with several short ones.
    """
    return (
        "The first world currency was a local coin.\n"
        "In 1497 Spain minted the Spanish dollar, and within decades merchants "
        "from Manila to Antwerp priced their goods in it.\n"
        "Then the mines ran dry.\n"
        "Spain kept spending against silver nobody had dug yet.\n"
        "Debts came due in Genoa, Antwerp and Naples on the same winter.\n"
        "The crown defaulted four times in sixty years.\n"
        "The worst part came later.\n"
        "The coin that ruled global trade became the instrument of its own collapse.\n"
        "Why did the world's first currency start as a local coin?\n"
        "Follow for more."
    )


def test_a_clean_script_reports_no_violations():
    assert sw._body_violations(_clean_script()) == []
    assert sw._body_pre_check(_clean_script()) is None


def test_multiple_violations_are_all_reported():
    """The whole point: one generation should surface every problem."""
    bad = "Money moved. Banks failed. Trade died. Prices rose. Coins fell."
    violations = sw._body_violations(bad)
    assert len(violations) > 1, f"only got {violations}"


def test_pre_check_headline_matches_first_violation():
    """_body_pre_check stays the single string logged per attempt and shown in
    the dashboard's rejection table, so it must agree with the full list."""
    bad = "Money moved. Banks failed. Trade died. Prices rose. Coins fell."
    assert sw._body_pre_check(bad) == sw._body_violations(bad)[0]


def test_word_count_and_structure_violations_coexist():
    """A too-short script also missing its loop echo must report BOTH, not
    just the length — reporting length alone is what produced the ladders."""
    short = "Silver vanished. Gold arrived. Nobody noticed the swap."
    violations = sw._body_violations(short)
    assert any("too short" in v for v in violations)
    assert any("loop no echo" in v for v in violations)


def test_violations_are_unique_strings():
    bad = "A. B. C. D. E."
    violations = sw._body_violations(bad)
    assert len(violations) == len(set(violations))


# ── Every violation maps to an actionable correction ─────────────────────────
# accumulated_fixes is only useful if _fix_for_rejection understands the
# strings _body_violations actually emits — a violation with no mapping is a
# rejection the model is never told how to fix.

@pytest.mark.parametrize("script", [
    "Money moved. Banks failed. Trade died. Prices rose. Coins fell.",
    "Silver vanished. Gold arrived. Nobody noticed.",
    "A. B. C. D. E. F. G.",
])
def test_every_violation_produces_a_correction(script):
    std = sw._standards()
    for violation in sw._body_violations(script):
        fix = sw._fix_for_rejection(violation, std, "hook,tokens", "worst, best")
        assert fix, f"no correction mapped for violation: {violation!r}"


def test_real_logged_rejections_all_map_to_corrections():
    """Verbatim rejection strings from the live /failures table."""
    std = sw._standards()
    logged = [
        "loop no echo (second-to-last line shares no content tokens with hook)",
        "cadence: missing a longer, flowing sentence (≥15 words) — every sentence is a similar length",
        "cadence: missing a short, punchy sentence (≤6 words) — every sentence is a similar length",
        "no opinion word (need ≥1 from opinion_pool)",
        "sentences too short (avg 4.6 words, floor 5.5)",
        "too long (117 words, cap 115)",
        "too short (79 words, need ≥80)",
        "banned phrase: 'landscape'",
        "hedging word: 'could be'",
        "em-dash overuse (3 in the script — vary punctuation)",
        "number '2008' repeated 2x — restate with a NEW specific each time, not the same figure",
    ]
    for rejection in logged:
        assert sw._fix_for_rejection(rejection, std, "hook,tokens", "worst, best"), \
            f"no correction for logged rejection: {rejection!r}"


# ── _allowed_numbers_block: stop the hook factory inventing figures ──────────
# Live: 21.4% of all rejections were accuracy, dominated by invented numbers.
# One Ibn Battuta run alone invented 1331, 1352, 1354 and 1355.

def test_allowed_numbers_lists_the_source_figures():
    block = sw._allowed_numbers_block("Ibn Battuta crossed in 1353 with 12,000 camels.")
    assert "1353" in block and "12,000" in block


def test_allowed_numbers_merges_multiple_sources():
    block = sw._allowed_numbers_block("Minted in 1497.", "Weighed 25.56 grams.")
    assert "1497" in block and "25.56" in block


def test_allowed_numbers_deduplicates():
    block = sw._allowed_numbers_block("In 1873. Again 1873.", "And 1873.")
    assert block.count("1873") == 1


def test_allowed_numbers_states_the_rule_explicitly():
    block = sw._allowed_numbers_block("Minted in 1497.")
    assert "rejection" in block.lower()


def test_allowed_numbers_handles_a_source_with_no_figures():
    """An empty list would read as 'no constraint' — the opposite of intent."""
    block = sw._allowed_numbers_block("The Templars grew powerful and were disbanded.")
    assert "NO numbers" in block
    assert "1" not in block.replace("- The source", "")   # no stray digits offered


def test_allowed_numbers_ignores_pre_analysis_list_markers():
    """The pre-analysis arrives as a numbered list (1. … 8. …); those markers
    are formatting, and treating them as source figures is what previously let
    a fabricated '$1 billion' hook pass the grounding check."""
    analysis = "1. CONTRADICTION: banks failed\n2. HOOK ANGLE: silver\n3. STAKES: high"
    block = sw._allowed_numbers_block(analysis)
    assert "NO numbers" in block


def test_prompt_list_agrees_with_the_grounding_check():
    """The listed numbers must be exactly the ones the gate accepts — if the
    prompt offers a token the check then rejects, the model is being set up
    to fail."""
    source = "In 1497 Spain minted a coin weighing 25.56 grams, worth $1,200 today."
    block = sw._allowed_numbers_block(source)
    for token in ("1497", "25.56", "1,200"):
        assert token in block
        # A hook using a listed number must survive the grounding check.
        assert sw._hook_grounding_check(f"The coin of {token} changed trade.", source) is None


def test_grounding_check_still_rejects_an_unlisted_number():
    source = "Ibn Battuta crossed the Sahara in 1353."
    assert sw._hook_grounding_check("In 1355, a caravan struggled.", source) is not None


# ── Ban-list evasion ─────────────────────────────────────────────────────────
# A shipped Monte dei Paschi script opened with "Picture the Medici family
# counting coins" and pivoted on "The truth?" — both a hair outside the exact
# phrases 'picture this' and 'the truth is', both through every gate.

@pytest.mark.parametrize("script,label", [
    ("Picture the Medici family counting coins in Florence.", "picture"),
    ("Picture a trading floor in 1929.", "picture"),
    ("Picture yourself in the queue.", "picture"),
    ("Banks failed. Picture the scene in Florence.", "picture"),
    ("Imagine the queue outside the bank.", "imagine"),
    ("Imagine if the crown had defaulted twice.", "imagine"),
    ("The truth? Ignore history and you misunderstand finance.", "truth"),
    ("The truth: banks never actually die.", "truth"),
    ("Here's the truth about 1929.", "truth"),
    ("Ask yourself why nobody noticed.", "ask yourself"),
])
def test_cliche_families_are_caught_not_just_exact_phrases(script, label):
    assert sw._find_banned(script) is not None, f"{label} evasion slipped through"


@pytest.mark.parametrize("script", [
    "The picture showed a queue outside the bank.",
    "He kept a picture of the trading floor on his desk.",
    "Investors could not imagine a default of that size.",
    "Nobody could picture what came next.",
    "He told the truth about the ledger.",
    "Nobody knew the truth until 1472.",
])
def test_legitimate_uses_of_those_words_still_pass(script):
    """The ban targets the imperative cliché, not the vocabulary — over-blocking
    would push the model away from perfectly good sentences."""
    assert sw._find_banned(script) is None, f"false positive on: {script!r}"


def test_the_exact_shipped_script_would_now_be_caught():
    """Verbatim from a video that actually went out."""
    shipped = (
        "Banking practices date back to 2000 BCE. Banca Monte dei Paschi di Siena "
        "has been operating since 1472. Picture the Medici family counting coins "
        "in Florence, transforming finance. The truth? Ignore history, and you "
        "risk misunderstanding finance today."
    )
    assert sw._find_banned(shipped) is not None


def test_cliche_patterns_are_not_auto_repaired_into_nonsense():
    """A single banned WORD has a synonym swap; a cliché CONSTRUCTION does not,
    and deleting it would gut the sentence. These must go back for a rewrite
    rather than be silently patched."""
    script = "Picture the Medici family counting coins in Florence."
    assert sw._find_banned(sw._repair_banned(script)) is not None


# ── Story architect grounding: catch it before the body, not after ──────────
# Live pattern, verbatim across three consecutive runs: the architect wrote a
# plausible-sounding STAKES GAP/THE TURN, the body writer dramatized it into
# an unsupported claim ("revolutionary", "a political tide hardened the
# transition", "gold coins vanished from circulation"), and only the FINAL
# fact gate caught it — after a whole body-generation cycle had already been
# spent. All three runs exhausted 3 cycles and still shipped at the hard cap.

class _FakeArchitectClient:
    """First call is the architect's own generation; every call whose prompt
    contains 'SCRIPT TO VERIFY' is a _fact_gate check instead. `plan_replies`
    are consumed in order for successive architect generations (retries);
    `verdicts` likewise for successive fact-gate checks."""
    def __init__(self, plan_replies, verdicts):
        self.plan_replies = list(plan_replies)
        self.verdicts = list(verdicts)
        self.calls = []

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _FakeArchitectClient._Msg(content)

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Resp:
        def __init__(self, content):
            self.choices = [_FakeArchitectClient._Choice(content)]
            self.usage = _FakeArchitectClient._Usage()

    @property
    def chat(self):
        outer = self

        class _Chat:
            class completions:
                @staticmethod
                def create(**kw):
                    outer.calls.append(kw)
                    is_check = "SCRIPT TO VERIFY" in kw["messages"][0]["content"]
                    if is_check:
                        verdict = outer.verdicts.pop(0)
                        content = "PASS" if verdict is True else f"FAIL: {verdict}"
                    else:
                        content = outer.plan_replies.pop(0)
                    return _FakeArchitectClient._Resp(content)
        return _Chat()


def test_architect_plan_passing_the_gate_costs_one_generation_and_one_check(monkeypatch):
    import script_writer as sw
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    client = _FakeArchitectClient(
        plan_replies=["SPINE FACT: x\nTHE TURN: y\nSTAKES GAP: z\nWHY NOW: w"],
        verdicts=[True])
    plan, cost = sw._story_architect(client, {"content": "x"}, "analysis",
                                     "hook", "run1", "finance")
    assert "SPINE FACT" in plan
    assert len(client.calls) == 2   # one generation, one check — no wasted retry


def test_ungrounded_plan_is_regenerated_before_reaching_the_body(monkeypatch):
    """The exact live failure: an ungrounded plan must be caught and retried
    HERE, cheaply, rather than shipped to the body writer."""
    import script_writer as sw
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    client = _FakeArchitectClient(
        plan_replies=[
            "SPINE FACT: x\nTHE TURN: a secret cabal orchestrated the transition\n"
            "STAKES GAP: z\nWHY NOW: w",
            "SPINE FACT: x\nTHE TURN: the government took control in 1936\n"
            "STAKES GAP: z\nWHY NOW: w",
        ],
        verdicts=["asserts an unsupported secret motive", True])
    plan, cost = sw._story_architect(client, {"content": "x"}, "analysis",
                                     "hook", "run1", "finance")
    assert "1936" in plan
    assert "secret cabal" not in plan
    assert len(client.calls) == 4   # gen 1, check 1 (fail), gen 2, check 2 (pass)


def test_plan_retry_feeds_the_rejection_reason_back(monkeypatch):
    """The regenerated plan must be told WHY the first one failed, not just
    asked to try again blind."""
    import script_writer as sw
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    client = _FakeArchitectClient(
        plan_replies=["SPINE FACT: bad plan", "SPINE FACT: better plan"],
        verdicts=["invented a secret motive not in the source", True])
    sw._story_architect(client, {"content": "x"}, "analysis", "hook", "run1", "finance")
    second_gen_prompt = client.calls[2]["messages"][0]["content"]
    assert "invented a secret motive not in the source" in second_gen_prompt


def test_plan_exhausting_retries_is_used_anyway_not_blocked(monkeypatch):
    """Fail-open: this is a cost-saving pre-check, not the sole gate — the
    body's own fact check still runs regardless, so a render must never be
    blocked here even if both architect attempts stay ungrounded."""
    import script_writer as sw
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    client = _FakeArchitectClient(
        plan_replies=["SPINE FACT: attempt one", "SPINE FACT: attempt two"],
        verdicts=["still ungrounded", "still ungrounded"])
    plan, cost = sw._story_architect(client, {"content": "x"}, "analysis",
                                     "hook", "run1", "finance")
    assert plan == "SPINE FACT: attempt two"   # the last attempt, not blank
    assert len(client.calls) == 4               # exactly 2 attempts, not unbounded


def test_plan_check_makes_at_most_two_generation_attempts(monkeypatch):
    """The cost-saving property depends on this staying small — an unbounded
    retry here would recreate the exact waste this change removes."""
    import script_writer as sw
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    client = _FakeArchitectClient(
        plan_replies=["SPINE FACT: a", "SPINE FACT: b", "SPINE FACT: c"],
        verdicts=[False, False, False])
    sw._story_architect(client, {"content": "x"}, "analysis", "hook", "run1", "finance")
    generation_calls = [c for c in client.calls
                        if "SCRIPT TO VERIFY" not in c["messages"][0]["content"]]
    assert len(generation_calls) == 2


# ── SOUND: the script is heard, not read ─────────────────────────────────────
# Diagnosed from a live 8/10 script that was correct and dull: two abstract
# nouns ("evolution", "transformation"), ZERO second-person words, and the
# architect's STAKES GAP pasted in as meta-commentary ("Ignoring this would
# mean missing how ancient practices shaped modern currencies"). Prompt-level
# nudges rather than new hard gates — this repo has already been bitten by
# stacking deterministic gates for stylistic preferences.

def _system_prompt():
    """Whitespace-normalised: the prompt is wrapped source, so a phrase can
    straddle a newline and a naive substring check would miss it."""
    import re
    import script_writer
    raw = script_writer._build_system(
        hook="A test hook line", cta="A test CTA", niche_name="money_history",
        niche_cfg={"gpt_system": "Write about money history."})
    return re.sub(r"\s+", " ", raw)


def test_system_prompt_bans_verb_derived_abstract_nouns():
    p = _system_prompt()
    assert "SOUND" in p
    for word in ("transformation", "evolution", "significance"):
        assert word in p, f"the banned-abstract-noun list must name '{word}'"
    assert "Use the" in p and "VERB" in p


def test_system_prompt_requires_talking_to_the_viewer():
    """Every gold example addresses the viewer; the dull live script had no
    'you' anywhere in it."""
    p = _system_prompt()
    low = p.lower()
    assert '"you" or "your"' in low or "use \"you\"" in low
    assert "lecture" in low


def test_system_prompt_forbids_narrating_the_videos_own_purpose():
    """The live script ended on the architect's STAKES GAP verbatim — that is
    commentary about the script, not the script."""
    p = _system_prompt().lower()
    assert "ignoring this would mean missing" in p
    assert "commentary about the script" in p


# ── Fact-gate cap is shown, not hidden ───────────────────────────────────────
# Every external review of runs #48-#51 reported the same "scoring bug": the
# dashboard header said 4/10 while the critic reasoning under it ended
# "TOTAL: 8/10". The cap itself is deliberate and correct — a script the fact
# gate rejected must not present as publishable — but keeping the critic's
# verbatim TOTAL beside it showed a reviewer two different scores for one video
# with no way to tell which one the pipeline acted on.

def test_capped_reasoning_restates_the_total():
    import script_writer
    out = script_writer._restate_total(
        "SPECIFICITY: 2/3 — good.\nHOOK: 2/2 — fine.\nTOTAL: 8/10", 8, 4)
    assert "TOTAL: 4/10" in out
    assert "critic scored 8/10" in out
    assert "capped by the fact gate" in out
    assert "TOTAL: 8/10" not in out


def test_uncapped_reasoning_is_left_verbatim():
    import script_writer
    r = "SPECIFICITY: 3/3.\nTOTAL: 10/10"
    assert script_writer._restate_total(r, 10, 10) == r


def test_restate_survives_reasoning_with_no_total_line():
    """Fail-open: an unparseable critic reply must not lose the reasoning."""
    import script_writer
    r = "The script was fine but unsupported in places."
    assert script_writer._restate_total(r, 8, 4) == r


def test_restate_only_touches_the_total_not_the_criteria():
    import script_writer
    out = script_writer._restate_total(
        "SPECIFICITY: 2/3\nHOOK: 2/2\nCOMPRESSION: 1/2\nTOTAL: 7/10", 7, 4)
    assert "SPECIFICITY: 2/3" in out and "HOOK: 2/2" in out
    assert "COMPRESSION: 1/2" in out


# ── The hook's contradiction must be grounded ────────────────────────────────
# Run #59 (Comstock Lode) cost a whole run — 10 images, TTS and a render — to a
# hook that was never checked. "Henry Comstock didn't discover the Comstock
# Lode" scores perfectly against the scorer's own rubric: proper noun, opposite
# of common belief, under 10 words, high surprise. The source says only that
# the lode was "named after Canadian miner Henry Comstock", so the final fact
# gate rejected it and capped 8/10 → 4/10.
#
# The cause is in the rubric itself: the binary gate REWARDS contradiction and
# never asks whether the contradiction is true. And the hook is the one thing
# that cannot be repaired downstream — the architect is handed it as "HOOK
# (already chosen, will not change)".

def _scorer_prompt(monkeypatch,
                   seed_content="The Comstock Lode is a lode of silver ore "
                                "named after Canadian miner Henry Comstock."):
    """The scorer's real prompt, captured through a stub client.

    The scorer persists every pre-filter rejection, which needs a database
    this test has no business creating — stub the two writers out."""
    import re
    import script_writer

    monkeypatch.setattr(script_writer, "save_attempt", lambda **kw: None)
    monkeypatch.setattr(script_writer, "log_attempt", lambda *a, **kw: None)
    captured = {}

    class Resp:
        class C:
            class M:
                content = '[{"i": 1, "score": 9, "reason": "ok"}]'
            message = M()
        choices = [C()]

        class U:
            prompt_tokens = completion_tokens = 10
        usage = U()

    class Client:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw):
                    captured["prompt"] = kw["messages"][0]["content"]
                    return Resp()
            completions = Completions()
        chat = Chat()

    script_writer._hook_scorer(
        Client(), ["Henry Comstock didn't discover the Comstock Lode.",
                   "In 1859 a silver strike made Nevada rich.",
                   "The Comstock Lode paid for San Francisco."],
        {"content": seed_content, "title": "Comstock Lode", "type": "wikipedia"},
        "money_history", "run-test")
    return re.sub(r"\s+", " ", captured.get("prompt", ""))


def test_scorer_requires_the_contradiction_to_be_supported(monkeypatch):
    p = _scorer_prompt(monkeypatch).lower()
    assert "actually supported by the source" in p
    assert "fails this gate" in p


def test_scorer_gate_count_matches_the_gates_listed(monkeypatch):
    """An off-by-one here silently lets a hook through: "if all three pass"
    beside four bullets tells the model one gate is optional."""
    p = _scorer_prompt(monkeypatch)
    assert "If all four pass" in p
    assert "If all three pass" not in p


def test_scorer_is_shown_enough_source_to_judge_grounding(monkeypatch):
    """The deciding fact in run #59 sat past the old 300-char cut, so the
    scorer was asked a question it had no way to answer."""
    long_seed = ("A" * 400) + " named after Canadian miner Henry Comstock. " + ("B" * 400)
    p = _scorer_prompt(monkeypatch, long_seed)
    assert "named after Canadian miner Henry Comstock" in p


def test_scorer_still_rewards_a_genuine_contradiction(monkeypatch):
    """The fix must not turn the gate into "no contradictions" — the paradox
    hook is the format. Only UNSUPPORTED ones are rejected."""
    p = _scorer_prompt(monkeypatch)
    assert "OPPOSITE of common belief" in p
    assert "SURPRISE INTENSITY" in p


# ── Invented motive is the dominant fact-gate failure ────────────────────────
# Counted across runs #59-#63, five of eight fact-check rejections were an
# attributed MOTIVE rather than a wrong date or figure:
#
#   #59  "Comstock merely took credit"
#   #60  "policymakers were scared to act"
#   #61  "asserts a specific secret motive for the disbandment"
#   #63  "silenced by those who feared inflation more than inequality"
#   #63  "implies a conspiracy or intentional suppression"
#
# It is a structural collision, like the "you"/present-day one: the HUMAN
# criterion pays for opinionated language, the architect is asked for a TURN,
# and the fact gate kills invented intent. The rejection lands AFTER the images
# and the render are paid for, so it is also the most expensive failure in the
# pipeline.

def test_system_prompt_names_motive_as_the_top_rejection_cause():
    p = _system_prompt()
    assert "MOTIVE" in p
    low = p.lower()
    assert "policymakers were scared to act" in low
    assert "sources record what people did" in low


def test_system_prompt_shows_the_outcome_rewrite():
    """A ban with no replacement just produces bland writing — the fix has to
    show the substitution."""
    low = _system_prompt().lower()
    assert "attribute to the outcome" in low
    assert "mind-reading" in low


def test_system_prompt_does_not_ask_for_blandness():
    """The HUMAN criterion still pays for indignation. Only certainty about
    someone's state of mind is out."""
    low = _system_prompt().lower()
    assert "does not mean writing blandly" in low
    assert "indignation about what happened is" in low


def test_architect_is_told_the_turn_cannot_be_a_state_of_mind():
    """THE TURN is where the invented motive originates — the body writer only
    dramatizes what the plan hands it."""
    import inspect
    import script_writer
    src = inspect.getsource(script_writer)
    assert "THE TURN must therefore be an EVENT or an" in src
    assert "#1 REJECTION CAUSE" in src


# ── The fact gate must separate "false" from "not in the excerpt" ────────────
# Every wrong rejection said some version of "unsupported by the source
# material". Rule 1 offered two escape routes — "neither supported by the
# source NOR well-established mainstream history" — and only the first was ever
# used. So true claims were failed for being outside a Wikipedia excerpt: the
# Gold Standard Act of 1900 is real, the Latin Monetary Union really was undone
# by swings in metal value, panic really did hit Paris in 1720.
#
# Rejecting those teaches the writer to quote the excerpt back, which is
# exactly the dry-fact-list failure the channel is trying to avoid.

def _fact_gate_prompt(script="Some script.", seed_content="Some source."):
    import re
    import script_writer

    captured = {}

    class Resp:
        class C:
            class M:
                content = "PASS"
            message = M()
        choices = [C()]

        class U:
            prompt_tokens = completion_tokens = 10
        usage = U()

    class Client:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw):
                    captured["prompt"] = kw["messages"][0]["content"]
                    return Resp()
            completions = Completions()
        chat = Chat()

    script_writer._fact_gate(Client(),
                             {"type": "wikipedia", "content": seed_content,
                              "title": "T", "source": "Wikipedia"}, script)
    return re.sub(r"\s+", " ", captured["prompt"])


def test_gate_states_the_excerpt_is_not_all_of_history():
    p = _fact_gate_prompt()
    assert "NOT THE SUM OF HISTORY" in p
    assert "ABSENT is a PASS" in p


def test_gate_requires_a_category_not_just_a_complaint():
    """"unsupported by the source material" is the phrasing that hid the bug —
    naming the category forces the checker to decide which finding it has."""
    p = _fact_gate_prompt()
    for category in ("CONTRADICTED", "INVENTED", "MIND-READ", "CONSPIRACY", "ABSENT"):
        assert category in p, category
    assert "FAIL: <CATEGORY>" in p


def test_gate_carves_out_cause_and_effect_narration():
    """The exact live rejection: "could not survive the swings" is how history
    is explained, not a factual violation."""
    p = _fact_gate_prompt()
    assert "could not survive the swings" in p
    assert "how history is explained" in p


def test_gate_explicitly_permits_emotional_but_factual_writing():
    p = _fact_gate_prompt()
    assert "still went home hungry" in p
    assert "vivid AND factual" in p


def test_gate_still_fails_the_things_it_should():
    """Loosening must not cost coverage — the real hallucinations are now named
    individually rather than lumped under "unsupported"."""
    p = _fact_gate_prompt()
    assert "70,000 tons of SILVER" in p          # contradicted by the source
    assert "1327" in p                            # invented date
    assert "policymakers were scared to act" in p  # mind-read


# ── Emotion has to come from consequence, not adjectives ─────────────────────

def test_system_prompt_teaches_where_feeling_comes_from():
    p = _system_prompt()
    assert "WHERE THE FEELING ACTUALLY COMES FROM" in p
    low = p.lower()
    assert "still went home hungry" in low
    assert "physical consequence landing on one person" in low


def test_system_prompt_warns_against_reaching_for_adjectives():
    low = _system_prompt().lower()
    assert "devastating" in low and "shocking" in low
    assert "understatement outperforms emphasis" in low


# ── A hook may round a source figure ─────────────────────────────────────────
# A Short is HEARD. "$4,210,500,000,000 for one dollar?" — which shipped in a
# real video — is read out as forty syllables of numerals: unfollowable, and
# the clearest possible signal that a machine wrote it. A person says "four
# point two trillion marks".
#
# The writer could not say that. The grounding check compared number TOKENS, so
# "4.2 trillion" was rejected as "number '4.2' not in source (invented figure)"
# while the unspeakable full form passed. The rule against invented figures was
# mandating unspeakable ones — the same shape of collision as "you"/present-day
# and motive/HUMAN.

_HYPERINFLATION_SOURCE = (
    "By November 1923, one US dollar was worth 4,210,500,000,000 marks. "
    "The national debt was 156 billion marks. 99.3% of notes returned.")


def _grounding(hook, source=_HYPERINFLATION_SOURCE):
    import script_writer
    return script_writer._hook_grounding_check(hook, source)


def test_a_correct_rounding_is_grounded():
    """4.2 trillion IS 4,210,500,000,000 to two significant figures."""
    assert _grounding("4.2 trillion marks for one dollar.") is None
    assert _grounding("4.21 trillion marks for one dollar.") is None


def test_the_verbatim_figure_still_passes():
    assert _grounding("$4,210,500,000,000 for one dollar?") is None


def test_an_invented_magnitude_is_still_rejected():
    """Loosening must not open the door the rule was built to close."""
    assert _grounding("9.9 trillion marks for one dollar.") is not None
    assert _grounding("$50 billion vanished.") is not None
    assert _grounding("Debt hit 900 billion marks.") is not None


def test_a_scale_word_is_read_as_part_of_the_number():
    """Parsed without it, "4.2 trillion" is the number 4.2 — which matches
    nothing in the source and is also not what the hook says."""
    import script_writer
    vals = dict((raw, v) for raw, v, _ in
                script_writer._numeric_values("4.2 trillion and 156 billion"))
    assert vals["4.2"] == 4.2e12
    assert vals["156"] == 156e9


def test_years_and_percentages_are_unaffected():
    assert _grounding("In 1923 the mark died.") is None
    assert _grounding("99.3% of the notes came back.") is None


def test_first_person_check_still_runs_before_numbers():
    """Order matters: a fabricated persona is rejected on its own terms, not
    left to the number check."""
    reason = _grounding("I escaped 4.2 trillion marks of debt.")
    assert reason and "first-person" in reason


# ── The writer is told to speak them ─────────────────────────────────────────

def test_system_prompt_requires_spoken_number_forms():
    p = _system_prompt()
    assert "NUMBERS ARE SPOKEN, NOT PRINTED" in p
    low = p.lower()
    assert "four point two trillion" in low
    assert "one big number per script" in low
