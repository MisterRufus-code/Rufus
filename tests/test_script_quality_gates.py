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
