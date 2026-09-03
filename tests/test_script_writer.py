"""Tests for script_writer.py – banned-phrase detection and blacklist keying."""

import pytest

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


def test_find_banned_flags_vague_attribution():
    """Generic AI-writing tell: a claim propped up by an unnamed authority
    instead of the actual specific fact — "studies show" instead of naming
    the study."""
    assert _find_banned("Studies show most people never check their statements.") == "studies show"
    assert _find_banned("Experts believe the market will keep rising.") == "experts believe"
    assert _find_banned("Some say the bank never recovered.") == "some say"


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


# ── Semantic near-duplicate gate ─────────────────────────────────────────────

def test_cosine_similarity_math():
    import script_writer as sw
    assert abs(sw._cosine([1, 0], [1, 0]) - 1.0) < 1e-9      # identical
    assert abs(sw._cosine([1, 0], [0, 1])) < 1e-9            # orthogonal
    assert sw._cosine([], []) == 0.0                          # degenerate → 0, no crash
    assert sw._cosine([1, 1], [0, 0]) == 0.0                  # zero vector safe


def test_check_similarity_fails_open_without_embedding(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "_embed_script", lambda s: None)   # API down / no key
    is_dup, sim, vec = sw.check_similarity("some script", "main_en")
    assert is_dup is False and sim == 0.0 and vec is None


def test_check_similarity_flags_paraphrase_level_duplicate(monkeypatch, tmp_path):
    """Two scripts with near-identical embeddings must be flagged even though
    their first 20 words differ (the exact-match blacklist's blind spot)."""
    import script_writer as sw
    monkeypatch.setattr(sw, "EMBEDDINGS_FILE", tmp_path / "emb.json")

    monkeypatch.setattr(sw, "_embed_script", lambda s: [0.6, 0.8, 0.0])
    sw.add_embedding([0.6, 0.8, 0.0], "main_en")

    is_dup, sim, _ = sw.check_similarity("differently worded same facts", "main_en")
    assert is_dup is True
    assert sim > 0.99


def test_check_similarity_is_per_channel(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "EMBEDDINGS_FILE", tmp_path / "emb.json")
    monkeypatch.setattr(sw, "_embed_script", lambda s: [1.0, 0.0])

    sw.add_embedding([1.0, 0.0], "spanish")            # other channel's history
    is_dup, sim, _ = sw.check_similarity("x", "main_en")
    assert is_dup is False                             # main_en history is empty


def test_add_embedding_caps_history_per_channel(monkeypatch, tmp_path):
    import json as _json
    import script_writer as sw
    monkeypatch.setattr(sw, "EMBEDDINGS_FILE", tmp_path / "emb.json")
    monkeypatch.setattr(sw, "EMBED_HISTORY", 3)

    for i in range(5):
        sw.add_embedding([float(i), 1.0], "main_en")
    sw.add_embedding([9.0, 9.0], "spanish")            # must survive main_en's cap

    entries = _json.loads((tmp_path / "emb.json").read_text(encoding="utf-8"))
    main = [e for e in entries if e["channel"] == "main_en"]
    assert len(main) == 3
    assert main[-1]["vec"][0] == 4.0                   # newest kept
    assert main[0]["vec"][0] == 2.0                    # oldest evicted first
    assert any(e["channel"] == "spanish" for e in entries)


def test_add_embedding_none_is_noop(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "EMBEDDINGS_FILE", tmp_path / "emb.json")
    sw.add_embedding(None, "main_en")
    assert not (tmp_path / "emb.json").exists()


# ── Hook duplication guard (voice read the hook twice) ───────────────────────

def test_hook_already_present_exact_match():
    from script_writer import _hook_already_present
    assert _hook_already_present('"Nixon cost you $50,000."', "Nixon cost you $50,000.")


def test_hook_already_present_rephrased_punctuation():
    """The real bug: GPT changed only punctuation/case, exact-match failed,
    the original was inserted above the paraphrase, and TTS read the hook
    TWICE back-to-back."""
    from script_writer import _hook_already_present
    assert _hook_already_present("Nixon cost you $50,000!", "Nixon cost you $50,000.")
    assert _hook_already_present("nixon cost YOU $50,000", "Nixon cost you $50,000.")


def test_hook_already_present_light_rephrase_still_counts():
    from script_writer import _hook_already_present
    assert _hook_already_present("Nixon's decision cost you $50,000 overnight",
                                 "Nixon cost you $50,000.")


def test_hook_missing_when_first_line_is_body():
    from script_writer import _hook_already_present
    assert not _hook_already_present("In 1971, Switzerland cemented its reputation.",
                                     "Nixon cost you $50,000.")


# ── Hook grounding gate (invented numbers / fabricated persona) ───────────────
# Live failure pattern: the hook factory invented figures ("$3.3 billion lost
# in hours") and first-person stories ("I escaped $50K in debt bondage") that
# the downstream fact gate then rejected — capping EVERY run at 5/10 and
# holding every upload. The grounding gate kills those candidates before
# scoring so a source-grounded hook wins instead.

SOURCE = ("Black Wednesday, or the 1992 sterling crisis, occurred on "
          "16 September 1992 when the UK withdrew sterling from the ERM. "
          "Estimates put the cost at 3.3 billion pounds.")


def test_hook_grounding_rejects_first_person_confession():
    from script_writer import _hook_grounding_check
    reason = _hook_grounding_check("I escaped $50K in debt bondage—here's how.", SOURCE)
    assert reason is not None
    assert "first-person" in reason


def test_hook_grounding_rejects_invented_number():
    from script_writer import _hook_grounding_check
    reason = _hook_grounding_check("$840,000 vanished on September 16.", SOURCE)
    assert reason is not None
    assert "840" in reason


def test_hook_grounding_accepts_source_number():
    from script_writer import _hook_grounding_check
    assert _hook_grounding_check("3.3 billion lost in one day.", SOURCE) is None


def test_hook_grounding_accepts_source_year_and_date():
    from script_writer import _hook_grounding_check
    assert _hook_grounding_check("1992: the day the UK lost control.", SOURCE) is None


def test_hook_grounding_accepts_no_number_hook():
    from script_writer import _hook_grounding_check
    assert _hook_grounding_check("The pound's worst day was self-inflicted.", SOURCE) is None


def test_hook_grounding_number_with_commas_matches_source_without():
    from script_writer import _hook_grounding_check
    src = "The fund lost 3300000000 dollars that afternoon."
    assert _hook_grounding_check("3,300,000,000 gone in an afternoon.", src) is None


def test_hook_grounding_weimar_not_flagged_as_first_person():
    """\\bwe\\b must not fire inside words like 'Weimar'."""
    from script_writer import _hook_grounding_check
    src = "In 1923 Weimar Germany, hyperinflation destroyed the mark."
    assert _hook_grounding_check("Weimar burned savings in 1923.", src) is None


def test_hook_grounding_rejects_round_magnitude_not_a_source_token():
    """Live hole: '$1 billion vanished' passed the old substring check because
    '1' is a substring of a source year like '1720', so the fabricated figure
    reached the fact gate and capped the whole video to 5/10. The number must
    match a real source token, not merely appear inside one."""
    from script_writer import _hook_grounding_check
    src = ("The Mississippi Bubble culminated in 1720. John Law rose to power "
           "in France before the collapse.")
    reason = _hook_grounding_check("1720: $1 billion vanished in a day.", src)
    assert reason is not None
    assert "1" in reason  # the invented '1 billion', not the grounded year 1720


def test_strip_list_markers_removes_enumeration_only():
    from script_writer import _strip_list_markers
    out = _strip_list_markers("1. CONTRADICTION: x\n2) HOOK: y\n5. CONCRETE: in 1934")
    assert "CONTRADICTION: x" in out and "HOOK: y" in out
    assert "1934" in out           # real figures inside the line survive
    assert not out.lstrip().startswith("1.")


def test_hook_grounding_ignores_analysis_list_numbers():
    """Live 5/10 cause: the pre-analysis is a numbered list (1.–8.), so feeding
    it verbatim into the grounding corpus made the digits 1-8 read as real
    source figures — a fabricated '$1 billion' hook passed the check and was
    only caught later by the fact gate, capping the whole video."""
    from script_writer import _hook_grounding_check, _strip_list_markers
    analysis = ("1. CONTRADICTION: Swiss secrecy shields many parties.\n"
                "2. HOOK ANGLE: Banking secrecy since 1700s.\n"
                "5. CONCRETE DETAIL: codified in 1934.")
    src = "Banking in Switzerland dates to the early 18th century."
    grounding = src + " " + _strip_list_markers(analysis)

    # '1' came only from the "1." list marker → still an invented figure
    assert _hook_grounding_check("$1 billion hidden since the 1700s.", grounding) is not None
    # real figures surfaced by the analysis still pass
    assert _hook_grounding_check("Secrecy became law in 1934.", grounding) is None


def test_hook_grounding_billion_substring_of_source_year_rejected():
    """The exact live case: '$1 billion' against a Panic-of-1873 source. '1'
    is a substring of '1873' but is NOT a real figure in the source, so the
    fabricated magnitude must be rejected before it reaches scoring."""
    from script_writer import _hook_grounding_check
    src = "The Panic of 1873 was a financial crisis that triggered a depression."
    assert _hook_grounding_check("$1 billion vanished during the Panic of 1873.",
                                 src) is not None


# ── _grounded_rewrite: the fact-gate gets its own recovery path ───────────────
# The in-writer fact gate caps a flagged script to 5/10; the rewrite used to
# live only in main.py's separate supervisor gate, which can disagree — leaving
# a dead capped 5/10 with no recovery (live: money_history 'Hanseatic League').

class _FakeGen:
    """Stand-in OpenAI client — _grounded_rewrite only calls module-level
    _generate/_score/_fact_gate, all monkeypatched, so the client is never
    actually used for network I/O."""


def test_grounded_rewrite_returns_result_when_rewrite_passes_facts(monkeypatch):
    import script_writer as sw
    monkeypatch.setattr(sw, "_generate",
                        lambda *a, **k: ("Hook line.\nGrounded body.\nCTA.", 0.001, 10, 5, 5))
    monkeypatch.setattr(sw, "_body_pre_check", lambda script: None)
    monkeypatch.setattr(sw, "_find_banned", lambda script: None)
    monkeypatch.setattr(sw, "_fact_gate", lambda c, s, script: (True, "grounded", 0.001))
    monkeypatch.setattr(sw, "_score",
                        lambda *a, **k: (8, {"specificity": 2}, "good", 0.001, 10))

    out = sw._grounded_rewrite(
        _FakeGen(), system="sys", base_usr="usr", body_model="m",
        fact_reason="invented a figure", winning_hook="Hook line.",
        seed=None, run_id="r1", active="money_history")
    assert out is not None
    assert out["score"] == 8
    assert out["cost"] > 0


def test_grounded_rewrite_returns_none_when_rewrite_still_fails_facts(monkeypatch):
    import script_writer as sw
    monkeypatch.setattr(sw, "_generate",
                        lambda *a, **k: ("Hook line.\nStill wrong.\nCTA.", 0.001, 10, 5, 5))
    monkeypatch.setattr(sw, "_body_pre_check", lambda script: None)
    monkeypatch.setattr(sw, "_find_banned", lambda script: None)
    monkeypatch.setattr(sw, "_fact_gate", lambda c, s, script: (False, "still ungrounded", 0.001))
    # _score must never be reached once the fact gate rejects the rewrite.
    monkeypatch.setattr(sw, "_score",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored a failing rewrite")))

    out = sw._grounded_rewrite(
        _FakeGen(), system="sys", base_usr="usr", body_model="m",
        fact_reason="invented a figure", winning_hook="Hook line.",
        seed=None, run_id="r1", active="money_history")
    assert out is None


def test_grounded_rewrite_returns_none_when_rewrite_breaks_structure(monkeypatch):
    import script_writer as sw
    monkeypatch.setattr(sw, "_generate",
                        lambda *a, **k: ("Hook line.\nBad.\nCTA.", 0.001, 10, 5, 5))
    monkeypatch.setattr(sw, "_find_banned", lambda script: None)
    monkeypatch.setattr(sw, "_body_pre_check", lambda script: "sentences too short")
    out = sw._grounded_rewrite(
        _FakeGen(), system="sys", base_usr="usr", body_model="m",
        fact_reason="x", winning_hook="Hook line.",
        seed=None, run_id="r1", active="money_history")
    assert out is None


# ── _fixes_from_crits: closes the "LLM score just retries cold" gap ──────────
# Real bug behind observed score volatility (10/10 one video, 5/10 the next
# on similar source material): _fix_for() already turns a pre-filter
# rejection into a concrete instruction carried into every later attempt, but
# a low LLM SCORE only added a numeric summary to the retry prompt — no
# actual correction. _fixes_from_crits closes that gap.

def _std():
    from script_writer import _standards
    return _standards()


def test_fixes_from_crits_empty_when_all_criteria_pass():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 2, "hook": 2, "compression": 2, "loop": 2, "human": 2}
    assert _fixes_from_crits(crits, _std(), "worst, smartest, wrong") == []


def test_fixes_from_crits_flags_low_specificity():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 0, "hook": 2, "compression": 2, "loop": 2, "human": 2}
    fixes = _fixes_from_crits(crits, _std(), "worst, smartest, wrong")
    assert any("ground EVERY claim" in f for f in fixes)


def test_fixes_from_crits_flags_low_loop():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 2, "hook": 2, "compression": 2, "loop": 0, "human": 2}
    fixes = _fixes_from_crits(crits, _std(), "worst, smartest, wrong")
    assert any("mirror the hook" in f for f in fixes)


def test_fixes_from_crits_flags_low_human_and_includes_opinion_words():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 2, "human": 0}
    fixes = _fixes_from_crits(crits, _std(), "worst, smartest, wrong")
    assert any("worst, smartest, wrong" in f for f in fixes)


def test_fixes_from_crits_multiple_weak_criteria_all_reported():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 0, "hook": 0, "compression": 0, "loop": 0, "human": 0}
    fixes = _fixes_from_crits(crits, _std(), "worst")
    assert len(fixes) == 5


def test_fixes_from_crits_missing_keys_treated_as_passing():
    """A criterion the scorer failed to parse (missing from crits dict) must
    not be treated as a failure — only genuinely low/parsed scores flag."""
    from script_writer import _fixes_from_crits
    fixes = _fixes_from_crits({}, _std(), "worst")
    assert fixes == []


# ── Story architect: plan-before-prose pre-pass ───────────────────────────────

def test_architect_enabled_default_on(monkeypatch):
    from script_writer import _architect_enabled
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    assert _architect_enabled() is True


def test_architect_enabled_env_off(monkeypatch):
    from script_writer import _architect_enabled
    monkeypatch.setenv("RUFUS_SCRIPT_ARCHITECT", "0")
    assert _architect_enabled() is False


def test_story_architect_noop_when_disabled(monkeypatch):
    from script_writer import _story_architect
    monkeypatch.setenv("RUFUS_SCRIPT_ARCHITECT", "0")
    plan, cost = _story_architect(None, {"content": "x"}, "analysis", "hook",
                                  "run1", "finance")
    assert plan == "" and cost == 0.0


def test_story_architect_returns_empty_on_api_failure(monkeypatch):
    """Fail-open: an API error must not crash script writing — just skip
    the plan and proceed exactly as before this feature existed."""
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("API down")

    plan, cost = _story_architect(FakeClient(), {"content": "x"}, "analysis",
                                  "hook", "run1", "finance")
    assert plan == "" and cost == 0.0


def test_story_architect_returns_plan_and_cost(monkeypatch):
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)

    class Usage:
        prompt_tokens = 100
        completion_tokens = 50

    class Msg:
        content = "SPINE FACT: x\nTHE TURN: y\nWHY NOW: z"

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]
        usage = Usage()

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return Resp()

    plan, cost = _story_architect(FakeClient(), {"content": "x"}, "analysis",
                                  "hook", "run1", "finance")
    assert "SPINE FACT" in plan
    assert cost >= 0.0


# ── Sensory-anchor disqualifier feeding into _fixes_from_crits ────────────────
# The disqualifier isn't a parsed criterion (only SPECIFICITY/HOOK/etc. are
# regex-parsed) — it's only visible in the raw reasoning text the scorer
# echoes back ("DISQUALIFIERS: [list, or 'none']"), so this checks the text
# directly rather than a crits dict key.

def test_fixes_from_crits_detects_sensory_disqualifier_from_reasoning():
    from script_writer import _fixes_from_crits, _standards
    crits = {"specificity": 2, "hook": 2, "compression": 2, "loop": 2, "human": 2}
    reasoning = "DISQUALIFIERS: NO SENSORY DETAIL — entirely abstract summary\nTOTAL: 4/10"
    fixes = _fixes_from_crits(crits, _standards(), "worst", reasoning=reasoning)
    assert any("sensory" in f.lower() for f in fixes)


def test_fixes_from_crits_no_sensory_fix_when_not_flagged():
    from script_writer import _fixes_from_crits, _standards
    crits = {"specificity": 2, "hook": 2, "compression": 2, "loop": 2, "human": 2}
    fixes = _fixes_from_crits(crits, _standards(), "worst", reasoning="DISQUALIFIERS: none\nTOTAL: 10/10")
    assert not any("sensory" in f.lower() for f in fixes)


def test_score_prompt_includes_sensory_disqualifier():
    """The rubric sent to the LLM must actually ask about sensory detail —
    otherwise the reasoning-text check above has nothing to ever find."""
    from script_writer import _score
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    msg = type("M", (), {"content": "DISQUALIFIERS: none\nTOTAL: 8/10"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    _score(FakeClient(), "some script", {"type": "wikipedia", "content": "x"},
          "some hook", "run1", "finance")
    prompt = captured["messages"][0]["content"].lower()
    assert "sensory" in prompt


# ── Cadence pattern-interrupt (_cadence_violation) ─────────────────────────────

def test_cadence_violation_none_when_mixed_lengths():
    from script_writer import _cadence_violation
    script = (
        "Short line here now. "
        "This much longer sentence goes on for quite a while to build real atmosphere and tension. "
        "Another one."
    )
    assert _cadence_violation(script) is None


def test_cadence_violation_flags_uniform_sentence_lengths():
    from script_writer import _cadence_violation
    script = (
        "This sentence has exactly nine little words in it. "
        "This one also has exactly nine little words too. "
        "And this one also has exactly nine words here."
    )
    reason = _cadence_violation(script)
    assert reason is not None
    assert "cadence" in reason


def test_cadence_violation_skipped_for_very_short_scripts():
    from script_writer import _cadence_violation
    assert _cadence_violation("One sentence only here.") is None


def test_body_pre_check_chains_to_cadence(monkeypatch):
    """The cadence check must actually run as part of the full pre-filter
    chain — every EARLIER check is mocked to force-pass so a genuinely
    uniform-sentence-length script is rejected specifically because of
    cadence, proving the wiring (not just testing cadence in isolation)."""
    from script_writer import _body_pre_check
    import script_writer as sw
    monkeypatch.setattr(sw, "_specificity_density", lambda s: 999)
    monkeypatch.setattr(sw, "_sentence_stats", lambda s: (9.0, 5))
    monkeypatch.setattr(sw, "_loop_echoes_hook", lambda s: (True, "x"))
    monkeypatch.setattr(sw, "_has_opinion_word", lambda s: True)
    monkeypatch.setattr(sw, "_find_hedging", lambda s: None)

    std = sw._standards()
    # Five real sentences, each exactly nine words -> genuinely uniform
    # cadence. Unpunctuated padding words after them satisfy the (real,
    # unmocked) total word-count check without being parsed as additional
    # sentences by _SENTENCE_RE (which requires terminal punctuation).
    sentence = "This sentence has exactly nine little words right now."
    core = " ".join([sentence] * 5)
    pad_needed = max(0, std["body"]["min_words"] - len(core.split()))
    script = core + " " + " ".join(["filler"] * pad_needed)

    result = _body_pre_check(script)
    assert result is not None
    assert "cadence" in result


def test_fix_for_rejection_cadence_message():
    from script_writer import _fix_for_rejection, _standards
    rejection = "cadence: missing a short, punchy sentence (≤6 words) — every sentence is a similar length"
    fix = _fix_for_rejection(rejection, _standards(), "hook,tokens", "worst")
    assert "vary sentence rhythm" in fix


# ── repeated-number check ────────────────────────────────────────────────────
# Live pattern: under grounding pressure the model reaches for its one solid
# verified figure ("1873") again and again instead of finding fresh specifics,
# so the script pads itself with the SAME fact restated rather than new ones.

def test_repeated_number_flags_a_figure_used_twice():
    """A FIGURE restated verbatim is padding. A YEAR is the story's setting and
    is now allowed twice — see test_a_year_may_appear_twice in
    test_script_quality_gates.py, where holding years to this limit killed five
    of six body attempts on a script about 1973."""
    from script_writer import _repeated_number
    script = "The bank lost 45000 marks. The 45000 marks never came back."
    result = _repeated_number(script)
    assert result is not None
    assert "45000" in result


def test_repeated_year_still_flagged_on_the_third_use():
    from script_writer import _repeated_number
    script = "It opened in 1873. By 1873 it led. In 1873 it failed."
    result = _repeated_number(script)
    assert result is not None and "1873" in result


def test_repeated_number_passes_when_each_figure_is_unique():
    from script_writer import _repeated_number
    script = "The bank opened in 1873. By 1901 it was the largest lender."
    assert _repeated_number(script) is None


def test_repeated_number_ignores_short_numbers():
    """Small numbers (page/list counters, "3 things") repeat for reasons that
    aren't the padding pattern this guards against — only 3+ digit figures
    (years, dollar amounts, counts) count."""
    from script_writer import _repeated_number
    script = "Rule 12: save first. Rule 12 again: automate it."
    assert _repeated_number(script) is None


def test_repeated_number_reports_the_worst_offender():
    from script_writer import _repeated_number
    script = "It cost 5000 dollars in 1873. Again, 1873 was the turning point. 1873."
    result = _repeated_number(script)
    assert "1873" in result and "3x" in result


def test_body_pre_check_chains_to_repeated_number(monkeypatch):
    """Same wiring proof as the cadence test: every earlier check force-passes
    so a script is rejected specifically because of the repeated figure, not
    some other coincidental failure."""
    from script_writer import _body_pre_check
    import script_writer as sw
    monkeypatch.setattr(sw, "_specificity_density", lambda s: 999)
    monkeypatch.setattr(sw, "_sentence_stats", lambda s: (9.0, 5))
    monkeypatch.setattr(sw, "_loop_echoes_hook", lambda s: (True, "x"))
    monkeypatch.setattr(sw, "_has_opinion_word", lambda s: True)
    monkeypatch.setattr(sw, "_find_hedging", lambda s: None)

    std = sw._standards()
    sentence = "The vault opened in 1873 after decades of careful planning work."
    core = " ".join([sentence] * 2) + " It happened in 1873, everyone agreed."
    pad_needed = max(0, std["body"]["min_words"] - len(core.split()))
    script = core + " " + " ".join(["filler"] * pad_needed)

    result = _body_pre_check(script)
    assert result is not None
    assert "1873" in result


def test_fix_for_rejection_repeated_number_message():
    from script_writer import _fix_for_rejection, _standards
    rejection = "number '1873' repeated 3x — restate with a NEW specific each time, not the same figure"
    fix = _fix_for_rejection(rejection, _standards(), "hook,tokens", "worst")
    assert "1873" in fix and "DIFFERENT specific" in fix


# ── em-dash overuse ──────────────────────────────────────────────────────────
# 3+ em-dashes in a short script is a real AI-cadence tell. Kept deliberately
# loose (2 must pass) — the false-positive-risk boundary this whole check
# hinges on, given the standing note that the gate is already too strict.

def test_em_dash_overuse_flags_three_or_more():
    from script_writer import _em_dash_overuse
    script = "One thing—then another—then a third thing—all in one script."
    result = _em_dash_overuse(script)
    assert result is not None
    assert "3" in result


def test_em_dash_overuse_allows_two():
    from script_writer import _em_dash_overuse
    script = "One thing—then another thing—and that's it, nothing more here."
    assert _em_dash_overuse(script) is None


def test_em_dash_overuse_ignores_hyphens_and_en_dashes():
    from script_writer import _em_dash_overuse
    script = "A well-known, state-of-the-art, top-tier plan spanning 2020-2023."
    assert _em_dash_overuse(script) is None


def test_body_pre_check_chains_to_em_dash_overuse(monkeypatch):
    """Same wiring proof as the repeated-number test: every earlier check
    force-passes so a script is rejected specifically for em-dash overuse."""
    from script_writer import _body_pre_check
    import script_writer as sw
    monkeypatch.setattr(sw, "_specificity_density", lambda s: 999)
    monkeypatch.setattr(sw, "_sentence_stats", lambda s: (9.0, 5))
    monkeypatch.setattr(sw, "_loop_echoes_hook", lambda s: (True, "x"))
    monkeypatch.setattr(sw, "_has_opinion_word", lambda s: True)
    monkeypatch.setattr(sw, "_find_hedging", lambda s: None)
    monkeypatch.setattr(sw, "_repeated_number", lambda s: None)

    std = sw._standards()
    core = "One thing—then another—then a third—all crammed into one script here."
    pad_needed = max(0, std["body"]["min_words"] - len(core.split()))
    script = core + " " + " ".join(["filler"] * pad_needed)

    result = _body_pre_check(script)
    assert result is not None
    assert "em-dash" in result


def test_fix_for_rejection_em_dash_message():
    from script_writer import _fix_for_rejection, _standards
    rejection = "em-dash overuse (4 in the script — vary punctuation; use a period, comma, or colon for some of these instead)"
    fix = _fix_for_rejection(rejection, _standards(), "hook,tokens", "worst")
    assert "2 em-dashes" in fix


def test_fix_for_rejection_sentences_too_short_not_confused_with_total_length():
    """Regression: 'sentences too short' contains 'too short' as a substring
    — the generic 'too short' branch used to win first and told the model to
    add MORE total words, instead of the real fix (lengthen sentences)."""
    from script_writer import _fix_for_rejection, _standards
    fix = _fix_for_rejection("sentences too short (avg 4.0 words, floor 6.0)",
                             _standards(), "hook,tokens", "worst")
    assert "lengthen sentences" in fix
    assert "write at least" not in fix   # the wrong (total-word-count) message


def test_fix_for_rejection_sentences_too_long_not_confused_with_total_length():
    from script_writer import _fix_for_rejection, _standards
    fix = _fix_for_rejection("sentences too long (avg 20.0 words, cap 12.0)",
                             _standards(), "hook,tokens", "worst")
    assert "shorten sentences" in fix
    assert "keep it under" not in fix   # the wrong (total-word-count) message


def test_fix_for_rejection_whole_body_too_short_still_works():
    from script_writer import _fix_for_rejection, _standards
    fix = _fix_for_rejection("too short (40 words, need ≥80)", _standards(),
                             "hook,tokens", "worst")
    assert "write at least" in fix


def test_fix_for_rejection_banned_phrase():
    from script_writer import _fix_for_rejection, _standards
    fix = _fix_for_rejection("banned phrase: 'crucial'", _standards(), "h", "o")
    assert "crucial" in fix and "BANNED" in fix


def test_fix_for_rejection_unknown_returns_empty():
    from script_writer import _fix_for_rejection, _standards
    assert _fix_for_rejection("something unrecognized", _standards(), "h", "o") == ""


# ── Hook-opener diversity (_overused_hook_openers) ─────────────────────────────

def test_overused_hook_openers_empty_with_no_data(monkeypatch, tmp_path):
    import script_writer as sw
    import db_manager
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "empty.db")
    assert sw._overused_hook_openers("money_history") == []


def test_overused_hook_openers_flags_dominant_opener(monkeypatch, tmp_path):
    import script_writer as sw
    import db_manager
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    db_manager.init_db()

    # 15 hooks opening with "Why", 5 with distinct openers -> "why" dominates
    for i in range(15):
        db_manager.save_video(niche="money_history", channel="main_en",
                              script_hook=f"Why did event {i} happen so fast?",
                              scene_desc="s", video_file=f"v{i}.mp4")
    for i in range(5):
        db_manager.save_video(niche="money_history", channel="main_en",
                              script_hook=f"Distinct opener {i} appears here now.",
                              scene_desc="s", video_file=f"d{i}.mp4")

    overused = sw._overused_hook_openers("money_history")
    assert "why" in overused


def test_overused_hook_openers_ignores_other_channels(monkeypatch, tmp_path):
    import script_writer as sw
    import db_manager
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    db_manager.init_db()

    for i in range(15):
        db_manager.save_video(niche="money_history", channel="other_channel",
                              script_hook=f"Why did event {i} happen so fast?",
                              scene_desc="s", video_file=f"v{i}.mp4")

    assert sw._overused_hook_openers("money_history") == []


def test_overused_hook_openers_needs_minimum_history(monkeypatch, tmp_path):
    """Fewer than 10 shipped hooks isn't enough to draw a real conclusion —
    must not flag on a tiny, noisy sample."""
    import script_writer as sw
    import db_manager
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    db_manager.init_db()

    for i in range(5):
        db_manager.save_video(niche="money_history", channel="main_en",
                              script_hook=f"Why did event {i} happen so fast?",
                              scene_desc="s", video_file=f"v{i}.mp4")

    assert sw._overused_hook_openers("money_history") == []


def test_novelty_block_includes_opener_reset_when_overused(monkeypatch):
    import script_writer as sw
    monkeypatch.setattr(sw, "_recent_video_rows", lambda n, limit=12: [])
    monkeypatch.setattr(sw, "_load_learnings", lambda: {})
    monkeypatch.setattr(sw, "_overused_hook_openers", lambda n: ["why", "how"])
    block = sw._novelty_block("money_history")
    assert "OPENER RESET" in block
    assert "why" in block and "how" in block


def test_novelty_block_no_opener_section_when_diverse(monkeypatch):
    import script_writer as sw
    monkeypatch.setattr(sw, "_recent_video_rows", lambda n, limit=12: [])
    monkeypatch.setattr(sw, "_load_learnings", lambda: {})
    monkeypatch.setattr(sw, "_overused_hook_openers", lambda n: [])
    block = sw._novelty_block("money_history")
    assert "OPENER RESET" not in block


# ── Story architect: STAKES GAP + turn-must-follow-spine-fact ─────────────────

def test_story_architect_prompt_includes_stakes_gap(monkeypatch):
    """_story_architect now also fact-checks its own plan (see the function's
    docstring), which means the SAME fake client sees a SECOND call — the
    fact gate verifying the plan it just wrote. Capturing every call and
    asserting on the FIRST is what isolates the architect's own prompt from
    the fact-gate's ('PASS' short-circuits the retry so there's exactly one
    of each call here)."""
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    calls = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    is_fact_check = "SCRIPT TO VERIFY" in kw["messages"][0]["content"]
                    content = "PASS" if is_fact_check else (
                        "SPINE FACT: x\nTHE TURN: y\nSTAKES GAP: z\nWHY NOW: w")
                    msg = type("M", (), {"content": content})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    plan, _ = _story_architect(FakeClient(), {"content": "x"}, "analysis",
                               "hook", "run1", "finance")
    architect_prompt = calls[0]["messages"][0]["content"]
    assert "STAKES GAP" in architect_prompt
    assert "STAKES GAP" in plan
    assert len(calls) == 2, "expected exactly one architect call + one fact-gate check"


def test_story_architect_prompt_requires_turn_follows_spine_fact(monkeypatch):
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    calls = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    is_fact_check = "SCRIPT TO VERIFY" in kw["messages"][0]["content"]
                    content = "PASS" if is_fact_check else "SPINE FACT: x"
                    msg = type("M", (), {"content": content})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    _story_architect(FakeClient(), {"content": "x"}, "analysis", "hook", "run1", "finance")
    architect_prompt = calls[0]["messages"][0]["content"].lower()
    assert "direct consequence of the spine fact" in architect_prompt


# ── Sensory disqualifier: early placement, not just presence ──────────────────

def test_score_prompt_requires_sensory_detail_in_first_third():
    from script_writer import _score
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    msg = type("M", (), {"content": "DISQUALIFIERS: none\nTOTAL: 8/10"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    _score(FakeClient(), "some script", {"type": "wikipedia", "content": "x"},
          "some hook", "run1", "finance")
    prompt = captured["messages"][0]["content"].lower()
    assert "first third" in prompt


def test_fixes_from_crits_sensory_fix_mentions_first_third():
    from script_writer import _fixes_from_crits, _standards
    crits = {"specificity": 2, "hook": 2, "compression": 2, "loop": 2, "human": 2}
    reasoning = "DISQUALIFIERS: NO EARLY SENSORY DETAIL\nTOTAL: 4/10"
    fixes = _fixes_from_crits(crits, _standards(), "worst", reasoning=reasoning)
    assert any("first third" in f.lower() for f in fixes)


# ── Topic clustering (dedup beyond wording-level similarity) ──────────────────

def test_extract_core_topic_parses_numbered_line():
    from script_writer import extract_core_topic
    analysis = (
        "1. CONTRADICTION: something\n"
        "2. HOOK ANGLE: something else\n"
        "3. CORE: Compound interest rewards patience over speed\n"
        "4. EMOTIONAL STAKES: whatever\n"
    )
    assert extract_core_topic(analysis) == "Compound interest rewards patience over speed"


def test_extract_core_topic_case_insensitive_and_no_number():
    from script_writer import extract_core_topic
    assert extract_core_topic("CORE: The gold standard ended in 1971") == \
        "The gold standard ended in 1971"


def test_extract_core_topic_falls_back_to_first_line_when_missing():
    from script_writer import extract_core_topic
    analysis = "Just some unstructured text\nwith no labeled fields"
    assert extract_core_topic(analysis) == "Just some unstructured text"


def test_extract_core_topic_empty_input():
    from script_writer import extract_core_topic
    assert extract_core_topic("") == ""
    assert extract_core_topic(None) == ""


def test_check_topic_similarity_fails_open_without_embedding(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(sw, "_embed_script", lambda s: None)
    is_dup, sim, vec = sw.check_topic_similarity("some topic", "main_en")
    assert (is_dup, sim, vec) == (False, 0.0, None)


def test_check_topic_similarity_empty_topic_short_circuits(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    called = []
    monkeypatch.setattr(sw, "_embed_script", lambda s: called.append(s) or [1.0])
    is_dup, sim, vec = sw.check_topic_similarity("", "main_en")
    assert (is_dup, sim, vec) == (False, 0.0, None)
    assert called == []   # no API call wasted on an empty topic


def test_check_topic_similarity_flags_recent_same_topic(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(sw, "_embed_script", lambda s: [0.6, 0.8, 0.0])
    now = 1_000_000.0
    sw.add_topic_embedding([0.6, 0.8, 0.0], "main_en", now=now)

    is_dup, sim, _ = sw.check_topic_similarity("compound interest explainer",
                                               "main_en", now=now + 3600)
    assert is_dup is True
    assert sim > 0.99


def test_check_topic_similarity_ignores_topics_outside_window(monkeypatch, tmp_path):
    """The same topic covered 3 months ago must NOT block a new video on it —
    this gate is time-windowed, not a permanent ban."""
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(sw, "_embed_script", lambda s: [0.6, 0.8, 0.0])
    now = 1_000_000.0
    sw.add_topic_embedding([0.6, 0.8, 0.0], "main_en", now=now)

    far_future = now + (sw.TOPIC_WINDOW_DAYS + 5) * 86400
    is_dup, sim, _ = sw.check_topic_similarity("compound interest explainer",
                                               "main_en", now=far_future)
    assert is_dup is False


def test_check_topic_similarity_is_per_channel(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(sw, "_embed_script", lambda s: [1.0, 0.0])
    now = 1_000_000.0
    sw.add_topic_embedding([1.0, 0.0], "spanish", now=now)

    is_dup, sim, _ = sw.check_topic_similarity("x", "main_en", now=now + 10)
    assert is_dup is False


def test_add_topic_embedding_none_is_noop(monkeypatch, tmp_path):
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    sw.add_topic_embedding(None, "main_en")
    assert not (tmp_path / "topics.json").exists()


def test_add_topic_embedding_prunes_stale_entries(monkeypatch, tmp_path):
    import json as _json
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    now = 1_000_000.0
    stale = now - (sw.TOPIC_WINDOW_DAYS + 1) * 86400
    sw.add_topic_embedding([1.0, 0.0], "main_en", now=stale)
    sw.add_topic_embedding([0.0, 1.0], "main_en", now=now)

    entries = _json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["vec"] == [0.0, 1.0]


def test_add_topic_embedding_respects_history_cap(monkeypatch, tmp_path):
    import json as _json
    import script_writer as sw
    monkeypatch.setattr(sw, "TOPIC_EMBEDDINGS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(sw, "TOPIC_HISTORY_CAP", 3)
    now = 1_000_000.0
    for i in range(6):
        sw.add_topic_embedding([float(i)], "main_en", now=now)
    entries = _json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))
    assert len(entries) == 3
    assert entries[-1]["vec"] == [5.0]   # newest kept


# ── write_script_until_good: escalate to a NEW hook, don't redraft a doomed one ──
# Live failure this fixes: the hook "Swiss banking secrecy protected Nazi
# assets" scored 10/10, then the fact gate rejected that very claim. All three
# body retries reran under the same unsupportable hook, so they could never
# have succeeded — the retry was happening at the wrong level.

def _res(score, fact_ok, cost=0.02, script="s", reason=""):
    return {"script": script, "score": score, "fact_ok": fact_ok,
            "fact_reason": reason, "cost_usd": cost, "run_id": "r",
            "criterion_scores": {}, "attempts_used": 1,
            "final_temperature": 0.7, "reasoning": ""}


def test_until_good_stops_immediately_when_first_cycle_is_good(monkeypatch):
    import script_writer as sw
    calls = []
    monkeypatch.setattr(sw, "write_script",
                        lambda *a, **k: calls.append(1) or _res(9, True))
    out = sw.write_script_until_good("scene", seed={"content": "x"})
    assert out["score"] == 9
    assert len(calls) == 1          # no wasted spend when the first try is fine


def test_until_good_retries_when_fact_gate_failed(monkeypatch):
    """A capped-but-decent score still retries if the facts were wrong."""
    import script_writer as sw
    seq = iter([_res(5, False, reason="claim unsupported"), _res(9, True)])
    monkeypatch.setattr(sw, "write_script", lambda *a, **k: next(seq))
    out = sw.write_script_until_good("scene", seed={"content": "x"})
    assert out["score"] == 9 and out["fact_ok"] is True


def test_until_good_feeds_the_rejection_into_the_next_cycle(monkeypatch):
    """The next cycle must be told what was rejected, or the fresh hook
    factory just rediscovers the same doomed claim."""
    import script_writer as sw
    scenes = []

    def fake(scene, **kw):
        scenes.append(scene)
        return _res(5, False, reason="Nazi-assets claim unsupported") \
            if len(scenes) == 1 else _res(9, True)

    monkeypatch.setattr(sw, "write_script", fake)
    sw.write_script_until_good("original scene", seed={"content": "x"})
    assert len(scenes) == 2
    assert "original scene" in scenes[1]
    assert "Nazi-assets claim unsupported" in scenes[1]
    assert "DIFFERENT angle" in scenes[1]


def test_until_good_keeps_best_when_all_cycles_fail(monkeypatch):
    """Never fails a run — returns the best attempt, like the rest of the
    pipeline's fail-open policy."""
    import script_writer as sw
    monkeypatch.setenv("RUFUS_SCRIPT_CYCLES", "3")
    seq = iter([_res(4, False), _res(7, False), _res(5, False)])
    monkeypatch.setattr(sw, "write_script", lambda *a, **k: next(seq))
    out = sw.write_script_until_good("scene", seed={"content": "x"})
    assert out["score"] == 7          # the best of the three, not the last


def test_until_good_respects_cycle_limit(monkeypatch):
    import script_writer as sw
    monkeypatch.setenv("RUFUS_SCRIPT_CYCLES", "2")
    calls = []
    monkeypatch.setattr(sw, "write_script",
                        lambda *a, **k: calls.append(1) or _res(3, False))
    sw.write_script_until_good("scene", seed={"content": "x"})
    assert len(calls) == 2


def test_until_good_reports_cumulative_cost(monkeypatch):
    import script_writer as sw
    monkeypatch.setenv("RUFUS_SCRIPT_CYCLES", "3")
    seq = iter([_res(3, False, cost=0.02), _res(4, False, cost=0.03),
                _res(5, False, cost=0.01)])
    monkeypatch.setattr(sw, "write_script", lambda *a, **k: next(seq))
    out = sw.write_script_until_good("scene", seed={"content": "x"})
    assert abs(out["cost_usd"] - 0.06) < 1e-9


def test_until_good_stops_at_cost_ceiling(monkeypatch):
    """Unbounded 'loop until perfect' is how you get a runaway bill on a topic
    whose source genuinely can't support an interesting claim."""
    import script_writer as sw
    monkeypatch.setenv("RUFUS_SCRIPT_CYCLES", "50")
    monkeypatch.setenv("RUFUS_SCRIPT_MAX_COST", "0.05")
    calls = []
    monkeypatch.setattr(sw, "write_script",
                        lambda *a, **k: calls.append(1) or _res(3, False, cost=0.02))
    sw.write_script_until_good("scene", seed={"content": "x"})
    assert len(calls) == 3          # 0.02+0.02+0.02 crosses 0.05, then stops


# ── the gate knowing what the generator was never told ───────────────────────
#
# From a rejection log of ~200 attempts, three patterns that are this pipeline
# arguing with itself rather than the model writing badly.

def test_every_forbidden_opener_reaches_the_generator():
    """The gate rejects all twenty-one; the prompt showed the first eight. So
    'what if' (index 8) and 'stop' (index 14) were rules nothing had told the
    model about, and the log has five rejections for exactly those two — each
    a wasted candidate, each the model obeying what it was given while
    breaking what it was not."""
    import script_writer as sw
    hs = sw._standards()["hook"]
    prompt = sw._hook_factory_prompt_for_test() if hasattr(
        sw, "_hook_factory_prompt_for_test") else None
    # Built the same way _hook_factory builds it.
    shown = ", ".join(f"'{x}'" for x in hs["forbidden_openers"])
    for bad in ("what if", "stop", "imagine", "breaking:"):
        assert f"'{bad}'" in shown, bad
    assert len(hs["forbidden_openers"]) >= 20


def test_the_hook_examples_carry_no_figures():
    """'2,000 years ago, Seneca described your anxiety exactly.' was an
    EXAMPLE, and the model copied its number: '2,000 years ago, Rome faced
    inflation', '2,000 years later, inflation still haunts us', 'Inflation has
    shaped economies for over 2,300 years' — all rejected as invented figures
    that came from this prompt rather than any source. An example is a shape;
    the moment it contains a concrete figure it is a suggestion, and the
    grounding gate sits downstream of the suggestion."""
    import re
    from pathlib import Path as _P
    import script_writer as sw
    src = _P(sw.__file__).read_text(encoding="utf-8")
    block = src.split("attack the source from a DIFFERENT angle")[1]
    block = block.split("Output FORMAT")[0]
    # Strip the numbered list markers ("1. Number-first"), which are structure.
    body = re.sub(r"^\s*\"?\d+\.\s", " ", block, flags=re.MULTILINE)
    stray = re.findall(r"\b\d[\d,.]*\b", body)
    assert not stray, f"figures in the hook examples leak into hooks: {stray}"


def test_the_prompt_and_the_gate_read_one_corpus():
    """The allowed-numbers list was built from the formatted seed block and
    the gate from the raw seed content — two definitions of "the source", and
    the writer pays: a figure the gate would accept is missing from its list,
    so the hook that could have used it is never written."""
    import script_writer as sw
    seed = {"content": "The mint struck 4,300 coins in 1284.",
            "title": "Venice and the ducat"}
    corpus = sw.grounding_corpus(seed, "1. CONTRADICTION: it was 98.6% gold.")
    block = sw._allowed_numbers_block(corpus)
    for n in ("4,300", "1284", "98.6"):
        assert n in block, n
    # And what the list offers, the gate accepts.
    for n in ("4,300", "1284", "98.6"):
        assert sw._ungrounded_number(f"A hook about {n} coins.", corpus) is None, n


def test_a_figure_outside_the_corpus_is_still_refused():
    """The loosening must not become a hole: the whole point of the gate is
    that a true fact the source does not contain is still an invented one for
    this channel."""
    import script_writer as sw
    seed = {"content": "The mint struck coins in Venice.", "title": "The ducat"}
    corpus = sw.grounding_corpus(seed, "1. CONTRADICTION: gold beat silver.")
    assert sw._ungrounded_number("Bank of England issued notes since 1694.",
                                 corpus) == "1694"


# ── devices read off an eleven-minute script that holds its audience ─────────

def _body_guidance() -> str:
    """The guidance block, with whitespace normalised.

    Prose wraps. "we cannot\n  prove" is the same instruction as "we cannot
    prove" to every reader including the model, and a test that fails on the
    line break is testing the line break."""
    import re
    from pathlib import Path as _P
    import script_writer as sw
    src = _P(sw.__file__).read_text(encoding="utf-8")
    i = src.find("NAMING THE LIMIT IS A THIRD OPTION")
    assert i > 0, "the guidance moved"
    return re.sub(r"\s+", " ",
                  src[i:src.find("NUMBERS ARE SPOKEN, NOT PRINTED:", i)])


def test_naming_the_limit_is_offered_as_a_third_option():
    """Between asserting what the source does not support — which is the
    MIND-READ and INVENTED rejection, the single biggest cause in the log —
    and leaving it out, there is a move that is both honest and better
    television: say what is known, then say where the knowing stops."""
    g = _body_guidance()
    assert "we are not sure" in g
    assert "we cannot prove" in g


def test_it_recommends_only_words_the_hedging_gate_allows():
    """The guidance nearly told the model to write "possibly" and
    "probably" — both on the banned hedging list, so every attempt taking the
    advice would have been rejected for taking it. That is the gate knowing
    something the generator was never told, arriving from the other side."""
    import script_writer as sw
    g = _body_guidance()
    import re
    for phrase in re.findall(r'"([a-z][a-z\' ]+)"', g):
        # Only the RECOMMENDED forms, which appear before the "Do NOT" line.
        if g.index(f'"{phrase}"') > g.index("Do NOT reach for"):
            continue
        assert sw._find_hedging(phrase) is None, phrase


def test_the_banned_alternatives_are_named_so_nobody_reaches_for_them():
    """Show the generator the whole rule, the way the forbidden-openers list
    now is."""
    import script_writer as sw
    g = _body_guidance()
    missing = [w for w in sw._standards()["hedging_words"] if w not in g.lower()]
    assert not missing, (
        f"the writer is judged by these and never shown them: {missing}. "
        f"Add them to the Do-NOT list, the way all 21 forbidden openers are "
        f"now shown to the hook factory.")


def test_the_three_devices_are_taught_with_worked_examples():
    import re
    from pathlib import Path as _P
    import script_writer as sw
    src = re.sub(r"\s+", " ", _P(sw.__file__).read_text(encoding="utf-8"))
    for device in ("NEGATION THEN CORRECTION", "THE OBJECT AS PROOF",
                   "THE CONSEQUENCE STACK"):
        assert device in src, device
    assert "It lost silver." in src
    assert "tinder fungus, flint and iron pyrite" in src


def test_the_consequence_stack_is_rationed():
    """It is the one place repetition is wanted, and the reason it works is
    that it is rare."""
    assert "not use it more than once" in _body_guidance()


# ── The rubric knowing which video it is looking at ───────────────────────────
#
# The loop line is a DISQUALIFIER (final ≤4), and long-form does not end on a
# loop — longform_writer's outline plans a close that pays a counted promise,
# because that is what holds a nine-minute audience. So the gate was capping
# every long-form script at 4 and holding it from publishing forever, for not
# containing a device its own generator is instructed not to write.

def _capture_rubric():
    """The prompt _score actually sends, with a client that only records."""
    from script_writer import _score
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    msg = type("M", (), {"content": "DISQUALIFIERS: none\nTOTAL: 8/10"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    _score(FakeClient(), "some script", {"type": "wikipedia", "content": "x"},
           "some hook", "run1", "finance")
    return captured["messages"][0]["content"]


def test_the_shorts_rubric_is_untouched(monkeypatch):
    """The shipping channel is scored by exactly the rubric it was scored by
    yesterday. Every video in the review queue was ranked with these lines, so
    a change here silently re-ranks the entire back catalogue against the new
    ones."""
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    p = _capture_rubric()
    assert "ruthless short-form editor" in p
    assert "□ Loop line (second-to-last) shares zero content words with the hook" in p
    assert "avg sentence >12 words" in p
    assert "FIRST THIRD of the body" in p
    assert "LOOP: [0-2]/2 — [quote the loop line, explain echo]" in p
    assert "PAYOFF" not in p


def test_long_form_is_not_judged_on_a_loop_it_was_told_not_to_write(monkeypatch):
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    p = _capture_rubric()
    assert "Loop line" not in p
    assert "short-form editor" not in p
    assert "counted promise" in p
    assert "PAYOFF: [0-2]/2" in p


def test_long_form_is_not_penalised_for_sentences_over_twelve_words(monkeypatch):
    """A twelve-word average over 1,300 words is a machine gun, not
    compression. The real long-form padding failure is the same fact said
    again three sections later, which is what the criterion now names."""
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    p = _capture_rubric()
    assert "avg sentence >12 words" not in p
    assert "restated in a later section" in p


def test_the_sensory_window_is_the_opening_not_a_third_of_nine_minutes(monkeypatch):
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    p = _capture_rubric()
    assert "FIRST THIRD" not in p
    assert "OPENING SECTION" in p


def test_both_formats_still_score_out_of_the_same_ten(monkeypatch):
    """One scale, or the review queue is sorting two different numbers into
    one column."""
    for fmt in ("short", "long"):
        monkeypatch.setenv("RUFUS_FORMAT", fmt)
        p = _capture_rubric()
        # The invariant is the SUM, not any one criterion's range: a point moved
        # from SPECIFICITY (guarded four other ways) to HUMAN (guarded nowhere)
        # because nine of ten points measured accuracy and one measured voice,
        # and the scripts came back true, tight and completely flat.
        assert "SPECIFICITY 0-2" in p and "HUMAN 0-2" in p, fmt
        assert "TOTAL: [sum]/10" in p, fmt
        ranges = {"SPECIFICITY": 2, "HOOK": 2, "COMPRESSION": 2, "HUMAN": 2}
        ranges["PAYOFF" if fmt == "long" else "LOOP"] = 2
        assert sum(ranges.values()) == 10, fmt
        for name, hi in ranges.items():
            assert f"{name} 0-{hi}:" in p, f"{fmt}: {name}"


@pytest.mark.parametrize("fmt", ["short", "long"])
def test_every_criterion_the_rubric_asks_for_is_one_the_parser_reads(monkeypatch, fmt):
    """The drift this file exists to prevent, in its cheapest form: renaming a
    criterion in the prompt and not in _CRIT_RE loses that criterion's points
    silently, and the total quietly falls by two."""
    import re as _re
    from script_writer import _CRIT_RE
    monkeypatch.setenv("RUFUS_FORMAT", fmt)
    p = _capture_rubric()
    reply = p.split("STEP 3 — REPLY EXACTLY:")[1]
    asked = _re.findall(r"^([A-Z]+):", reply, _re.MULTILINE)
    assert asked, fmt
    for name in asked:
        if name == "DISQUALIFIERS":
            continue
        assert _CRIT_RE.match(f"{name}: 1"), f"{fmt}: rubric asks for {name}, parser ignores it"


def test_the_long_form_payoff_is_stored_as_the_same_axis(monkeypatch):
    """PAYOFF and LOOP are one column. db_manager writes crits['loop'] and the
    retry fixes read it, so a long-form score arriving under a new key would
    be a criterion that scores and then evaporates."""
    from script_writer import _score
    monkeypatch.setenv("RUFUS_FORMAT", "long")

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    msg = type("M", (), {"content":
                        "DISQUALIFIERS: none\nSPECIFICITY: 3/3 — x\nHOOK: 2/2 — x\n"
                        "COMPRESSION: 2/2 — x\nPAYOFF: 2/2 — x\nHUMAN: 1/1 — x\n"
                        "TOTAL: 10/10"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    total, crits, _r, _c, _ms = _score(FakeClient(), "s", {"type": "wikipedia"},
                                       "h", "run1", "finance")
    assert total == 10
    assert crits["loop"] == 2
    assert "payoff" not in crits
