"""Tests for script_writer.py – banned-phrase detection and blacklist keying."""

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

    entries = _json.loads((tmp_path / "emb.json").read_text())
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
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 2, "human": 1}
    assert _fixes_from_crits(crits, _std(), "worst, smartest, wrong") == []


def test_fixes_from_crits_flags_low_specificity():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 0, "hook": 2, "compression": 2, "loop": 2, "human": 1}
    fixes = _fixes_from_crits(crits, _std(), "worst, smartest, wrong")
    assert any("ground EVERY claim" in f for f in fixes)


def test_fixes_from_crits_flags_low_loop():
    from script_writer import _fixes_from_crits
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 0, "human": 1}
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
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 2, "human": 1}
    reasoning = "DISQUALIFIERS: NO SENSORY DETAIL — entirely abstract summary\nTOTAL: 4/10"
    fixes = _fixes_from_crits(crits, _standards(), "worst", reasoning=reasoning)
    assert any("sensory" in f.lower() for f in fixes)


def test_fixes_from_crits_no_sensory_fix_when_not_flagged():
    from script_writer import _fixes_from_crits, _standards
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 2, "human": 1}
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
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    msg = type("M", (), {"content": "SPINE FACT: x\nTHE TURN: y\n"
                                                     "STAKES GAP: z\nWHY NOW: w"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    plan, _ = _story_architect(FakeClient(), {"content": "x"}, "analysis",
                               "hook", "run1", "finance")
    prompt = captured["messages"][0]["content"]
    assert "STAKES GAP" in prompt
    assert "STAKES GAP" in plan


def test_story_architect_prompt_requires_turn_follows_spine_fact(monkeypatch):
    from script_writer import _story_architect
    monkeypatch.delenv("RUFUS_SCRIPT_ARCHITECT", raising=False)
    captured = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    msg = type("M", (), {"content": "SPINE FACT: x"})()
                    choice = type("C", (), {"message": msg})()
                    usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
                    return type("R", (), {"choices": [choice], "usage": usage})()

    _story_architect(FakeClient(), {"content": "x"}, "analysis", "hook", "run1", "finance")
    prompt = captured["messages"][0]["content"].lower()
    assert "direct consequence of the spine fact" in prompt


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
    crits = {"specificity": 3, "hook": 2, "compression": 2, "loop": 2, "human": 1}
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

    entries = _json.loads((tmp_path / "topics.json").read_text())
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
    entries = _json.loads((tmp_path / "topics.json").read_text())
    assert len(entries) == 3
    assert entries[-1]["vec"] == [5.0]   # newest kept
