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
