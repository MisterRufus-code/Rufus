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
