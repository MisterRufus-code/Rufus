"""Every niche in niches.json must be fully wired across all support maps.

This is the guard against 'half-wired niche' bugs: a niche that exists in
niches.json but is missing from the music/hashtag/seed maps silently degrades
(generic music, #Shorts-only hashtags, or a single repeated fallback quote).
Adding a new niche must fail these tests until every map knows it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

ROOT   = Path(__file__).parent.parent
NICHES = json.loads((ROOT / "config" / "niches.json").read_text())["niches"]


def test_every_niche_has_wisdom_pool():
    """The wisdom pool is the zero-API-key seed source of last resort — without
    a file, the pipeline repeats one hardcoded fallback quote forever."""
    for niche in NICHES:
        f = ROOT / "config" / "wisdom" / f"{niche}.json"
        assert f.exists(), f"missing wisdom pool: config/wisdom/{niche}.json"
        quotes = json.loads(f.read_text()).get("quotes", [])
        assert len(quotes) >= 20, f"{niche} wisdom pool too small ({len(quotes)} < 20)"
        for q in quotes[:5]:
            assert q.get("text") and q.get("author"), f"malformed entry in {niche}.json"


def test_every_niche_has_gold_examples():
    """Gold examples are the strongest quality lever in the script writer —
    gold_examples.json's own note says "the model mimics these more than any
    instruction, so they define the voice". A niche with none ships a system
    prompt with ZERO demonstrations (_build_gold_block returns "" on an empty
    list), which is exactly how money_history ran in production while every
    other niche had two: its scripts opened as biography ("In 1397, Giovanni
    di Bicci de' Medici opened the Medici Bank") instead of viewer-first, the
    one thing the note explicitly warns against."""
    gold = json.loads((ROOT / "config" / "gold_examples.json").read_text())
    for niche in NICHES:
        examples = gold.get(niche, [])
        assert len(examples) >= 2, \
            f"{niche} has {len(examples)} gold example(s) in gold_examples.json, need >= 2"
        for ex in examples:
            assert ex.get("script"), f"{niche} gold example missing 'script'"
            assert ex.get("seed_content"), f"{niche} gold example missing 'seed_content'"


# Legacy gold examples (everything except money_history) predate the word cap,
# the cadence check and the em-dash check, and 9 of the 10 of them would be
# REJECTED by the pipeline's own gates today — they demonstrate 122-128 word
# scripts against a 115 cap, and uniform sentence lengths against the cadence
# rule. That is actively harmful few-shot data: the model mimics the example,
# then burns a generation attempt getting rejected for copying it. Fixing them
# means rewriting five niches' channel voice, which is the owner's call, so
# they are pinned here rather than silently tolerated or silently rewritten.
#
# 9 -> 8: one of them was failing "loop no echo" on a word FORM. The gate
# matched the hook against the loop by exact string, so an example whose loop
# line echoed the hook's noun in the plural counted as no echo at all. Nothing
# about the example changed; the gate stopped being wrong about it.
_LEGACY_GOLD_NICHES = {"finance", "motivation", "mindset", "business",
                       "personal_development"}
_KNOWN_FAILING_LEGACY_EXAMPLES = 8


def test_money_history_gold_examples_pass_the_pipelines_own_body_gates():
    """money_history is the niche actually in production, and its examples were
    written against the current gates. A gold example the pipeline would reject
    teaches the model to write scripts that get rejected, so these must stay
    clean. Uses the real _body_violations rather than re-checking the rules."""
    import script_writer

    gold = json.loads((ROOT / "config" / "gold_examples.json").read_text())
    for i, ex in enumerate(gold.get("money_history", []), 1):
        violations = script_writer._body_violations(ex["script"])
        assert not violations, \
            f"money_history gold example {i} would be REJECTED by the pipeline: {violations}"


def test_legacy_gold_examples_do_not_get_worse():
    """Ratchet, not an endorsement — see _KNOWN_FAILING_LEGACY_EXAMPLES above.
    Adding another gate-failing example to a legacy niche fails this test; fixing
    the existing ones fails it too, with a message saying to lower the number."""
    import script_writer

    gold = json.loads((ROOT / "config" / "gold_examples.json").read_text())
    failing = [
        f"{niche}#{i}: {script_writer._body_violations(ex['script'])}"
        for niche in sorted(_LEGACY_GOLD_NICHES)
        for i, ex in enumerate(gold.get(niche, []), 1)
        if script_writer._body_violations(ex["script"])
    ]
    assert len(failing) <= _KNOWN_FAILING_LEGACY_EXAMPLES, (
        f"a legacy gold example now fails the body gates that didn't before:\n"
        + "\n".join(failing))
    assert len(failing) == _KNOWN_FAILING_LEGACY_EXAMPLES, (
        f"legacy gold examples were fixed ({len(failing)} now failing, expected "
        f"{_KNOWN_FAILING_LEGACY_EXAMPLES}) — lower _KNOWN_FAILING_LEGACY_EXAMPLES "
        f"to lock the improvement in.")


def test_every_niche_has_music_mood():
    from music_fetcher import MOOD_MAP
    for niche in NICHES:
        assert niche in MOOD_MAP, f"{niche} missing from music_fetcher.MOOD_MAP"


def test_every_niche_has_musicgen_prompt():
    from musicgen_gen import NICHE_PROMPTS
    for niche in NICHES:
        assert niche in NICHE_PROMPTS, f"{niche} missing from musicgen_gen.NICHE_PROMPTS"


def test_every_niche_has_synth_bed():
    from music_gen import NICHE_BEDS
    for niche in NICHES:
        assert niche in NICHE_BEDS, f"{niche} missing from music_gen.NICHE_BEDS"


def test_every_niche_has_hashtags():
    from youtube_uploader import NICHE_HASHTAGS, DEFAULT_CATEGORIES
    for niche in NICHES:
        assert niche in NICHE_HASHTAGS, f"{niche} missing from NICHE_HASHTAGS"
        assert len(NICHE_HASHTAGS[niche]) >= 4
        assert "#Shorts" in NICHE_HASHTAGS[niche]
        assert niche in DEFAULT_CATEGORIES, f"{niche} missing from DEFAULT_CATEGORIES"


def test_every_niche_has_trend_seeds():
    from research import NICHE_TREND_SEEDS
    for niche in NICHES:
        assert niche in NICHE_TREND_SEEDS, f"{niche} missing from NICHE_TREND_SEEDS"


def test_every_niche_declared_in_research_source_maps():
    """SE/HN maps may map a niche to None (source not suitable) — but the key
    must exist, proving the choice was deliberate, not forgotten."""
    from research import SE_NICHE_SITES, HN_NICHE_QUERIES, RSS_FEEDS
    for niche in NICHES:
        assert niche in SE_NICHE_SITES, f"{niche} missing from SE_NICHE_SITES"
        assert niche in HN_NICHE_QUERIES, f"{niche} missing from HN_NICHE_QUERIES"
        assert niche in RSS_FEEDS, f"{niche} missing from RSS_FEEDS"


def test_every_niche_has_required_config_keys():
    required = ("display_name", "subreddits", "gpt_system", "cta", "cta_pool",
                "accent_color", "style_suffix", "youtube_category_id",
                "llava_context", "video_keywords")
    for niche, cfg in NICHES.items():
        for key in required:
            assert key in cfg, f"{niche} missing config key '{key}' in niches.json"


def test_scheduled_and_active_niches_exist():
    data = json.loads((ROOT / "config" / "niches.json").read_text())
    assert data["active"] in NICHES
    for n in data.get("schedule", []):
        assert n in NICHES, f"scheduled niche '{n}' not defined"


def test_money_history_permits_evergreen_concepts_not_just_events():
    """Per clarified channel-owner direction: the niche was strictly
    'one true, specific story from monetary history' — only 155 real
    events exist, which is what produced the topic-repetition complaint.
    Broadened to also permit timeless financial/economic CONCEPT scripts
    (illustrated with a real historical example each time), while keeping
    the existing boundaries (no investment advice, no motivational fluff)
    fully intact — this locks in both halves of that change."""
    gpt_system = NICHES["money_history"]["gpt_system"].lower()
    assert "concept" in gpt_system
    assert "compound interest" in gpt_system   # a concrete example concept present
    # The boundaries that made this a real editorial call, not a scope
    # blowout, must survive the change.
    assert "no investment advice" in gpt_system
    assert "no motivational fluff" in gpt_system
    assert "no get-rich talk" in gpt_system


def test_money_history_has_no_recurring_character():
    """The Chronicler was removed at the owner's request — see
    test_character_engine.test_money_history_ships_no_character_at_all for the
    reasoning. A niche with no `character` key runs the whole pipeline with
    every character branch falling through, which is the intended state here."""
    assert "character" not in NICHES["money_history"]


def test_every_sd_niche_has_a_starter_character_disabled_by_default():
    """character_engine.py is generic per-niche, not FLUX/money_history-only
    — finance/motivation/mindset/business/personal_development each ship a
    distinct starter mascot (per channel-owner direction: 'build characters
    for more niches, I'll use them in the future if I want'), but OFF by
    default since none has been reviewed/approved for real videos yet."""
    sd_niches = [n for n, cfg in NICHES.items() if cfg.get("video_source") == "sd"]
    assert sd_niches   # sanity: the SD niches actually exist in this config
    names = set()
    for niche in sd_niches:
        char = NICHES[niche].get("character")
        assert isinstance(char, dict), f"{niche} missing a character block"
        assert char.get("enabled") is False, f"{niche}'s character must ship disabled"
        assert char.get("name"), f"{niche} character has no name"
        assert char.get("description"), f"{niche} character has no description"
        names.add(char["name"])
    # Each niche's mascot must be visually distinct — not the same character
    # relabeled across niches.
    assert len(names) == len(sd_niches)


# ── the config that was declared and never read ────────────────────────────
#
# Two keys in this repo described how the scripts should sound and were wired
# to nothing. That is the same shape as every other bug this pipeline has had:
# the knowledge existed, computed and written down, and never reached the place
# that needed it.

def test_the_voice_note_actually_reaches_the_model():
    """`gold_examples.json` held the single best description of this channel's
    voice — open on the viewer, second person, the famous name as PROOF not as
    subject — in a key `_load_gold_examples` never read. Two examples were
    carrying the whole weight of teaching a voice that was written down in
    words the entire time."""
    import script_writer as sw
    voice = sw._gold_voice_note()
    assert voice, "no voice note is being sent at all"
    block = sw._build_gold_block(sw._load_gold_examples("money_history"))
    assert voice.splitlines()[0] in block


def test_the_developer_note_does_not_reach_the_model():
    """`note` is addressed to whoever opens the file. Telling the model it
    "mimics these more than any instruction" is true, useless to it, and a
    strange thing to say to something you are about to instruct."""
    import json as _json
    import script_writer as sw
    raw = _json.loads((ROOT / "config" / "gold_examples.json").read_text(encoding="utf-8"))
    block = sw._build_gold_block(sw._load_gold_examples("money_history"))
    assert raw["note"].split(".")[0] not in block


def test_every_niche_that_declares_hook_styles_has_them_sent_somewhere():
    """money_history declared counterintuitive/shocking_stat/warning since it
    was written, and a grep for hook_styles across every script returned
    nothing. A niche was describing how it wanted to open its videos into a
    void."""
    import json as _json
    import script_writer as sw
    niches = _json.loads((ROOT / "config" / "niches.json").read_text(encoding="utf-8"))["niches"]
    seen = False
    for name, cfg in niches.items():
        if not cfg.get("hook_styles"):
            continue
        seen = True
        blk = sw._hook_styles_block(cfg)
        for style in cfg["hook_styles"]:
            assert style in blk, f"{name}: {style} never reaches the generator"
    assert seen, "no niche declares hook_styles — this test is watching nothing"


def test_a_niche_with_no_hook_styles_adds_nothing_to_the_prompt():
    import script_writer as sw
    assert sw._hook_styles_block({}) == ""
    assert sw._hook_styles_block({"hook_styles": []}) == ""


def test_the_gold_set_teaches_more_than_one_shape():
    """Both money_history originals denied the obvious explanation — "Rome
    didn't run out of silver", "Henry gutted the currency". Two examples of one
    move teach one lesson twice, and a writer shown one lesson writes one
    shape."""
    import json as _json
    raw = _json.loads((ROOT / "config" / "gold_examples.json").read_text(encoding="utf-8"))
    hooks = [ex["script"].splitlines()[0].lower() for ex in raw["money_history"]]
    assert len(hooks) >= 3, "not enough examples to demonstrate a range"
    # At least one must put the viewer in the scene rather than deny something.
    assert any(h.startswith("in ") and "you" in h for h in hooks), hooks


def test_every_gold_example_is_shown_not_just_the_first_two():
    """The slice was examples[:2], written when every niche had exactly two —
    so examples added to fix the range problem were silently discarded."""
    import script_writer as sw
    examples = sw._load_gold_examples("money_history")
    block = sw._build_gold_block(examples)
    assert block.count("Example ") == len(examples), \
        f"{len(examples)} examples on disk, {block.count('Example ')} in the prompt"
