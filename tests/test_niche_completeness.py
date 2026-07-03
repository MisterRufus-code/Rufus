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
