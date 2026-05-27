#!/usr/bin/env python3
"""
research.py – Returns real source material the script writer compresses into a Short.

Strategy:
    1. Try Reddit hot/top posts from niche-specific subreddits
    2. Filter aggressively for substance (score, body length, no rage-bait)
    3. If nothing passes, fall back to a curated wisdom quote pool

A Seed dict is returned with keys:
    - type:    "reddit" | "wisdom"
    - content: the substantive text (story or quote)
    - source:  subreddit name OR author name
    - title:   reddit post title (empty for wisdom)
    - url:     reddit permalink (empty for wisdom)
"""

import json
import os
import random
import re
import sys
from pathlib import Path

import httpx

CONFIG_DIR       = Path(__file__).parent.parent / "config"
NICHES_FILE      = CONFIG_DIR / "niches.json"
WISDOM_DIR       = CONFIG_DIR / "wisdom"
USED_SEEDS_FILE  = CONFIG_DIR / "used_seeds.json"
MAX_USED_HISTORY = 500  # cap to avoid unbounded growth

REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/123.0 rufus-research/1.0",
    "Accept":     "application/json",
}
REDDIT_TIMEOUT  = 10.0
REDDIT_ENDPOINTS = [
    "https://www.reddit.com/r/{sub}/hot.json?limit={lim}",
    "https://old.reddit.com/r/{sub}/hot.json?limit={lim}",
    "https://api.reddit.com/r/{sub}/hot?limit={lim}",
]

# Quality thresholds for Reddit posts
MIN_SCORE         = 500
MIN_BODY_LEN      = 300
MIN_COMMENTS      = 50
MAX_BODY_LEN      = 3000  # too long = won't fit a 35-50s Short

# Title patterns that signal low-substance content (rage, drama, mod posts)
TITLE_BAD_RE = re.compile(
    r"\b(aita|am i the|rant|vent|meta|update|edit|removed|deleted|"
    r"\[meta\]|\[mod\]|\[removed\]|\[deleted\]|"
    r"help|advice needed|question|what should i do)\b",
    re.IGNORECASE,
)

# Title patterns that signal a DISCUSSION (no story arc, no payoff — bad for Shorts)
TITLE_DISCUSSION_RE = re.compile(
    r"\b(thoughts on|anyone else|opinion on|worth it|should i|"
    r"is it worth|are you|do you|how do you|what do you|"
    r"discussion|debate|poll|survey|"
    r"competitiveness|alternatives|recommendations)\b",
    re.IGNORECASE,
)

# Title patterns that signal a STORY (concrete event, gold for Shorts).
# A post must match at least one of these to pass quality filter.
TITLE_STORY_RE = re.compile(
    r"(\$|\d|"                                 # any dollar sign or digit
    r"\bsaved\b|\bmade\b|\blost\b|\bpaid\b|"
    r"\bearned\b|\bspent\b|\bquit\b|\bfired\b|"
    r"\bstarted\b|\blearned\b|\bbought\b|\bsold\b|"
    r"\bbuilt\b|\bfailed\b|\bretired\b|\bescaped\b|"
    r"\bturned\b|\bwent\b|\btook\b|\bfound\b|"
    r"\byears? ago\b|\blast year\b|\blast month\b)",
    re.IGNORECASE,
)


def _load_niche():
    data   = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


# ── Seed deduplication ──────────────────────────────────────────────────────────

import hashlib

def _seed_id(seed: dict) -> str:
    """Stable unique ID for a seed so we don't reuse the same source twice."""
    if not seed:
        return ""
    if seed.get("type") == "reddit":
        return "reddit:" + (seed.get("url") or seed.get("title", ""))
    if seed.get("type") == "wisdom":
        text = (seed.get("content") or "").strip().lower()
        return "wisdom:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return ""


def _load_used_seeds() -> list:
    if not USED_SEEDS_FILE.exists():
        return []
    try:
        return json.loads(USED_SEEDS_FILE.read_text())
    except Exception:
        return []


def _mark_seed_used(seed: dict) -> None:
    sid = _seed_id(seed)
    if not sid:
        return
    used = _load_used_seeds()
    if sid in used:
        used.remove(sid)        # move to end (most-recent-used at tail)
    used.append(sid)
    used = used[-MAX_USED_HISTORY:]  # keep last N
    USED_SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USED_SEEDS_FILE.write_text(json.dumps(used, indent=2))


def _post_seed_id(post_data: dict) -> str:
    permalink = post_data.get("permalink", "")
    return "reddit:" + (f"https://reddit.com{permalink}" if permalink else post_data.get("title", ""))


def _quote_seed_id(quote: dict) -> str:
    text = (quote.get("text") or "").strip().lower()
    return "wisdom:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """Strip markdown, URLs, and excessive whitespace so it reads as plain prose."""
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"http\S+", "", text)         # urls
    text = re.sub(r"\*+", "", text)             # bold/italic markdown
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)  # blockquote markers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _passes_quality_filter(post: dict) -> bool:
    """Return True if a Reddit post has enough substance to be worth using.

    Requirements:
      - Engagement: score >= MIN_SCORE, num_comments >= MIN_COMMENTS
      - Substance: body length within MIN_BODY_LEN..MAX_BODY_LEN
      - Title contains a STORY signal (dollar sign, number, or past-tense action verb)
      - Title does NOT match BAD patterns (rage/drama/mod)
      - Title does NOT match DISCUSSION patterns (no story arc — bad for Shorts)
    """
    d = post.get("data", {})
    if d.get("stickied") or d.get("over_18") or d.get("removed_by_category"):
        return False
    if d.get("score", 0) < MIN_SCORE:
        return False
    if d.get("num_comments", 0) < MIN_COMMENTS:
        return False
    body = d.get("selftext") or ""
    if not body or len(body) < MIN_BODY_LEN or len(body) > MAX_BODY_LEN:
        return False
    title = d.get("title", "")
    if TITLE_BAD_RE.search(title):
        return False
    if TITLE_DISCUSSION_RE.search(title):
        return False
    if not TITLE_STORY_RE.search(title):
        # Title has no number, no past-tense action — almost certainly opinion/question
        return False
    return True


def fetch_reddit_story(subreddit: str, limit: int = 50, used_ids: set | None = None) -> dict | None:
    """Fetch the first quality-filtered hot post from a subreddit, skipping seen IDs.

    Larger default limit (50) gives more candidates after the used-filter prunes the list.
    """
    if used_ids is None:
        used_ids = set()

    data = None
    last_err = None
    for template in REDDIT_ENDPOINTS:
        url = template.format(sub=subreddit, lim=limit)
        try:
            r = httpx.get(url, headers=REDDIT_HEADERS, timeout=REDDIT_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            continue

    if data is None:
        print(f"[research] reddit unreachable for r/{subreddit} ({last_err})")
        return None

    posts = data.get("data", {}).get("children", [])
    # Quality filter first, then drop anything we've already used
    quality = [
        p for p in posts
        if _passes_quality_filter(p)
        and _post_seed_id(p.get("data", {})) not in used_ids
    ]
    if not quality:
        return None

    chosen = random.choice(quality[:8])  # randomize among top 8 unused quality posts
    d = chosen["data"]
    return {
        "type":    "reddit",
        "source":  f"r/{subreddit}",
        "title":   d.get("title", ""),
        "content": _clean_text(d.get("selftext", "")),
        "url":     f"https://reddit.com{d.get('permalink', '')}",
    }


def pick_wisdom_quote(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Pick one random quote from the niche's curated pool, skipping seen ones."""
    if used_ids is None:
        used_ids = set()
    f = WISDOM_DIR / f"{niche_name}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    quotes = data.get("quotes", [])
    if not quotes:
        return None
    # Prefer unused quotes; if all used (rare), fall back to full pool
    fresh = [q for q in quotes if _quote_seed_id(q) not in used_ids]
    pool  = fresh if fresh else quotes
    q     = random.choice(pool)
    return {
        "type":    "wisdom",
        "source":  q.get("author", "Unknown"),
        "title":   "",
        "content": q.get("text", ""),
        "url":     "",
    }


def get_seed(niche_name: str | None = None) -> dict:
    """Get a seed for the script writer. Tracks history so seeds never repeat.

    Order: Reddit-first (story-filtered), wisdom fallback. Both skip anything
    already in used_seeds.json (last MAX_USED_HISTORY items).
    """
    niche, active = _load_niche()
    name      = niche_name or active
    used_list = _load_used_seeds()
    used_set  = set(used_list)

    print(f"[research] history: {len(used_list)} prior seeds will be skipped")

    subreddits = list(niche.get("subreddits", []))
    random.shuffle(subreddits)

    for sub in subreddits:
        seed = fetch_reddit_story(sub, used_ids=used_set)
        if seed:
            print(f"[research] using Reddit story from {seed['source']}: \"{seed['title'][:60]}\"")
            _mark_seed_used(seed)
            return seed

    # No Reddit story passed filters — fall back to wisdom
    seed = pick_wisdom_quote(name, used_ids=used_set)
    if seed:
        print(f"[research] using wisdom quote from {seed['source']}")
        _mark_seed_used(seed)
        return seed

    # Last resort
    print(f"[research] WARNING: no seed found for niche '{name}'")
    fallback = {
        "type":    "wisdom",
        "source":  "Marcus Aurelius",
        "content": "What stands in the way becomes the way.",
        "title":   "",
        "url":     "",
    }
    _mark_seed_used(fallback)
    return fallback


if __name__ == "__main__":
    niche = sys.argv[1] if len(sys.argv) > 1 else None
    seed  = get_seed(niche)
    print(json.dumps(seed, indent=2))
