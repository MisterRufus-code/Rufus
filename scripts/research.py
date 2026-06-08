#!/usr/bin/env python3
"""
research.py – Returns real source material the script writer compresses into a Short.

Strategy:
    1. Try Reddit hot/top posts from niche-specific subreddits
    2. Try Hacker News "Ask HN" posts (finance/business/mindset niches)
    3. If nothing passes quality filters, fall back to a curated wisdom quote pool

A Seed dict is returned with keys:
    - type:    "reddit" | "hackernews" | "wisdom"
    - content: the substantive text (story or quote)
    - source:  subreddit name | "Hacker News" | author name
    - title:   post title (empty for wisdom)
    - url:     post permalink (empty for wisdom)
"""

import hashlib
import html as html_module
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx

try:
    import praw as _praw_mod
    _PRAW_AVAILABLE = True
except ImportError:
    _PRAW_AVAILABLE = False

CONFIG_DIR       = Path(__file__).parent.parent / "config"
NICHES_FILE      = CONFIG_DIR / "niches.json"
WISDOM_DIR       = CONFIG_DIR / "wisdom"
USED_SEEDS_FILE  = CONFIG_DIR / "used_seeds.json"
MAX_USED_HISTORY = 500  # cap to avoid unbounded growth

REDDIT_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control":   "no-cache",
    "DNT":             "1",
}
REDDIT_TIMEOUT  = 10.0
# old.reddit.com .json is the most permissive endpoint for cloud IPs.
# www.reddit.com .json sometimes works; api.reddit.com is the most aggressive blocker.
REDDIT_ENDPOINTS = [
    "https://old.reddit.com/r/{sub}/hot.json?limit={lim}&raw_json=1",
    "https://www.reddit.com/r/{sub}/hot.json?limit={lim}&raw_json=1",
    "https://old.reddit.com/r/{sub}/top.json?limit={lim}&t=week&raw_json=1",
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

# Title patterns that signal off-topic content regardless of niche.
# Dating/relationship/food/parenting posts slip through story filters via past-tense verbs
# ("learned", "discovered") but have no niche-relevant content.
TITLE_OFFTOPIC_RE = re.compile(
    r"\b(dating|relationship|boyfriend|girlfriend|romance|"
    r"marriage|divorce|breakup|hookup|tinder|crush|"
    r"recipe|cooking|baking|meal|diet|nutrition|weight loss|"
    r"fashion|beauty|makeup|skincare|"
    r"pregnancy|parenting|baby|toddler|kids|"
    r"pets|dog|cat|hamster|gaming|minecraft|fortnite)\b",
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

# ── Hacker News config ───────────────────────────────────────────────────────────

HN_TIMEOUT      = 10.0
HN_MIN_POINTS   = 100   # lowered – HN posts compete with each other, 100+ is solid
HN_MIN_COMMENTS = 20    # lowered – many great Ask HN posts have 20-50 comments
HN_MIN_TEXT_LEN = 150   # lowered – some HN posts are short but punchy
HN_MAX_TEXT_LEN = 3000

# Niches that map well to HN's intellectual, founder-heavy audience.
# None = skip HN for that niche (motivation content doesn't resonate there).
# Terms chosen to match real Ask HN post vocabulary — keep them conversational.
HN_NICHE_QUERIES = {
    "finance":             "money investing wealth financial independence early retirement",
    "business":            "startup founder entrepreneur lessons learned failure",
    "mindset":             "mental clarity habits decision systems focus burnout thinking",
    "personal_development": "habits discipline self improvement learning productivity",
    "motivation":          None,
}


def _load_niche():
    data   = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


# ── Seed deduplication ──────────────────────────────────────────────────────────

def _seed_id(seed: dict) -> str:
    """Stable unique ID for a seed so we don't reuse the same source twice."""
    if not seed:
        return ""
    t = seed.get("type")
    if t == "reddit":
        return "reddit:" + (seed.get("url") or seed.get("title", ""))
    if t == "hackernews":
        return "hn:" + (seed.get("url") or seed.get("title", ""))
    if t == "wisdom":
        text = (seed.get("content") or "").strip().lower()
        return "wisdom:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return ""


def _load_used_seeds() -> list:
    if not USED_SEEDS_FILE.exists():
        return []
    try:
        return json.loads(USED_SEEDS_FILE.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        print("[research] ⚠ recovered from corrupted used_seeds.json — history reset")
        return []


def _mark_seed_used(seed: dict) -> None:
    sid = _seed_id(seed)
    if not sid:
        return
    used = _load_used_seeds()
    if sid in used:
        used.remove(sid)        # move to end (most-recent-used at tail)
    used.append(sid)
    used = used[-MAX_USED_HISTORY:]
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
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_html(text: str) -> str:
    """Remove HTML tags from Ask HN posts (which use <p>, <i>, <br> tags)."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html_module.unescape(text).strip()


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
    if TITLE_OFFTOPIC_RE.search(title):
        return False
    if not TITLE_STORY_RE.search(title):
        return False
    return True


def _load_keys() -> dict:
    keys_file = CONFIG_DIR / "keys.json"
    if not keys_file.exists():
        return {}
    try:
        return json.loads(keys_file.read_text())
    except Exception:
        return {}


def _fetch_reddit_praw(subreddit: str, limit: int = 50, used_ids: set | None = None) -> dict | None:
    """Fetch via Reddit OAuth (PRAW) — works from cloud/server IPs unlike the public JSON API.

    Requires in config/keys.json:
        "reddit_client_id":     "<your script app client_id>"
        "reddit_client_secret": "<your script app client_secret>"

    Create a free app at https://www.reddit.com/prefs/apps (choose type: script).
    """
    if not _PRAW_AVAILABLE:
        return None
    keys = _load_keys()
    cid  = keys.get("reddit_client_id", "")
    csec = keys.get("reddit_client_secret", "")
    if not cid or cid.startswith("YOUR_"):
        return None

    if used_ids is None:
        used_ids = set()
    try:
        reddit = _praw_mod.Reddit(
            client_id=cid,
            client_secret=csec,
            user_agent="script:rufus.shorts:v1.0",
        )
        candidates = []
        for post in reddit.subreddit(subreddit).top("week", limit=limit):
            if post.stickied or post.over_18:
                continue
            if post.score < MIN_SCORE or post.num_comments < MIN_COMMENTS:
                continue
            body = post.selftext or ""
            if not body or len(body) < MIN_BODY_LEN or len(body) > MAX_BODY_LEN:
                continue
            title = post.title
            if TITLE_BAD_RE.search(title) or TITLE_DISCUSSION_RE.search(title):
                continue
            if TITLE_OFFTOPIC_RE.search(title):
                continue
            if not TITLE_STORY_RE.search(title):
                continue
            sid = "reddit:" + f"https://reddit.com{post.permalink}"
            if sid in used_ids:
                continue
            candidates.append(post)
        if not candidates:
            return None
        chosen = random.choice(candidates[:8])
        return {
            "type":    "reddit",
            "source":  f"r/{subreddit}",
            "title":   chosen.title,
            "content": _clean_text(chosen.selftext),
            "url":     f"https://reddit.com{chosen.permalink}",
        }
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "INVALID_GRANT" in err or "Unauthorized" in err:
            print(f"[research] PRAW auth failed for r/{subreddit} — check reddit_client_id/"
                  f"reddit_client_secret in config/keys.json: {e}")
        else:
            print(f"[research] PRAW error for r/{subreddit}: {e}")
        return None


def fetch_reddit_story(subreddit: str, limit: int = 50, used_ids: set | None = None) -> dict | None:
    """Fetch the first quality-filtered hot post from a subreddit, skipping seen IDs.

    Tries PRAW (official OAuth API — works from cloud IPs) first, then falls back
    to the public JSON endpoints (blocked by Reddit on most cloud IPs).
    """
    if used_ids is None:
        used_ids = set()

    # Official OAuth path — doesn't get blocked from server IPs
    result = _fetch_reddit_praw(subreddit, limit, used_ids)
    if result:
        return result

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
    quality = [
        p for p in posts
        if _passes_quality_filter(p)
        and _post_seed_id(p.get("data", {})) not in used_ids
    ]
    if not quality:
        return None

    chosen = random.choice(quality[:8])
    d = chosen["data"]
    return {
        "type":    "reddit",
        "source":  f"r/{subreddit}",
        "title":   d.get("title", ""),
        "content": _clean_text(d.get("selftext", "")),
        "url":     f"https://reddit.com{d.get('permalink', '')}",
    }


def fetch_hackernews_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Fetch a substantive Ask HN post relevant to the niche, skipping seen IDs.

    Uses the public HN Algolia API (no auth required). Only returns posts where
    story_text is present — that means Ask HN posts and self-posts with real content.
    """
    if used_ids is None:
        used_ids = set()

    query = HN_NICHE_QUERIES.get(niche_name)
    if not query:
        return None

    q_enc = query.replace(" ", "+")
    numeric = f"points>{HN_MIN_POINTS},num_comments>{HN_MIN_COMMENTS}"
    endpoints = [
        f"https://hn.algolia.com/api/v1/search?tags=ask_hn&query={q_enc}&numericFilters={numeric}&hitsPerPage=40",
        f"https://hn.algolia.com/api/v1/search?tags=story&query={q_enc}&numericFilters={numeric}&hitsPerPage=40",
    ]

    all_hits: list = []
    for url in endpoints:
        try:
            r = httpx.get(url, headers=REDDIT_HEADERS, timeout=HN_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            all_hits.extend(r.json().get("hits", []))
        except Exception as e:
            print(f"[research] HN fetch error: {e}")

    if not all_hits:
        print(f"[research] HN: no hits returned for niche '{niche_name}'")
        return None

    def _hn_url(hit: dict) -> str:
        return f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"

    def _hn_item_id(hit: dict) -> str:
        return "hn:" + _hn_url(hit)

    # Filter: must have enough real text content, must not be already used
    quality = [
        h for h in all_hits
        if h.get("story_text")
        and HN_MIN_TEXT_LEN <= len(h.get("story_text", "")) <= HN_MAX_TEXT_LEN
        and _hn_item_id(h) not in used_ids
    ]
    if not quality:
        print(f"[research] HN: {len(all_hits)} hits but none passed quality filter (points>{HN_MIN_POINTS}, comments>{HN_MIN_COMMENTS}, text {HN_MIN_TEXT_LEN}-{HN_MAX_TEXT_LEN} chars, not already used)")
        return None

    # Deduplicate by objectID (both endpoints can return same story)
    seen_oids: set = set()
    unique = []
    for h in quality:
        oid = h.get("objectID")
        if oid not in seen_oids:
            seen_oids.add(oid)
            unique.append(h)

    chosen = random.choice(unique[:8])
    return {
        "type":    "hackernews",
        "source":  "Hacker News",
        "title":   chosen.get("title", ""),
        "content": _clean_text(_strip_html(chosen.get("story_text", ""))),
        "url":     _hn_url(chosen),
    }


def pick_wisdom_quote(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Pick one random quote from the niche's curated pool, skipping seen ones."""
    if used_ids is None:
        used_ids = set()
    f = WISDOM_DIR / f"{niche_name}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except Exception as e:
        print(f"[research] WARNING: could not parse {f.name}: {e}")
        return None
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

    Order: Reddit-first (story-filtered), Hacker News second (intellectual niches),
    wisdom fallback. All sources skip anything already in used_seeds.json
    (last MAX_USED_HISTORY items).
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

    # Try Hacker News for niches that align with its intellectual/founder audience
    seed = fetch_hackernews_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using HN story: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        return seed

    # No live story passed filters — fall back to wisdom quote pool
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
