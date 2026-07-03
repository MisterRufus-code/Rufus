#!/usr/bin/env python3
"""
research.py – Returns real source material the script writer compresses into a Short.

Strategy:
    1. Try Reddit hot/top posts from niche-specific subreddits
    2. Try StackExchange high-voted questions (keyless API — money/workplace stories)
    3. Try RSS/Atom feeds per niche (free public feeds — no auth required)
    4. Try Hacker News "Ask HN" posts (finance/business/mindset niches)
    5. If nothing passes quality filters, fall back to a curated wisdom quote pool

A Seed dict is returned with keys:
    - type:    "reddit" | "stackexchange" | "rss" | "hackernews" | "wisdom"
    - content: the substantive text (story or quote)
    - source:  subreddit name | SE site | domain | "Hacker News" | author name
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
from filelock import FileLock

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

# How many top-ranked candidates each source samples from. All sources are
# already quality-gated (score thresholds + story filters), so the posts in
# positions 9-20 are still strong — sampling from a wider window means more
# distinct seeds before the dedup history exhausts a source and we fall back.
# Bigger pool = fresher videos day over day.
SAMPLE_POOL = 20

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

# Quality thresholds for Reddit posts — per-subreddit because large subs
# (personalfinance) produce high-score posts while smaller ones (povertyfinance)
# max out around 200 but still contain gold. Flat 500 rejects too much.
SUBREDDIT_MIN_SCORE: dict[str, int] = {
    "personalfinance":        500,
    "financialindependence":  400,
    "FIRE":                   400,
    "povertyfinance":         150,
    "Frugal":                 200,
    "GetDisciplined":         250,
    "DecidingToBeBetter":     150,
    "selfimprovement":        200,
    "Stoicism":               200,
    "philosophy":             100,
    "psychology":             150,
    "Entrepreneur":           300,
    "startups":               250,
    "smallbusiness":          150,
    "EntrepreneurRideAlong":  150,
    "productivity":           200,
    "getdisciplined":         200,
}
DEFAULT_MIN_SCORE = 300   # fallback for any sub not in the dict above
MIN_SCORE         = DEFAULT_MIN_SCORE   # kept for legacy callers
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

# ── StackExchange config ─────────────────────────────────────────────────────────

SE_TIMEOUT      = 10.0
SE_MIN_SCORE    = 40     # SE scores run lower than Reddit — 40+ is a strong question
SE_MIN_BODY_LEN = 300
SE_MAX_BODY_LEN = 3000

# Niche → StackExchange site. Only sites whose top questions read as real
# first-person stories ("My employer did X…", "I inherited Y…").
SE_NICHE_SITES = {
    "finance":              "money",
    "business":             "workplace",
    "personal_development": "workplace",
    "mindset":              None,
    "motivation":           None,
    "money_history":        "history",
}


# ── RSS feed config ──────────────────────────────────────────────────────────────

RSS_TIMEOUT = 10.0
RSS_MIN_DESC_LEN = 100

# Finance/psychology keywords that pass quality filter even without TITLE_STORY_RE
RSS_FINANCE_PSYCH_KEYWORDS_RE = re.compile(
    r"\b(invest|stock|market|fund|portfolio|wealth|budget|saving|debt|"
    r"earn|income|profit|loss|return|compoun|interest|bank|tax|"
    r"mindset|habit|psycholog|cognitive|behavio|motivat|discipline|"
    r"productivity|focus|success|goal|growth)\b",
    re.IGNORECASE,
)

RSS_STORY_RE = re.compile(
    r"(\$|\d|"
    r"\bhow\s+(?:i|we|one)\b|"
    r"\bwhy\s+(?:i|we|your)\b|"
    r"\bstop\b|"
    r"\bworking\b|\bfail\b|\bfailed\b|\bwin\b|\bwon\b|"
    r"\bboost\b|\bgrow\b|\blose\b|\bsave\b|\bearn\b|"
    r"\bmistake\b|\bsecret\b|\brule\b|\bprinciple\b|"
    r"\btrick\b|\bstrategy\b|\blesson\b|"
    r"\bbest\b|\bworst\b|\bnever\b|\balways\b)",
    re.IGNORECASE,
)

RSS_FEEDS = {
    "finance": [
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "https://www.theguardian.com/money/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/YourMoney.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.investopedia.com/feeds/rss.aspx",
    ],
    "business": [
        "https://feeds.hbr.org/harvardbusiness",
        "https://www.inc.com/rss",
        "https://techcrunch.com/rss/",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
    "mindset": [
        "https://fs.blog/feed/",
        "https://bigthink.com/feed/",
        "https://www.psychologytoday.com/us/articles/rss",
    ],
    "motivation": [
        "https://jamesclear.com/feed",
        "https://www.success.com/feed/",
        "https://markmanson.net/feed",
    ],
    "personal_development": [
        "https://jamesclear.com/feed",
        "https://calnewport.com/blog/feed/",
        "https://fs.blog/feed/",
    ],
    "money_history": [
        "https://www.smithsonianmag.com/rss/history/",
        "https://daily.jstor.org/feed/",
        "https://aeon.co/feed.rss",
    ],
}


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
    "money_history":       "history of money gold standard inflation currency collapse",
}


# ── Trending signal (pytrends — graceful if not installed) ──────────────────────

# Seed keywords per niche — Google Trends needs 1-5 related terms to find rising queries
NICHE_TREND_SEEDS: dict[str, list[str]] = {
    "finance":             ["investing", "stock market", "personal finance", "debt", "savings"],
    "business":            ["startup", "entrepreneurship", "side hustle", "business strategy"],
    "mindset":             ["mindset", "mental health", "self improvement", "habits"],
    "motivation":          ["motivation", "discipline", "success mindset", "goal setting"],
    "personal_development":["productivity", "learning", "self improvement", "daily habits"],
    "money_history":       ["history of money", "gold standard", "hyperinflation", "roman empire economy"],
}


def get_trending_context(niche_name: str) -> str | None:
    """Return a comma-separated string of rising Google Trends queries for this niche.

    Uses pytrends (unofficial Google Trends scraper — pip install pytrends).
    Returns None gracefully if not installed, rate-limited, or any failure.
    Injected into pre-analysis so hooks can reference what people are actively
    searching this week — the single most reliable signal of viral potential.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return None

    seeds = NICHE_TREND_SEEDS.get(niche_name)
    if not seeds:
        return None

    try:
        pt = TrendReq(hl="en-US", tz=300, timeout=(5, 15))
        pt.build_payload(seeds[:5], timeframe="now 7-d", geo="US")
        related = pt.related_queries()

        trending: list[str] = []
        for kw in seeds[:3]:
            if kw in related:
                df = related[kw].get("rising")
                if df is not None and not df.empty:
                    trending.extend(df["query"].head(3).tolist())

        if not trending:
            # Fall back to top queries if no rising data
            for kw in seeds[:3]:
                if kw in related:
                    df = related[kw].get("top")
                    if df is not None and not df.empty:
                        trending.extend(df["query"].head(2).tolist())

        if not trending:
            return None

        # Deduplicate, remove generic single-word terms, cap at 5
        seen: set[str] = set()
        unique: list[str] = []
        for t in trending:
            t_norm = t.strip().lower()
            if len(t_norm) > 4 and t_norm not in seen:
                seen.add(t_norm)
                unique.append(t.strip())

        result = ", ".join(unique[:5])
        print(f"[research] Google Trends ({niche_name}): {result}")
        return result

    except Exception as e:
        print(f"[research] pytrends failed (non-fatal): {e}")
        return None


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
    if t == "stackexchange":
        return "se:" + (seed.get("url") or seed.get("title", ""))
    if t == "rss":
        return "rss:" + (seed.get("url") or seed.get("title", ""))
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
    USED_SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Cross-platform advisory lock (Windows + Linux) — filelock instead of POSIX
    # fcntl, which doesn't exist on Windows.
    with FileLock(str(USED_SEEDS_FILE.with_suffix(".lock"))):
        used = _load_used_seeds()
        if sid in used:
            used.remove(sid)
        used.append(sid)
        used = used[-MAX_USED_HISTORY:]
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


def _passes_quality_filter(post: dict, subreddit: str = "") -> bool:
    """Return True if a Reddit post has enough substance to be worth using.

    Requirements:
      - Engagement: score >= per-subreddit threshold, num_comments >= MIN_COMMENTS
      - Substance: body length within MIN_BODY_LEN..MAX_BODY_LEN
      - Title contains a STORY signal (dollar sign, number, or past-tense action verb)
      - Title does NOT match BAD patterns (rage/drama/mod)
      - Title does NOT match DISCUSSION patterns (no story arc — bad for Shorts)
    """
    d = post.get("data", {})
    if d.get("stickied") or d.get("over_18") or d.get("removed_by_category"):
        return False
    min_score = SUBREDDIT_MIN_SCORE.get(subreddit, DEFAULT_MIN_SCORE)
    if d.get("score", 0) < min_score:
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
        min_score_sub = SUBREDDIT_MIN_SCORE.get(subreddit, DEFAULT_MIN_SCORE)
        for post in reddit.subreddit(subreddit).top("week", limit=limit):
            if post.stickied or post.over_18:
                continue
            if post.score < min_score_sub or post.num_comments < MIN_COMMENTS:
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
        chosen = random.choice(candidates[:SAMPLE_POOL])
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
        if _passes_quality_filter(p, subreddit=subreddit)
        and _post_seed_id(p.get("data", {})) not in used_ids
    ]
    if not quality:
        return None

    chosen = random.choice(quality[:SAMPLE_POOL])
    d = chosen["data"]
    return {
        "type":    "reddit",
        "source":  f"r/{subreddit}",
        "title":   d.get("title", ""),
        "content": _clean_text(d.get("selftext", "")),
        "url":     f"https://reddit.com{d.get('permalink', '')}",
    }


def fetch_stackexchange_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Fetch a high-voted story-shaped question from the niche's SE site.

    The StackExchange API allows keyless access (IP rate limit ~300 req/day —
    Rufus uses 1 per run). money.SE and workplace.SE top questions are mostly
    first-person stories, which is exactly what the script writer needs.
    """
    if used_ids is None:
        used_ids = set()

    site = SE_NICHE_SITES.get(niche_name)
    if not site:
        return None

    url = (
        "https://api.stackexchange.com/2.3/questions"
        f"?order=desc&sort=votes&site={site}&filter=withbody&pagesize=60"
    )
    try:
        r = httpx.get(url, headers=REDDIT_HEADERS, timeout=SE_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        print(f"[research] StackExchange unreachable for {site} ({e})")
        return None

    quality = []
    for q in items:
        if q.get("score", 0) < SE_MIN_SCORE:
            continue
        body = _strip_html(q.get("body", ""))
        if not (SE_MIN_BODY_LEN <= len(body) <= SE_MAX_BODY_LEN):
            continue
        title = _strip_html(q.get("title", ""))
        if TITLE_BAD_RE.search(title) or TITLE_OFFTOPIC_RE.search(title):
            continue
        if not TITLE_STORY_RE.search(title):
            continue
        link = q.get("link", "")
        if "se:" + link in used_ids:
            continue
        quality.append((title, body, link))

    if not quality:
        print(f"[research] SE {site}: {len(items)} items, none passed quality filter")
        return None

    title, body, link = random.choice(quality[:SAMPLE_POOL])
    return {
        "type":    "stackexchange",
        "source":  f"{site}.stackexchange.com",
        "title":   title,
        "content": _clean_text(body),
        "url":     link,
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

    chosen = random.choice(unique[:SAMPLE_POOL])
    return {
        "type":    "hackernews",
        "source":  "Hacker News",
        "title":   chosen.get("title", ""),
        "content": _clean_text(_strip_html(chosen.get("story_text", ""))),
        "url":     _hn_url(chosen),
    }


def fetch_rss_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Fetch a quality-filtered article from RSS/Atom feeds for the given niche.

    Tries each feed URL in order, parsing both RSS 2.0 (<channel><item>) and
    Atom (<feed><entry>) formats using stdlib xml.etree.ElementTree. All network
    and parse errors are swallowed with a warning print — never raises.
    """
    import xml.etree.ElementTree as ET

    if used_ids is None:
        used_ids = set()

    feeds = RSS_FEEDS.get(niche_name)
    if not feeds:
        return None

    feed_list = list(feeds)
    random.shuffle(feed_list)

    # Atom namespace prefix used in <link href="..."/> elements
    ATOM_NS = "http://www.w3.org/2005/Atom"

    def _parse_rss_items(root: ET.Element) -> list[dict]:
        """Extract items from RSS 2.0 <channel><item> structure."""
        items = []
        channel = root.find("channel")
        if channel is None:
            return items
        for item in channel.findall("item"):
            title_el = item.find("title")
            desc_el  = item.find("description")
            link_el  = item.find("link")
            title = (title_el.text or "").strip() if title_el is not None else ""
            desc  = (desc_el.text  or "").strip() if desc_el  is not None else ""
            link  = (link_el.text  or "").strip() if link_el  is not None else ""
            items.append({"title": title, "description": desc, "link": link})
        return items

    def _parse_atom_items(root: ET.Element) -> list[dict]:
        """Extract entries from Atom <feed><entry> structure."""
        items = []
        ns = {"atom": ATOM_NS}
        for entry in root.findall("atom:entry", ns):
            title_el   = entry.find("atom:title",   ns)
            summary_el = entry.find("atom:summary", ns)
            link_el    = entry.find("atom:link",    ns)
            title = (title_el.text   or "").strip() if title_el   is not None else ""
            desc  = (summary_el.text or "").strip() if summary_el is not None else ""
            link  = ""
            if link_el is not None:
                link = link_el.get("href", "") or (link_el.text or "").strip()
            items.append({"title": title, "description": desc, "link": link})
        return items

    for feed_url in feed_list:
        try:
            r = httpx.get(feed_url, headers=REDDIT_HEADERS, timeout=RSS_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            raw_xml = r.text
        except Exception as e:
            print(f"[research] RSS fetch warning — {feed_url}: {e}")
            continue

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            print(f"[research] RSS parse warning — {feed_url}: {e}")
            continue

        # Detect format: RSS 2.0 has <channel>, Atom has tag ending with "feed"
        tag = root.tag.lower()
        if "feed" in tag:
            raw_items = _parse_atom_items(root)
        else:
            raw_items = _parse_rss_items(root)

        if not raw_items:
            continue

        # Extract domain for the source field
        try:
            from urllib.parse import urlparse
            domain = urlparse(feed_url).netloc
        except Exception:
            domain = feed_url

        candidates = []
        for item in raw_items:
            title = _clean_text(_strip_html(item["title"]))
            desc  = _clean_text(_strip_html(item["description"]))
            link  = item["link"].strip()

            if not title or not link:
                continue
            if len(desc) < RSS_MIN_DESC_LEN:
                continue
            if TITLE_BAD_RE.search(title):
                continue
            if TITLE_DISCUSSION_RE.search(title):
                continue
            if TITLE_OFFTOPIC_RE.search(title):
                continue
            # Title must pass story RE OR contain finance/psych keywords
            if not RSS_STORY_RE.search(title) and not RSS_FINANCE_PSYCH_KEYWORDS_RE.search(title):
                continue
            sid = "rss:" + link
            if sid in used_ids:
                continue
            candidates.append((title, desc, link))

        if not candidates:
            continue

        # Prefer items with narrative signals (first-person story, lesson, failure)
        # over pure-news articles so the script writer gets richer material.
        NARRATIVE_SIGNALS = [
            "failed", "mistake", "lost", "lesson", "realized",
            "discovered", "decided", "why i", "how i", "years ago",
            "never expected", "changed everything", "what i learned",
            "confession", "secret", "worst", "best decision",
        ]
        narrative = [
            c for c in candidates
            if any(s in (c[0] + " " + c[1]).lower() for s in NARRATIVE_SIGNALS)
        ]
        pool = narrative if narrative else candidates

        title, desc, link = random.choice(pool[:SAMPLE_POOL])
        return {
            "type":    "rss",
            "source":  domain,
            "title":   title,
            "content": desc,
            "url":     link,
        }

    print(f"[research] RSS: no qualifying items found for niche '{niche_name}'")
    return None


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
    # Author-uniform sampling: pick an AUTHOR first, then one of their quotes.
    # Plain random.choice over the pool lets over-represented authors (e.g.
    # Buffett is ~28% of the finance pool) dominate consecutive videos.
    by_author: dict = {}
    for q in pool:
        by_author.setdefault(q.get("author", "Unknown"), []).append(q)
    q = random.choice(by_author[random.choice(list(by_author))])
    return {
        "type":    "wisdom",
        "source":  q.get("author", "Unknown"),
        "title":   "",
        "content": q.get("text", ""),
        "url":     "",
    }


def get_seed(niche_name: str | None = None) -> dict:
    """Get a seed for the script writer. Tracks history so seeds never repeat.

    Order: Reddit → StackExchange → RSS → Hacker News → wisdom fallback.
    All sources skip anything already in used_seeds.json (last MAX_USED_HISTORY items).
    """
    niche, active = _load_niche()
    name      = niche_name or active
    used_list = _load_used_seeds()
    used_set  = set(used_list)

    print(f"[research] history: {len(used_list)} prior seeds will be skipped")

    # Trending context — what people are actively searching this week.
    # Injected into the seed so pre-analysis can build a timely hook.
    trending_context = get_trending_context(name)

    subreddits = list(niche.get("subreddits", []))
    random.shuffle(subreddits)

    def _with_trending(s: dict) -> dict:
        """Attach trending_context to a seed dict so script_writer can use it."""
        if trending_context:
            s["trending_context"] = trending_context
        return s

    for sub in subreddits:
        seed = fetch_reddit_story(sub, used_ids=used_set)
        if seed:
            print(f"[research] using Reddit story from {seed['source']}: \"{seed['title'][:60]}\"")
            _mark_seed_used(seed)
            return _with_trending(seed)

    # StackExchange: keyless API, never IP-blocked like Reddit's public JSON
    seed = fetch_stackexchange_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using StackExchange story from {seed['source']}: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        return _with_trending(seed)

    # RSS/Atom feeds: free public feeds, no auth required
    seed = fetch_rss_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using RSS story from {seed['source']}: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        return _with_trending(seed)

    # Try Hacker News for niches that align with its intellectual/founder audience
    seed = fetch_hackernews_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using HN story: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        return _with_trending(seed)

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
    return _with_trending(fallback)


if __name__ == "__main__":
    niche = sys.argv[1] if len(sys.argv) > 1 else None
    seed  = get_seed(niche)
    print(json.dumps(seed, indent=2))
