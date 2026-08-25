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
from urllib.parse import quote
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
# Wikipedia's REST API rejects/deprioritizes browser-spoofed User-Agents (the
# Chrome UA above works for Reddit/HN scraping but gets a 403 from Wikipedia) —
# their API etiquette policy requires an identifying UA instead:
# https://meta.wikimedia.org/wiki/User-Agent_policy
WIKI_HEADERS = {
    "User-Agent": "Rufus-ContentBot/1.0 (https://github.com/MisterRufus-code/Rufus; "
                  "automated research for a YouTube Shorts pipeline) httpx",
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

# The threshold above was calibrated against `/questions?sort=votes`, which
# returns a site's ALL-TIME TOP list — where 40+ is ordinary. Rotating to a
# topical `search/advanced` query fixed the frozen-input bug and immediately
# exposed a second one: a topical search returns the best questions ABOUT
# banking, not the best questions on the site, and those score 5-30. The first
# live run after the rotation landed shows it exactly:
#
#     SE history ["banking", page 1]: 46 items, none usable (45 score, 1 length)
#
# 45 of 46 thrown away on a number that no longer describes the population it
# filters. Fixing a frozen query while leaving its threshold alone just moves
# where the source dies. A topical hit is already evidence of relevance, which
# the top-list has no way to provide, so it can afford a much lower bar.
SE_MIN_SCORE_TOPICAL = 8
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

# money.SE and workplace.SE are inherently on-topic — every question there is
# about money or work by definition. history.stackexchange.com is NOT: it's a
# general history site (Vikings, wars, politics, anything), so without a topic
# filter it surfaces off-topic questions like "how would a Viking curse someone"
# for a niche that's specifically about monetary/economic history. Niches whose
# SE site needs this extra check go here; sites that are inherently on-topic
# are simply absent (no filter applied).
# Free-text queries used to ASK StackExchange for on-topic questions, instead
# of asking for the site's all-time top 60 and hoping some of them are about
# money. See _se_url for why that distinction decided whether this source
# worked at all.
SE_TOPIC_QUERIES = {
    "money_history": (
        "currency debasement", "coinage", "hyperinflation", "taxation",
        "trade route", "banking", "wages", "mint", "silver", "debt",
        "tribute", "price of grain", "merchant", "treasury",
    ),
}

# How many pages deep to rotate. Page 1 alone is a fixed set; rotating gives a
# different pool run to run, which is what keeps a daily channel from
# re-reading the same questions forever.
SE_MAX_PAGE = 4


SE_TOPIC_FILTER_RE = {
    "money_history": re.compile(
        r"\b(money|coin|currency|gold|silver|mint(?:ed|ing)?|"
        r"(?:hyper|de)?inflation\w*|"
        r"tax(?:es|ation)?|bank(?:ing)?|trade|econom\w*|debt|wage|price|wealth|"
        r"fortune|treasure|market|merchant|commerce|loan|interest|credit|"
        r"financ\w*|monetary|tribute|toll|tariff|ransom)\b",
        re.IGNORECASE,
    ),
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


def _trending_queries(niche_name: str) -> list[str]:
    """Rising Google Trends queries for this niche, as a list (deduped, generic
    single words dropped, capped at 5). [] if pytrends isn't installed, is
    rate-limited, the niche has no trend seeds, or anything fails — every caller
    treats [] as "no trend signal" and carries on.

    Shared core behind BOTH get_trending_context (prompt flavour) and
    fetch_trending_wikipedia (topic SELECTION), so the two can't drift apart."""
    return trending_queries_with_reason(niche_name)[0]


def trending_queries_with_reason(niche_name: str) -> tuple[list[str], str]:
    """The queries, and WHY there are none when there are none.

    FOUR SITUATIONS, ONE EMPTY LIST. A missing package, a niche with no trend
    seeds, a rate-limited request and a genuinely quiet week all returned [],
    so the Trending page could only say "pytrends not installed, rate-limited,
    or nothing rising this week" — three guesses and a shrug, on a page whose
    whole job is to tell you something. Only the first has a fix the owner can
    act on (`pip install pytrends`), only the third is worth retrying, and the
    fourth is not a problem at all. Reporting them as one sentence means the
    one that needs a command looks exactly like the one that needs nothing.

    The reason is "" when queries came back. Callers that only want the list
    keep using _trending_queries and are unaffected.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        import paths
        return [], (f"pytrends is not installed — run `{paths.pip_hint('pytrends')}` "
                    f"(it is in requirements-optional.txt). Runs still work: "
                    f"research falls back to its own topic chain.")

    seeds = NICHE_TREND_SEEDS.get(niche_name)
    if not seeds:
        return [], (f"no trend seeds are configured for {niche_name} — add "
                    f"them to research.NICHE_TREND_SEEDS")

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

        # Deduplicate, remove generic single-word terms, cap at 5
        seen: set[str] = set()
        unique: list[str] = []
        for t in trending:
            t_norm = t.strip().lower()
            if len(t_norm) > 4 and t_norm not in seen:
                seen.add(t_norm)
                unique.append(t.strip())
        if not unique:
            return [], ("Google Trends answered, and nothing is rising for "
                        "these seeds this week. Nothing to fix.")
        return unique[:5], ""

    except Exception as e:
        # Same memo as OpenAlex, and for the same reason: get_seed retries the
        # whole chain up to six times and this is called twice per attempt, so
        # one rate-limited Google was producing twelve requests and twelve
        # identical lines in a run that then failed for an unrelated reason.
        if _looks_rate_limited(e):
            _note_rate_limit("Google Trends")
        else:
            print(f"[research] pytrends failed (non-fatal): {e}")
        return [], (f"the Google Trends request failed: {e}. Usually rate "
                    f"limiting — it clears on its own.")


def get_trending_context(niche_name: str) -> str | None:
    """Comma-separated rising Google Trends queries for this niche, for prompt
    CONTEXT — hooks can reference what people are actively searching this week.
    None if unavailable. See _trending_queries for the raw list, and
    fetch_trending_wikipedia for using trends to pick the topic itself."""
    queries = _trending_queries(niche_name)
    if not queries:
        return None
    result = ", ".join(queries)
    print(f"[research] Google Trends ({niche_name}): {result}")
    return result


def _load_niche():
    data   = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
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
    if t == "openalex":
        # Must match the "oa:" prefix fetch_openalex_story checks against, or
        # every run re-reads the same most-cited paper forever.
        return "oa:" + (seed.get("url") or seed.get("title", ""))
    if t == "rss":
        return "rss:" + (seed.get("url") or seed.get("title", ""))
    if t == "wikipedia":
        return "wiki:" + (seed.get("url") or seed.get("title", ""))
    if t == "newspaper":
        return "news:" + (seed.get("url") or seed.get("title", ""))
    if t == "wisdom":
        text = (seed.get("content") or "").strip().lower()
        return "wisdom:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return ""


def _load_used_seeds() -> list:
    if not USED_SEEDS_FILE.exists():
        return []
    try:
        return json.loads(USED_SEEDS_FILE.read_text(encoding="utf-8"))
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
        USED_SEEDS_FILE.write_text(json.dumps(used, indent=2), encoding="utf-8")


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
        return json.loads(keys_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


_reddit_warned = False


def _warn_reddit_unauthenticated() -> None:
    """Say the fix once per run, not once per subreddit, and not never.

    Reddit stopped serving its .json endpoints to unauthenticated clients. The
    pipeline degrades correctly — it falls through to Wikipedia — but it
    degraded SILENTLY into a one-source funnel, and the log line it printed
    ("reddit unreachable") read like a network blip rather than a permanent
    configuration gap. Five subreddits x every run x months.
    """
    global _reddit_warned
    if _reddit_warned:
        return
    _reddit_warned = True
    print("[research]   Reddit needs OAuth now. Create a free 'script' app at "
          "https://www.reddit.com/prefs/apps and add reddit_client_id + "
          "reddit_client_secret to config/keys.json — that restores 5 seed "
          "sources; without it Wikipedia is the only one left.")


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
        _warn_reddit_unauthenticated()
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
                rejected["offtopic"] += 1
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
        # "unreachable" was the wrong word and it hid the fix for months.
        # Reddit is up; it refuses unauthenticated JSON and returns an HTML
        # block page, which is why the error is always a JSON parse error at
        # "line 2 column 5" and never a timeout. Five subreddits printed five
        # copies of that per run, none of them saying what to do about it.
        print(f"[research] reddit blocked r/{subreddit} — no OAuth credentials "
              f"({last_err})")
        _warn_reddit_unauthenticated()
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


def _se_url(site: str, niche_name: str) -> tuple[str, str]:
    """The StackExchange request, and a human description of it.

    WHY THIS IS NOT `/questions?sort=votes` ANY MORE. That call takes no query
    and no page, so it returns the SAME all-time top 60 questions on every
    single run, forever. For history.SE those 60 are the site's famous
    questions — wars, empires, daily life — and almost none are about money.
    The consequence is not "sometimes no seed": it is that this source could
    never succeed, deterministically, while still costing an HTTP call every
    run. Every log the owner has ever sent shows the same line,
    "SE history: 60 items, none passed quality filter", which is exactly the
    signature of a fixed input meeting a fixed filter.

    So ask the site for what this channel is actually about, and move the pool
    between runs: a topical free-text query where one is defined, and a rotated
    page either way. money.SE / workplace.SE need no query — every question
    there is already on topic — but they still get the page rotation, because
    re-reading the same top 60 for a year is its own dead end.
    """
    page = random.randint(1, SE_MAX_PAGE)
    base = (f"&site={site}&filter=withbody&pagesize=60&page={page}"
            f"&order=desc&sort=votes")
    queries = SE_TOPIC_QUERIES.get(niche_name)
    if queries:
        q = random.choice(queries)
        return (f"https://api.stackexchange.com/2.3/search/advanced?q={quote(q)}"
                + base), f'"{q}", page {page}'
    return f"https://api.stackexchange.com/2.3/questions?{base.lstrip('&')}", \
           f"top questions, page {page}"


def _se_min_score(niche_name: str) -> int:
    """The score bar for whichever query shape _se_url built.

    A topical search and an all-time-top list are different populations, and
    one threshold cannot serve both — see SE_MIN_SCORE_TOPICAL for the run that
    proved it.
    """
    return (SE_MIN_SCORE_TOPICAL if SE_TOPIC_QUERIES.get(niche_name)
            else SE_MIN_SCORE)


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
    topic_re = SE_TOPIC_FILTER_RE.get(niche_name)

    url, described = _se_url(site, niche_name)
    try:
        r = httpx.get(url, headers=REDDIT_HEADERS, timeout=SE_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        print(f"[research] StackExchange unreachable for {site} ({e})")
        return None

    min_score = _se_min_score(niche_name)
    rejected = {"score": 0, "length": 0, "title": 0, "offtopic": 0, "seen": 0}
    quality = []
    for q in items:
        if q.get("score", 0) < min_score:
            rejected["score"] += 1
            continue
        body = _strip_html(q.get("body", ""))
        if not (SE_MIN_BODY_LEN <= len(body) <= SE_MAX_BODY_LEN):
            rejected["length"] += 1
            continue
        title = _strip_html(q.get("title", ""))
        if TITLE_BAD_RE.search(title) or TITLE_OFFTOPIC_RE.search(title):
            rejected["title"] += 1
            continue
        if topic_re:
            # A general-purpose SE site (history.SE covers Vikings, wars,
            # politics — anything) gets its "is this actually on-topic"
            # signal from topic_re, not from TITLE_STORY_RE. Requiring BOTH
            # was the bug: TITLE_STORY_RE wants a first-person narrative
            # shape ("I lost $50k when…", a dollar sign, "years ago") that
            # money.SE/workplace.SE questions genuinely have and academic
            # history questions structurally don't — "Why did Roman denarii
            # devalue?" is exactly the on-topic, well-sourced material this
            # niche needs, and it fails TITLE_STORY_RE every time. Live
            # symptom: history.SE returned 0 passing items out of 60 in
            # every single logged run — not a strict filter, a mismatched
            # one, blocking a source that already costs nothing and is
            # never IP-blocked.
            if not (topic_re.search(title) or topic_re.search(body)):
                rejected["offtopic"] += 1
                continue
        else:
            # money.SE / workplace.SE are inherently on-topic (every
            # question there IS about money/work), so narrative shape is
            # the only quality signal available — stays required here,
            # unchanged from before.
            if not TITLE_STORY_RE.search(title):
                continue
        link = q.get("link", "")
        if "se:" + link in used_ids:
            rejected["seen"] += 1
            continue
        quality.append((title, body, link))

    if not quality:
        # Name the reason, not just the outcome. "none passed quality filter"
        # looks like bad luck; "60 items, 58 off-topic" is a dead source, and
        # the difference between those two readings cost months of runs.
        why = ", ".join(f"{n} {k}" for k, n in rejected.items() if n)
        print(f"[research] SE {site} [{described}]: {len(items)} items, "
              f"none usable ({why or 'no items'})")
        return None
    print(f"[research] SE {site} [{described}]: {len(quality)} of {len(items)} usable")

    title, body, link = random.choice(quality[:SAMPLE_POOL])
    return {
        "type":    "stackexchange",
        "source":  f"{site}.stackexchange.com",
        "title":   title,
        "content": _clean_text(body),
        "url":     link,
    }


# ── OpenAlex: peer-reviewed papers as source material ───────────────────────
#
# WHY A SCHOLARLY SOURCE, and why it is first. Every fact-gate rejection in the
# owner's log is one of two sentences — "not supported by the source, which
# does not provide this specific figure" and "MIND-READ: the script claims the
# government rigged the game". Both come from the same place: the writer must
# open on a dated moment with a named person and a hard number, and a
# StackExchange thread ("In what ways was the Gold Confiscation Act
# beneficial") contains an argument instead of an event. It cannot satisfy its
# rules from that material, so it invents the specifics and the gate correctly
# kills the result.
#
# An abstract is the opposite shape. It states a year, names authors, and
# carries the study's own figures, because that is what an abstract is for.
# supervisor.groundability() asks for exactly those three things, so papers
# pass the check the discussion threads fail — not by loosening the gate, but
# by feeding it material a story can actually be built from.
#
# OpenAlex is free, keyless, and asks only for a mailto in the query string as
# a courtesy (the "polite pool", which gets better rate limits). 250M works.
OPENALEX_URL     = "https://api.openalex.org/works"
OPENALEX_TIMEOUT = 12.0
OPENALEX_PER_PAGE = 25
OPENALEX_MAX_PAGE = 4

# An abstract shorter than this is a stub — a title restated, or a
# publisher's placeholder — and cannot ground a script.
OPENALEX_MIN_ABSTRACT = 400


def _openalex_enabled() -> bool:
    return os.environ.get("RUFUS_OPENALEX", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _openalex_mailto() -> str:
    """The polite-pool contact. OpenAlex asks for one and rewards it with
    higher rate limits; it is not authentication and nothing breaks without
    it."""
    return (os.environ.get("RUFUS_OPENALEX_MAILTO") or "").strip()


def _abstract_from_inverted_index(index: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}, not as text.

    A licensing artefact rather than a design one — the inverted index is not
    considered a reproduction of the abstract. Rebuilding it is the documented
    and intended use, and it is exact: every position is known, so the result
    is the abstract verbatim rather than an approximation.
    """
    if not isinstance(index, dict) or not index:
        return ""
    slots: dict[int, str] = {}
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                slots[pos] = word
    if not slots:
        return ""
    return " ".join(slots[i] for i in sorted(slots))



# A source with no history in it cannot support a history script.
#
# WHAT THIS COST, MEASURED. Four consecutive scripts scored 3, 3, 4, 4 against
# a target of 7, every one of them SPECIFICITY 0/3, every one held by the fact
# gate. Their seeds were the most-cited OpenAlex papers on "debt", "banking",
# "wages" and "taxation" — Berger & Ofek on diversification and firm value,
# Leland on corporate debt and capital structure, Schneider & Enste on the
# shadow economy, Autor on labour's share. Modern econometrics, all of it, and
# not one historical event between them.
#
# The writer is told to produce money HISTORY. Handed a paper about capital
# structure it does the only thing it can: it invents a history — Spanish
# silver, Nixon, Weimar — and the rubric then correctly gives it 0 for using
# no details from its source. Three full cycles burn on a premise that was
# lost at fetch time.
#
# The fact gate already knew. It said so, per script, in words: "not present
# in the provided source material". This moves that knowledge upstream to the
# only place it can prevent the spend instead of describing it — the same
# shape as every "the gate knew something the generator was never told" bug
# this repo has fixed.
#
# DELIBERATELY LENIENT. It asks for one historical marker anywhere in the
# title or abstract, not for a history journal: a pre-1960 year, or one of a
# handful of period words. "The price revolution in sixteenth-century Spain"
# passes on "century"; "we estimate diversification's effect on firm value"
# has neither and is exactly what should be dropped.
# TOO LOOSE AND TOO TIGHT AT ONCE, which is why it was worth measuring rather
# than trusting. Loose: an astronomy paper passed on "the history of cosmic
# expansion". Tight: it rejected "Byzantine coinage under Justinian" and
# "assay of the denarius from 64 AD" — the exact sources this channel exists
# for — because no named era matched and a two-digit year is not three digits.
# The loose half is now _is_on_subject's job; this half only has to know when.
_HISTORY_WORD_RE = re.compile(
    r"\b(histor\w+|century|centuries|medieval|mediaeval|ancient|antiquity|"
    r"dynast\w+|empire|imperial|colonial|pre-?modern|early modern|"
    r"archiv\w+|BCE?|AD\d|renaissance|reformation|"
    # The eras a monetary history actually runs through. Each names a period,
    # not a subject, so none of them can let a modern paper in on its own.
    r"roman|greek|hellenistic|byzantine|ottoman|mughal|carolingian|"
    r"sumerian|babylonian|mesopotamian|phoenician|feudal|victorian|"
    r"edwardian|interwar|antebellum|classical era|middle ages)\b",
    re.IGNORECASE)
# A bare 3-4 digit number, OR any number followed by an era marker — "64 AD",
# "300 BCE", "1971 CE". Without the second form every date before 100 AD read
# as no date at all.
_HISTORY_YEAR_RE = re.compile(r"\b(1[0-9]{3}|[1-9][0-9]{2})\b")
_HISTORY_ERA_RE = re.compile(r"\b\d{1,4}\s*(?:AD|BC|BCE|CE)\b")


# WHAT THE NICHE IS ABOUT, not just WHEN it happened. _is_historical answers
# "is this the past" and nothing more, which is half the question a niche named
# money_history is asking. A live run seeded on:
#
#   [tribute] → "Erlotinib in Previously Treated Non-Small-Cell Lung Cancer"
#   [silver]  → "Type Ia Supernova Discoveries at z>1"
#   [wages]   → "The productivity paradox of information technology"
#   [coinage] → "A Comprehensive Review of Blockchain Consensus Mechanisms"
#
# Four for four off-topic, and the supernova paper PASSED the history filter on
# the phrase "the history of cosmic expansion". The script then ignored the
# seed and wrote from the model's own knowledge, which the fact gate caught:
# "the source discusses the productivity paradox of information technology,
# not monetary history".
_SUBJECT_WORDS = {
    "money_history": re.compile(
        r"\b(money|monetary|currenc\w+|coin\w*|mint\w*|specie|bullion|"
        r"gold|silver|banknote|paper money|\w*inflation\w*|deflation|"
        r"stagflation|debase\w+|"
        r"bank\w*|credit|debt|loan|interest rate|usury|tax\w*|tariff|tribute|"
        r"trade|merchant|market|price\w*|wage\w*|econom\w+|financ\w+|"
        r"treasur\w+|exchange|barter|commerce|"
        # The money itself, by name. A paper titled "assay of the denarius"
        # says nothing about "money" and is entirely about it — naming the
        # coins is what a filter is for, unlike a prompt, where naming a thing
        # is what draws it.
        r"denarius|denarii|sestertius|aureus|solidus|drachma|obol|stater|"
        r"ducat|florin|guilder|thaler|groat|shilling|sovereign|doubloon|"
        r"cowrie|wampum|tally stick|greenback|reichsmark|rentenmark|"
        r"seigniorage|numismat\w+)\b", re.IGNORECASE),
}


# WHAT A RISING QUERY IS ABOUT WHEN IT SHARES A WORD WITH THIS CHANNEL.
# "gold standard" is a whey protein brand, "hyperinflation" is a lung
# condition, and "<term> synonym" or "<term> definition" is somebody using
# Google as a dictionary — none of which resolve to a story. A subject match
# alone is too weak here because the query is six words, not an abstract: one
# shared noun carries the whole decision.
_TREND_NOISE = {
    "money_history": re.compile(
        r"\b(whey|protein|supplement|creatine|workout|nutrition|calorie|"
        r"lung\w*|pulmonary|respirator\w+|emphysema|"
        r"synonym|antonym|definition|meaning|pronounce|spelling|"
        r"lyrics|recipe|near me|for sale|coupon|discount|review\w*)\b",
        re.IGNORECASE),
}


def _trend_is_usable(query: str, niche_name: str) -> bool:
    """Whether a rising search query is worth resolving into a topic."""
    if not _is_on_subject(query, niche_name):
        return False
    noise = _TREND_NOISE.get(niche_name)
    return not (noise and noise.search(query or ""))


def _is_on_subject(text: str, niche_name: str) -> bool:
    """True if this source is about what the niche is about.

    Fail-open by design: a niche with no pattern here is not filtered at all,
    so adding a niche never silently starves it of seeds.
    """
    pat = _SUBJECT_WORDS.get(niche_name)
    if pat is None:
        return True
    return bool(pat.search(text or ""))


def _is_historical(text: str) -> bool:
    """True if this source is about the past, not about last quarter."""
    if not text:
        return False
    if _HISTORY_WORD_RE.search(text) or _HISTORY_ERA_RE.search(text):
        return True
    for year in _HISTORY_YEAR_RE.findall(text):
        if int(year) < 1960:
            return True
    return False


# A source that has answered 429 IN THIS PROCESS. Asking it again is futile and
# makes the quota worse — and get_seed retries the whole chain up to six times,
# so a rate-limited source was being hit six times per run. Cleared only by a
# new process, which is the right lifetime: a quota that resets in an hour is
# not going to reset inside one run.
_RATE_LIMITED: set[str] = set()

# Said once per process, not once per attempt. Six identical paragraphs about
# the same missing setting is how a real instruction gets scrolled past.
_MAILTO_SAID = False


def _is_rate_limited(source: str) -> bool:
    return source in _RATE_LIMITED


def _note_rate_limit(source: str) -> None:
    """Record a 429 and say what it means, once."""
    if source in _RATE_LIMITED:
        return
    _RATE_LIMITED.add(source)
    print(f"[research] {source} is rate-limited (HTTP 429) — not asking it "
          f"again this run")


def _looks_rate_limited(exc: Exception) -> bool:
    """A 429 hiding inside whatever the client raised.

    Checked on the text rather than the type because httpx raises
    HTTPStatusError from raise_for_status but the pool can raise its own
    classes, and a quota reported as an outage sends the reader looking for a
    network problem that is not there.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or "429" in str(exc)


def _say_missing_mailto() -> None:
    """Name the one setting standing between this channel and OpenAlex's
    higher rate limit.

    OpenAlex runs a POLITE POOL: send a contact address and the limit rises
    sharply. It is not a signup, not a key and not authentication — it is a
    query parameter, and _openalex_mailto already reads the variable for it.
    Nothing has ever said it was empty, so this channel has been in the
    anonymous pool since OpenAlex was added, which is the pool that gets 429.
    """
    global _MAILTO_SAID
    if _MAILTO_SAID:
        return
    _MAILTO_SAID = True
    print("[research] RUFUS_OPENALEX_MAILTO is not set — you are in "
          "OpenAlex's anonymous pool, which is the one that gets rate "
          "limited. Setting it to any email address you read moves you to "
          "their polite pool and raises the limit a lot. It is not a signup "
          "and not a key, just a parameter they ask for.")


def fetch_openalex_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """A cited paper on one of the niche's topics, as a seed.

    Fail-open like every other fetcher: no network, no results, a malformed
    reply — all return None and the chain continues.
    """
    if not _openalex_enabled():
        return None
    if _is_rate_limited("OpenAlex"):
        return None
    if used_ids is None:
        used_ids = set()

    topics = SE_TOPIC_QUERIES.get(niche_name)
    if not topics:
        return None
    # The SAME topic list StackExchange rotates through. One niche, one set of
    # subjects — a second list here would drift from it within a month, and
    # the drift would be invisible.
    query = random.choice(topics)
    page = random.randint(1, OPENALEX_MAX_PAGE)

    params = {
        "search": query,
        # has_abstract is the whole point; without it most results are titles.
        "filter": "has_abstract:true,type:article",
        "per-page": str(OPENALEX_PER_PAGE),
        "page": str(page),
        # Most-cited first: a well-cited paper has been read, checked and
        # argued with, which is a decent proxy for "the facts in it hold up".
        "sort": "cited_by_count:desc",
    }
    if _openalex_mailto():
        params["mailto"] = _openalex_mailto()
    else:
        _say_missing_mailto()

    try:
        r = httpx.get(OPENALEX_URL, params=params, timeout=OPENALEX_TIMEOUT,
                      follow_redirects=True)
        r.raise_for_status()
        works = r.json().get("results", []) or []
    except Exception as e:
        if _looks_rate_limited(e):
            _note_rate_limit("OpenAlex")
        else:
            print(f"[research] OpenAlex unreachable ({e})")
        return None

    rejected = {"no_abstract": 0, "short": 0, "seen": 0, "not history": 0,
                "off subject": 0}
    usable = []
    for w in works:
        # THE SAME KEY _seed_id RECORDS, or nothing is ever seen as used. The
        # first version checked the OpenAlex id here while _seed_id stored the
        # DOI, so the two never matched and the most-cited paper on a topic
        # would have come back every single run — the "136 prior seeds will be
        # skipped" machinery quietly doing nothing for this source.
        wid = str(w.get("doi") or w.get("id") or "")
        if not wid or f"oa:{wid}" in used_ids:
            rejected["seen"] += 1
            continue
        abstract = _abstract_from_inverted_index(w.get("abstract_inverted_index"))
        if not abstract:
            rejected["no_abstract"] += 1
            continue
        if len(abstract) < OPENALEX_MIN_ABSTRACT:
            rejected["short"] += 1
            continue
        blob = f"{w.get('display_name') or ''} {abstract}"
        if not _is_historical(blob):
            rejected["not history"] += 1
            continue
        if not _is_on_subject(blob, niche_name):
            rejected["off subject"] += 1
            continue
        usable.append((w, abstract))

    if not usable:
        why = ", ".join(f"{n} {k}" for k, n in rejected.items() if n)
        print(f"[research] OpenAlex [{query}, page {page}]: {len(works)} works, "
              f"none usable ({why or 'no works'})")
        return None
    print(f"[research] OpenAlex [{query}, page {page}]: "
          f"{len(usable)} of {len(works)} usable")

    work, abstract = random.choice(usable[:SAMPLE_POOL])
    authors = [a.get("author", {}).get("display_name", "")
               for a in (work.get("authorships") or [])[:4]]
    authors = [a for a in authors if a]
    year = work.get("publication_year")
    journal = (((work.get("primary_location") or {}).get("source") or {})
               .get("display_name") or "")

    # THE PROVENANCE GOES INTO THE TEXT, not just into metadata. groundability
    # reads title+content, the fact gate compares the script against content,
    # and the hook's allowed-numbers list is built from the same string — so a
    # year that lives only in a JSON field is a year the writer may not use.
    # Stated as a sentence, it is citable: "In 2015, Darimont and colleagues
    # found..." is a legitimate opening this channel could never write before.
    head = []
    if year:
        head.append(f"Published {year}")
    if journal:
        head.append(f"in {journal}")
    if authors:
        head.append(f"by {', '.join(authors)}")
    prefix = (" ".join(head) + ". ") if head else ""

    return {
        "type":    "openalex",
        "source":  journal or "openalex.org",
        "title":   str(work.get("title") or work.get("display_name") or "").strip(),
        "content": _clean_text(prefix + abstract),
        "url":     work.get("doi") or work.get("id") or "",
    }


WIKI_TOPICS_FILE = CONFIG_DIR / "wiki_topics.json"
WIKI_TIMEOUT       = 10.0
WIKI_MIN_EXTRACT   = 200   # a summary shorter than this can't carry a 45s script
WIKI_MAX_ATTEMPTS  = 8     # bound network fetches per run
# How much full-article text to keep as the grounding corpus. The REST
# /page/summary endpoint returns only the lead paragraph (~1-3 sentences) —
# far too thin to fact-check a 110-word script against, which is why nearly
# every run got capped to 5/10 with "not supported by the source material"
# even when the script was factually fine (observed live across the Doubloon,
# Monte dei Paschi, Manila galleon and Swiss-banking runs). Pulling the real
# article body gives both the writer richer specifics AND the fact gate
# something to actually verify against. Capped so a long article doesn't
# blow up prompt cost.
WIKI_FULLTEXT_CHARS = 6000


def fetch_wikipedia_fulltext(url_title: str) -> str:
    """Plain-text article body (lead + sections), or '' on any failure.

    Uses action=query&prop=extracts&explaintext — the full article, unlike
    the REST summary endpoint's lead-paragraph-only extract. Truncated to
    WIKI_FULLTEXT_CHARS on a paragraph boundary where possible."""
    try:
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "explaintext": "1",
                    "redirects": "1", "format": "json", "titles": url_title.replace("_", " ")},
            headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT, follow_redirects=True,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except Exception:
        return ""
    for page in pages.values():
        body = _clean_text(page.get("extract") or "")
        if len(body) <= WIKI_FULLTEXT_CHARS:
            return body
        cut = body[:WIKI_FULLTEXT_CHARS]
        # Prefer to end on a sentence rather than mid-word.
        stop = max(cut.rfind(". "), cut.rfind("\n"))
        return cut[:stop + 1] if stop > WIKI_FULLTEXT_CHARS // 2 else cut
    return ""

# Auto-replenish: below this many UNUSED topics left for a niche, propose more
# before the pool actually runs dry. At 1 video/day the ~155-topic pool lasts
# ~5 months; at 5/day (multi-run scheduling) that's ~1 month — this is what
# keeps a scaled-up schedule from just running out.
#
# Raised from 30/40: at the OLD threshold, a niche didn't top up until it was
# down to its last 30 topics — random.shuffle() over a shrinking, heavily-used
# pool means the LAST third disproportionately reuses whatever GPT proposed
# early on, since later replenishments only ever add another 40 on top of an
# already-large used-history list feeding the "don't repeat" prompt (a longer
# exclusion list gives GPT less room and it starts circling the same handful
# of well-known events). Topping up earlier (50) and adding more each time
# (60) keeps the ACTIVE unused pool bigger at all times, which is what
# actually fixes "the topics started repeating" — not a bigger pool in total,
# a bigger pool of topics not yet seen.
WIKI_REPLENISH_THRESHOLD = 50
WIKI_REPLENISH_COUNT     = 60


def _unused_wiki_topic_count(niche_name: str, used_ids: set) -> int:
    try:
        topics = json.loads(WIKI_TOPICS_FILE.read_text(encoding="utf-8")).get(niche_name, [])
    except (OSError, json.JSONDecodeError):
        return 0
    n = 0
    for title in topics:
        url_title = title.strip().replace(" ", "_")
        page_url  = f"https://en.wikipedia.org/wiki/{url_title}"
        if "wiki:" + page_url not in used_ids:
            n += 1
    return n


def replenish_wiki_topics(niche_name: str, count: int = WIKI_REPLENISH_COUNT) -> int:
    """GPT proposes `count` new real Wikipedia article titles for this niche,
    each VALIDATED by actually fetching its summary before being trusted
    (GPT invents plausible-sounding titles that don't exist often enough to
    matter) — anything that doesn't resolve to a real, sufficiently long
    article is silently dropped, never appended. Returns how many were added.
    Non-fatal on any failure (missing key, GPT error, all-invalid response):
    the pool just doesn't grow this run, exactly as if this function didn't
    exist — a topic-pool refill failure must never block a video from being
    made from whatever topics remain.
    """
    try:
        keys = _load_keys()
        key  = keys.get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            return 0

        try:
            data = json.loads(WIKI_TOPICS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        existing = list(data.get(niche_name, []))

        from openai import OpenAI
        client = OpenAI(api_key=key)
        existing_sample = ", ".join(existing[-60:])   # keep the prompt bounded
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"List {count} REAL, EXISTING English Wikipedia article "
                    f"titles about the history of money/currency/finance "
                    f"('{niche_name}' niche). Mix of two kinds: (1) specific "
                    f"historical events, currencies, financial crises, "
                    f"institutions, or figures, AND (2) timeless financial or "
                    f"economic CONCEPTS with their own Wikipedia article — "
                    f"compound interest, opportunity cost, moral hazard, "
                    f"Gresham's law, the time value of money, sunk cost, "
                    f"scarcity, diversification, and similar — a concept "
                    f"article gets illustrated with a real historical example "
                    f"when the script is written, so it isn't a repeat every "
                    f"time it's picked. Exact, correctly-spelled article "
                    f"titles only, one per line, no numbering, no commentary. "
                    f"Do NOT repeat any of these already-used topics: {existing_sample}"
                ),
            }],
            temperature=0.8,
            max_tokens=900,
            timeout=60,
        )
        proposed = [l.strip().lstrip("-•").strip() for l in
                   resp.choices[0].message.content.strip().split("\n")]
        proposed = [t for t in proposed if t and t not in existing]
    except Exception as e:
        print(f"[research] topic replenish skipped (non-fatal): {e}")
        return 0

    validated = []
    for title in proposed:
        if len(validated) >= count:
            break
        try:
            url_title = title.replace(" ", "_")
            r = httpx.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}",
                headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT, follow_redirects=True,
            )
            r.raise_for_status()
            extract = _clean_text(r.json().get("extract") or "")
            if len(extract) >= WIKI_MIN_EXTRACT:
                validated.append(title)
        except Exception:
            continue   # GPT invented a title, or it's a disambiguation/stub page

    if not validated:
        return 0
    try:
        data.setdefault(niche_name, [])
        data[niche_name].extend(validated)
        WIKI_TOPICS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[research] topic pool: added {len(validated)} validated "
              f"topic(s) to '{niche_name}' ({len(proposed) - len(validated)} "
              f"proposed titles didn't resolve to real articles)")
    except OSError as e:
        print(f"[research] topic replenish: couldn't write wiki_topics.json: {e}")
        return 0
    return len(validated)


_DISAMBIG_OPENER = re.compile(
    r"\b(?:may|can|could|might)\s+(?:also\s+)?refer\s+to\b"
    r"|\brefers?\s+to\s+the\s+following\b")


def _is_disambiguation(data: dict, extract: str) -> bool:
    """A page that is a LIST of other pages, not an article.

    THE RUN THIS COMES FROM. The supervisor had already rejected one seed and
    the retry landed on:

        [research] using Wikipedia article: "Potosi"
        → Quote: "Potosí or Potosi may refer to the following topics, whose
          names generally origin" — Wikipedia

    A disambiguation page contains no facts, so there is nothing for the story
    architect to find a moment in and nothing for the fact gate to check
    against. The supervisor caught it and the retry logic used it anyway, which
    is how "too generic" made it all the way into a rendered video.

    Two signals, because the REST summary API is not consistent about the
    first: an explicit type, and the sentence every one of these pages opens
    with.

    THE OPENING SENTENCE IS NOT ALWAYS "MAY". A live run asked for Bretton
    Woods and got:

        → Quote: "Bretton Woods can refer to: Bretton Woods, New Hampshire, a
          village in the Unite" — Wikipedia

    which this missed by one word, and the whole run was written on a page
    with no facts in it. Wikipedia's disambiguation pages open with "may",
    "can" and "could", and some say "refer to the following" instead.
    """
    if (data.get("type") or "").lower() == "disambiguation":
        return True
    head = " ".join(extract.split())[:200].lower()
    return bool(_DISAMBIG_OPENER.search(head))


def fetch_wikipedia_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Self-directed, grounded seed source: the machine picks a topic from the
    niche's topic list (config/wiki_topics.json) and pulls the REAL facts from
    Wikipedia's summary API (free, keyless, follows redirects).

    This is the safe version of "the machine invents its own seed": GPT never
    invents the facts — it compresses a real, sourced article extract, exactly
    like a Reddit story. Topics are just titles, so expanding coverage is one
    line of config, not hand-writing researched seed text.
    """
    if used_ids is None:
        used_ids = set()

    if _unused_wiki_topic_count(niche_name, used_ids) < WIKI_REPLENISH_THRESHOLD:
        replenish_wiki_topics(niche_name)   # non-fatal no-op on any failure

    try:
        topics = json.loads(WIKI_TOPICS_FILE.read_text(encoding="utf-8")).get(niche_name, [])
    except (OSError, json.JSONDecodeError):
        topics = []
    if not topics:
        return None

    topics = list(topics)
    random.shuffle(topics)
    attempts = 0
    for title in topics:
        url_title = title.strip().replace(" ", "_")
        page_url  = f"https://en.wikipedia.org/wiki/{url_title}"
        if "wiki:" + page_url in used_ids:
            continue
        if attempts >= WIKI_MAX_ATTEMPTS:
            break
        attempts += 1
        try:
            r = httpx.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}",
                headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT, follow_redirects=True,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[research] Wikipedia fetch warning — {title}: {e}")
            continue

        extract = _clean_text(data.get("extract") or "")
        if len(extract) < WIKI_MIN_EXTRACT:
            continue
        if _is_disambiguation(data, extract):
            print(f"[research] Wikipedia: \"{title}\" is a disambiguation page "
                  f"— skipping (no facts on it to build from)")
            continue
        # Prefer the full article body over the lead-paragraph summary — see
        # WIKI_FULLTEXT_CHARS. Falls back to the summary if the fetch fails.
        body = fetch_wikipedia_fulltext(url_title)
        return {
            "type":    "wikipedia",
            "source":  "Wikipedia",
            "title":   data.get("title") or title,
            "content": body if len(body) > len(extract) else extract,
            "url":     page_url,
        }

    print(f"[research] Wikipedia: no fresh topic found for '{niche_name}' "
          f"({attempts} fetch attempts)")
    return None


def fetch_wikipedia_by_title(query: str) -> dict | None:
    """Grounded seed from a USER-CHOSEN topic instead of the automatic pool —
    still real Wikipedia facts, never invented text, so the fact-gate and
    hook-grounding checks work exactly the same as an auto-picked topic (a
    free-typed topic with no real source would just get invented claims
    rejected downstream, which is why this resolves to a real article
    instead of handing the raw string straight to the script writer).

    Two-step resolution: try the query as an exact title first (fast path
    when it's typed precisely), then fall back to Wikipedia's search API for
    an imprecise/partial query ("bretton woods" -> "Bretton Woods Conference").
    Returns None only if nothing usable is found either way.
    """
    query = (query or "").strip()
    if not query:
        return None

    def _try_title(title: str) -> dict | None:
        url_title = title.strip().replace(" ", "_")
        if not url_title:
            return None
        page_url = f"https://en.wikipedia.org/wiki/{url_title}"
        try:
            r = httpx.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}",
                headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT, follow_redirects=True,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        extract = _clean_text(data.get("extract") or "")
        if len(extract) < WIKI_MIN_EXTRACT:
            return None
        # A disambiguation page is long enough to pass the length check and
        # contains no facts at all, so accepting it here ends the resolution
        # early and the search fallback below — the half that turns "bretton
        # woods" into "Bretton Woods Conference" — never runs. That is exactly
        # how one live run came to be written on a list of village names.
        if _is_disambiguation(data, extract):
            print(f"[research] \"{title}\" is a disambiguation page — "
                  f"searching for the article it points at")
            return None
        body = fetch_wikipedia_fulltext(url_title)
        return {
            "type": "wikipedia", "source": "Wikipedia",
            "title": data.get("title") or title,
            "content": body if len(body) > len(extract) else extract,
            "url": page_url,
        }

    direct = _try_title(query)
    if direct:
        return direct

    try:
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                   "format": "json", "srlimit": 5},
            headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except Exception as e:
        print(f"[research] Wikipedia search failed for '{query}': {e}")
        return None

    for hit in hits:
        result = _try_title(hit.get("title", ""))
        if result:
            return result

    print(f"[research] no Wikipedia article found for topic '{query}'")
    return None


def fetch_trending_wikipedia(niche_name: str, used_ids: set | None = None) -> dict | None:
    """TREND-DRIVEN topic selection: resolve THIS WEEK's rising Google Trends
    queries for the niche into a real Wikipedia article, so the video's SUBJECT
    tracks what people are actually searching — the single biggest reach lever,
    ahead of script polish.

    This is the crucial difference from get_trending_context, which only attaches
    the trending terms as prompt flavour while the topic itself stays random: here
    the trend actually PICKS the topic. Each rising query is resolved through the
    same grounded title→search path as a user-chosen --topic, so the fact-gate and
    hook-grounding checks still apply (a trend term with no real article just gets
    skipped, never handed to the writer as free text). Skips already-used seeds;
    returns None so the caller falls through to the normal source chain whenever
    trends are unavailable (pytrends not installed, rate-limited, or nothing
    resolves to a fresh article)."""
    if used_ids is None:
        used_ids = set()
    for query in _trending_queries(niche_name):
        # THE RISING QUERY IS NOT AUTOMATICALLY THIS CHANNEL'S SUBJECT. Google
        # matched "gold standard" to a whey protein brand and "hyperinflation"
        # to a lung condition, and a live run then resolved
        #     "optimum nutrition gold standard pre-workout"
        # into the Wikipedia article "Sprint (running)" and made it the topic
        # of a monetary-history video. The docstring above promised that "a
        # trend term with no real article just gets skipped" — it guarded
        # against the ABSENCE of an article and never against the wrong one.
        if not _trend_is_usable(query, niche_name):
            print(f"[research] trend '{query}' is not this niche's subject — skipped")
            continue
        seed = fetch_wikipedia_by_title(query)
        if not seed:
            continue
        # And the article has to be on subject too: a term can be about money
        # and still land on an article that is not.
        if not _is_on_subject(f"{seed.get('title', '')} {seed.get('content', '')}",
                              niche_name):
            print(f"[research] trend '{query}' resolved to an off-topic "
                  f"article ({seed.get('title', '?')}) — skipped")
            continue
        if _seed_id(seed) in used_ids:
            continue
        seed["trend_query"] = query   # provenance: which rising search picked this
        return seed
    return None


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
    topic_re = SE_TOPIC_FILTER_RE.get(niche_name)

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
            # General-interest feeds (Smithsonian/JSTOR/Aeon) cover every topic —
            # require genuinely on-topic content for niches that need it, same as
            # the StackExchange filter (e.g. money_history: not just any history).
            if topic_re and not (topic_re.search(title) or topic_re.search(desc)):
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


# ── Historic newspapers (Library of Congress, "Chronicling America") ─────────
#
# WHY THIS SOURCE EXISTS. Reddit has needed OAuth for months and blocks five of
# six seed sources on every run; Stack Exchange answers maybe half the time; and
# what is left is Wikipedia, which hands back an encyclopedia OVERVIEW. An
# overview is the exact shape the pipeline cannot use — "Sterling was the
# fourth-most-traded currency in 2022" is a fact about a category, and a
# category has no date, no street and nobody standing in it. That is what
# produced "The secret? Its historical resilience and trust."
#
# A newspaper page is the opposite shape and it is not a matter of luck: every
# result carries a printing date, a city, a paper, and a reporter writing about
# something that happened THAT WEEK. THE SCENE asks for "a date or year, a
# place, and ONE named person doing ONE specific thing" — a front page from
# Philadelphia on February 21, 1893 supplies three of those before the model
# reads a word.
#
# NO KEY, NO OAUTH, NO ACCOUNT. It is a US federal public-domain archive:
# ~20 million pages, 1756-1963, open JSON. That is the whole reason it is
# worth adding rather than a sixth thing to authenticate.
#
# THE HONEST COST: this is OCR of microfilm of 19th-century newsprint, so some
# pages come back as soup — column bleed, broken hyphenation, a masthead
# dissolved into punctuation. _ocr_is_legible is the filter, and it is
# deliberately strict: a rejected page costs one HTTP call, an accepted bad one
# costs a whole video. Expect a fair number of skips in the log; that is the
# filter working, not the source failing.
NEWS_TOPIC_QUERIES = {
    "money_history": (
        "run on the bank", "bank failure", "gold reserve", "silver dollar",
        "counterfeit notes", "the mint", "panic in wall street",
        "wages cut", "price of bread", "railroad receivers", "bankrupt",
        "specie payment", "greenbacks", "treasury notes",
    ),
}

# The window this channel is about, and the window where OCR is good enough to
# be worth reading. Overridable per run.
NEWS_YEAR_MIN = 1850
NEWS_YEAR_MAX = 1922
NEWS_TIMEOUT = 25.0
NEWS_MIN_WORDS = 140          # below this there is no story on the page
NEWS_WINDOW_CHARS = 1800      # what we hand the writer, centred on the hit


def _ocr_is_legible(text: str) -> bool:
    """Whether an OCR page is clean enough to build a script on.

    Microfilm OCR fails in a recognisable way: it produces many short
    non-words ("tlie", "ii", "0f", "rn") rather than a few long ones. Counting
    the share of tokens that look like ordinary words separates a readable
    column from a scanned advertisement page far more reliably than length
    does, and costs nothing.
    """
    words = (text or "").split()
    if len(words) < NEWS_MIN_WORDS:
        return False
    real = sum(1 for w in words
               if w.isalpha() and 3 <= len(w) <= 14)
    return real / len(words) >= 0.62


def _ocr_window(text: str, query: str) -> str:
    """The passage AROUND the search hit, not the top of the page.

    A newspaper page is six unrelated columns. The top of the page is a
    masthead and a shipping list; the story we searched for is somewhere in the
    middle, and handing the writer the whole page means handing it five stories
    it did not ask for.
    """
    flat = " ".join((text or "").split())
    # The LONGEST word in the query, not the first. "run on the bank" starting
    # from "run" lands on "running", "drunk" or "runaway" long before it finds
    # the story; "bank" is the token that actually marks it.
    key = max((query or "").split(), key=len, default="").lower()
    at = flat.lower().find(key) if key else -1
    if at < 0:
        at = 0
    start = max(0, at - NEWS_WINDOW_CHARS // 3)
    return flat[start:start + NEWS_WINDOW_CHARS].strip()


def _news_date_place(item: dict) -> tuple[str, str]:
    """A readable date and place from one Chronicling America item."""
    raw = str(item.get("date") or "")
    date = raw
    if len(raw) == 8 and raw.isdigit():          # 18930220
        date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    city = (item.get("city") or [""])[0] if isinstance(item.get("city"), list) \
        else (item.get("city") or "")
    state = (item.get("state") or [""])[0] if isinstance(item.get("state"), list) \
        else (item.get("state") or "")
    place = ", ".join(p for p in (city, state) if p)
    return date, place


def fetch_newspaper_story(niche_name: str, used_ids: set | None = None) -> dict | None:
    """A dated page from a named American town, via the Library of Congress.

    Keyless and unauthenticated by design — see the block comment above.
    Fail-open like every other source here: any failure returns None and the
    chain continues to Wikipedia exactly as it does today.
    """
    if used_ids is None:
        used_ids = set()
    queries = NEWS_TOPIC_QUERIES.get(niche_name)
    if not queries:
        return None                       # niche has no newspaper vocabulary
    if os.environ.get("RUFUS_NEWSPAPERS", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None

    q = random.choice(list(queries))
    y1 = int(os.environ.get("RUFUS_NEWS_YEAR_MIN", str(NEWS_YEAR_MIN)))
    y2 = int(os.environ.get("RUFUS_NEWS_YEAR_MAX", str(NEWS_YEAR_MAX)))
    page = random.randint(1, 5)           # rotate so a daily channel keeps moving
    # THE LEGACY HOST IS GONE. chroniclingamerica.loc.gov now 302s to
    # www.loc.gov/chroniclingamerica/… and that path 404s, so the old call
    # failed on every run with a 404 that looked like a bad query rather than a
    # retired API. The modern collection search is tried first; the legacy URL
    # is kept behind it because mirrors and older deployments still answer it.
    modern = ("https://www.loc.gov/collections/chronicling-america/"
              f"?q={quote(q)}&fo=json&c=12&sp={page}"
              f"&start_date={y1}-01-01&end_date={y2}-12-31")
    legacy = ("https://chroniclingamerica.loc.gov/search/pages/results/"
              f"?andtext={quote(q)}&format=json&rows=12&page={page}"
              f"&date1={y1}&date2={y2}&dateFilterType=yearRange")

    items, failures = [], []
    for url in (modern, legacy):
        try:
            r = httpx.get(url, headers=WIKI_HEADERS, timeout=NEWS_TIMEOUT,
                          follow_redirects=True)
            r.raise_for_status()
            payload = r.json()
            # The two APIs disagree on the envelope: the collection search
            # returns "results", the page search returns "items".
            items = payload.get("results") or payload.get("items") or []
            if items:
                break
        except Exception as e:
            failures.append(f"{url.split('/')[2]}: {type(e).__name__}")

    if not items:
        # Loud and specific. "newspapers unavailable" alone would read like a
        # network blip for months, which is how the retired host went unnoticed.
        print(f"[research] newspapers unavailable for \"{q}\" "
              f"({'; '.join(failures) or 'no results'}) — both the modern and "
              f"legacy LoC endpoints came back empty. RUFUS_NEWSPAPERS=0 "
              f"silences this source if it stays dead.")
        return None

    skipped = 0
    for item in items:
        page_url = "https://chroniclingamerica.loc.gov" + str(item.get("id") or "")
        if "news:" + page_url in used_ids:
            continue
        # "ocr_eng" is the legacy page search; the collection search puts its
        # text in "description" (a list) or "item.
        raw_text = item.get("ocr_eng")
        if not raw_text:
            desc = item.get("description")
            raw_text = " ".join(desc) if isinstance(desc, list) else (desc or "")
        text = _clean_text(raw_text or "")
        # Window FIRST, then judge. A newspaper page is six columns, and most
        # of them are classified ads and shipping tables that read as soup to
        # the legibility test. Judging the whole page throws away good stories
        # for the company they keep; judging the passage we are actually going
        # to send is both stricter about what matters and fairer to the page.
        body = _ocr_window(text, q)
        if not _ocr_is_legible(body):
            skipped += 1
            continue
        date, place = _news_date_place(item)
        if not date:
            skipped += 1
            continue
        paper = str(item.get("title") or "an American newspaper")
        print(f"[research] newspapers [\"{q}\", page {page}]: "
              f"{len(items)} page(s), {skipped} unreadable")
        return {
            "type":    "newspaper",
            "source":  f"{paper}, {place} ({date})" if place else f"{paper} ({date})",
            "title":   f"{paper}, {date}",
            # The date and place lead the content on purpose. They are the two
            # things the story architect needs and the two things an OCR column
            # is least likely to state in a sentence.
            "content": (f"Printed {date} in {place or 'the United States'}, in "
                        f"{paper}. Searched for \"{q}\".\n\n{body}"),
            "url":     page_url,
        }

    print(f"[research] newspapers [\"{q}\", page {page}]: "
          f"{len(items)} page(s), none usable ({skipped} unreadable)")
    return None


def pick_wisdom_quote(niche_name: str, used_ids: set | None = None) -> dict | None:
    """Pick one random quote from the niche's curated pool, skipping seen ones."""
    if used_ids is None:
        used_ids = set()
    f = WISDOM_DIR / f"{niche_name}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
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


def _reddit_skip_reason() -> str | None:
    """Why Reddit will not be tried this run, or None if it will be.

    ASK BEFORE KNOCKING FIVE TIMES. Reddit stopped serving its .json endpoints
    to unauthenticated clients, so without credentials every run walked all
    five subreddits, failed identically on each, and printed:

        [research] reddit blocked r/EconomicHistory — no OAuth credentials
        [research] reddit blocked r/badeconomics — no OAuth credentials
        [research] reddit blocked r/history — no OAuth credentials
        [research] reddit blocked r/AskHistorians — no OAuth credentials
        [research] reddit blocked r/economics — no OAuth credentials

    Five HTTP round trips that cannot succeed, and five lines at the top of
    every log that train the reader to skim the research section — which is
    where the lines that DO matter live. The credentials are readable before
    the first request, so the answer is knowable before the first request.

    Returns a reason rather than a bool so the caller can print WHICH of the
    three causes it is. Absent credentials is not the same as being switched
    off, and neither is a missing library.
    """
    if os.environ.get("RUFUS_SKIP_REDDIT", "0").strip().lower() in (
            "1", "true", "yes", "on"):
        return "RUFUS_SKIP_REDDIT=1"
    keys = _load_keys()
    has_creds = bool(str(keys.get("reddit_client_id", "")).strip()
                     and str(keys.get("reddit_client_secret", "")).strip())
    if not has_creds:
        return ("no reddit_client_id/reddit_client_secret in config/keys.json. "
                "Reddit refuses unauthenticated JSON now, so all five "
                "subreddits would fail identically. Create a free 'script' app "
                "at https://www.reddit.com/prefs/apps to restore them")
    if not _PRAW_AVAILABLE:
        import paths
        return (f"praw is not installed (run `{paths.pip_hint('praw')}`) — "
                f"credentials are set but unusable")
    return None


def _skip_reddit() -> bool:
    """Whether Reddit is bypassed this run. See _reddit_skip_reason."""
    return _reddit_skip_reason() is not None


def get_seed(niche_name: str | None = None, topic: str | None = None) -> dict:
    """Get a seed for the script writer. Tracks history so seeds never repeat.

    Order: Reddit → StackExchange → Wikipedia → RSS → Hacker News → wisdom fallback.
    Set RUFUS_SKIP_REDDIT=1 to skip straight past Reddit (e.g. no OAuth app set up).
    All sources skip anything already in used_seeds.json (last MAX_USED_HISTORY items).

    `topic`: bypass the automatic source chain entirely and build the seed
    from a topic YOU chose (main.py --topic "..."), resolved to a real
    Wikipedia article so it's still grounded in real facts, not free text —
    a raw user string handed straight to the script writer would just get
    its invented claims rejected by the fact-gate downstream anyway. Raises
    if nothing matches, rather than silently falling through to a random
    topic — a failed manual request should be obvious, not swapped out.
    """
    if topic:
        seed = fetch_wikipedia_by_title(topic)
        if not seed:
            raise RuntimeError(
                f"No Wikipedia article found for topic '{topic}' — try a more "
                f"specific or differently-worded topic.")
        print(f"[research] using YOUR topic → Wikipedia: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        trending_context = get_trending_context(niche_name or _load_niche()[1])
        if trending_context:
            seed["trending_context"] = trending_context
        return seed

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

    # TREND-FIRST topic selection: before the standard source chain, try to build
    # the topic FROM this week's rising searches (fetch_trending_wikipedia). This
    # makes the video's subject track demand, not just its hook. Fully graceful —
    # returns None (falls straight through to the chain below) whenever pytrends
    # is absent/rate-limited or nothing resolves. Opt out with RUFUS_TREND_TOPICS=0.
    if os.environ.get("RUFUS_TREND_TOPICS", "1").strip().lower() not in ("0", "false", "no", "off"):
        seed = fetch_trending_wikipedia(name, used_ids=used_set)
        if seed:
            print(f"[research] TREND-DRIVEN topic → Wikipedia: "
                  f"\"{seed['title'][:60]}\" (rising: {seed.get('trend_query', '?')})")
            _mark_seed_used(seed)
            return _with_trending(seed)

    _reddit_off = _reddit_skip_reason()
    if _reddit_off:
        print(f"[research] Reddit off — {_reddit_off}")
    else:
        for sub in subreddits:
            seed = fetch_reddit_story(sub, used_ids=used_set)
            if seed:
                print(f"[research] using Reddit story from {seed['source']}: \"{seed['title'][:60]}\"")
                _mark_seed_used(seed)
                return _with_trending(seed)

    # OpenAlex FIRST among the keyless sources, and deliberately ahead of
    # StackExchange. An abstract states a year, names its authors and carries
    # the study's own figures; a discussion thread contains an argument. The
    # fact gate has been rejecting scripts for missing exactly the first three
    # things, so this is that failure answered at the source rather than at
    # the gate. RUFUS_OPENALEX=0 restores the old order.
    seed = fetch_openalex_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using paper: \"{seed['title'][:60]}\" ({seed['source']})")
        _mark_seed_used(seed)
        return _with_trending(seed)

    # StackExchange: keyless API, never IP-blocked like Reddit's public JSON
    seed = fetch_stackexchange_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using StackExchange story from {seed['source']}: \"{seed['title'][:60]}\"")
        _mark_seed_used(seed)
        return _with_trending(seed)

    # Historic newspapers: keyless, and the only source here that hands back a
    # date and a town with every result. Ahead of Wikipedia deliberately —
    # Wikipedia returns an overview of a subject, this returns a day.
    seed = fetch_newspaper_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using newspaper page: {seed['source']}")
        _mark_seed_used(seed)
        return _with_trending(seed)

    # Wikipedia: self-directed topic pick, real sourced facts, keyless, infinite
    seed = fetch_wikipedia_story(name, used_ids=used_set)
    if seed:
        print(f"[research] using Wikipedia article: \"{seed['title'][:60]}\"")
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
    # `--newspapers [niche]` exercises the Library of Congress source alone.
    # It exists because that endpoint cannot be reached from every network, so
    # "does this work here?" has to be answerable in one command instead of by
    # reading a full run's log.
    if "--newspapers" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--newspapers"]
        got = fetch_newspaper_story(rest[0] if rest else "money_history", set())
        if not got:
            print("no usable page this time — re-run (the query rotates), and "
                  "check the line above for the reason")
            sys.exit(1)
        print(json.dumps(got, indent=2)[:2000])
        sys.exit(0)
    niche = sys.argv[1] if len(sys.argv) > 1 else None
    seed  = get_seed(niche)
    print(json.dumps(seed, indent=2))
