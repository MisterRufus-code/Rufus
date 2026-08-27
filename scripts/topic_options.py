#!/usr/bin/env python3
"""
topic_options.py — three topics to choose between, or the one you already want.

WHY THIS EXISTS BESIDE THE SCOUT. scout.py answers "what did the neighbours do
that worked" and needs config/competitors.json to answer anything at all. This
answers a different question — "give me three I could make today" — and must
work on a fresh install with nothing configured, because it is the first thing
a person touches.

WHERE THE THREE COME FROM, in order of how much they are worth:

    already rising   a competitor video that beat its own channel's median,
                     if the scout has ever observed one. Evidence, not a guess.
    trending now     research.trending_queries_with_reason, when it answers.
    the standing list config/wiki_topics.json — 155 money-history subjects the
                     owner curated, minus everything already made.

Never fewer than three while the list has three left, because a stage that
sometimes offers one option is not a choice. Each carries WHY it is there, for
the same reason a scout proposal does: "make a video about the Panic of 1893"
is an instruction, and an instruction from an agent is the thing a person
cannot audit.

A TYPED TOPIC SKIPS ALL OF IT. If you already know what you want, none of the
above is a question worth asking — take_topic() grounds it against Wikipedia
and nothing else runs.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONFIG_DIR = Path(__file__).parent.parent / "config"
WIKI_TOPICS = CONFIG_DIR / "wiki_topics.json"

DEFAULT_N = 3


def _standing_list(niche: str) -> list[str]:
    try:
        data = json.loads(WIKI_TOPICS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    pool = data.get(niche) if isinstance(data, dict) else None
    return [str(t).strip() for t in (pool or []) if str(t).strip()]


def _already_made(limit: int = 400) -> list[str]:
    """Titles this channel has made or already proposed. Loud on failure.

    Failing open here means "nothing has been made", which is not a degraded
    answer — it is the wrong one, and it would offer the same three topics
    every time while looking perfectly reasonable.
    """
    out: list[str] = []
    try:
        import db_manager
        out += db_manager.recent_titles(limit=limit)
        out += [p.get("topic") or ""
                for p in db_manager.proposals(status=None, limit=limit)]
        out += [p.get("title") or ""
                for p in db_manager.projects(status=None, limit=limit)]
    except Exception as e:
        print(f"[topics] ⚠ could not read what has already been made ({e}) — "
              f"the suggestions below may repeat something recent")
    return [t for t in out if t]


def _fresh(candidates: list[str], made: list[str]) -> list[str]:
    """Drop anything this channel has covered, by words rather than by string.

    Two videos about the same event never share a title, so exact matching
    would catch nothing. scout.already_covered owns that comparison; this
    borrows it rather than inventing a second one that disagrees.
    """
    try:
        from scout import already_covered
    except Exception:
        seen = {t.lower() for t in made}
        return [c for c in candidates if c.lower() not in seen]
    return [c for c in candidates if not already_covered(c, made)]


def _rising(niche: str, made: list[str]) -> list[dict]:
    try:
        import db_manager
        import scout
        rows = db_manager.rising(min_outperformance=scout.OUTPERFORMANCE)
    except Exception:
        return []
    out = []
    for r in rows[:6]:
        title = (r.get("title") or "").strip()
        if not title or already_made(title, made):
            continue
        subject = title
        try:
            subject = scout.subject_of(title, niche)
        except Exception:
            pass
        out.append({"title": subject,
                    "why": (f"{r.get('channel_title', 'a watched channel')} did "
                            f"{r.get('outperformance', 0):.1f}x their own median "
                            f"with this")})
    return out


def already_made(title: str, made: list[str]) -> bool:
    try:
        from scout import already_covered
        return already_covered(title, made)
    except Exception:
        return title.lower() in {t.lower() for t in made}


def _trending(niche: str, made: list[str]) -> list[dict]:
    try:
        import research
        queries, reason = research.trending_queries_with_reason(niche)
    except Exception:
        return []
    out = []
    for q in (queries or [])[:4]:
        q = str(q).strip()
        if q and not already_made(q, made):
            out.append({"title": q, "why": f"trending now — {reason}"})
    return out


def suggest(niche: str = "money_history", n: int = DEFAULT_N) -> list[dict]:
    """`n` topics worth making, best evidence first."""
    made = _already_made()
    options: list[dict] = []
    seen: set[str] = set()

    def _add(items):
        for it in items:
            key = it["title"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                options.append(it)

    _add(_rising(niche, made))
    if len(options) < n:
        _add(_trending(niche, made))
    if len(options) < n:
        standing = _fresh(_standing_list(niche), made)
        random.shuffle(standing)
        _add({"title": t,
              "why": "from your standing money-history list, not covered yet"}
             for t in standing[:n * 2])

    if not options:
        print("[topics] nothing left to suggest — config/wiki_topics.json is "
              "empty or everything in it has been covered")
    return options[:n]


def take_topic(topic: str, niche: str = "money_history") -> dict:
    """The topic you already wanted, grounded and nothing else asked.

    No trending check, no standing list, no de-duplication against past videos:
    you said what you want, and second-guessing that is the pipeline arguing
    with the person operating it. It is still resolved to a real article,
    because the fact gate needs a source to check against.
    """
    out = {"title": topic.strip(), "why": "you asked for this one",
           "source": ""}
    try:
        import research
        seed = research.get_seed(niche, topic=topic)
        title = (seed.get("title") or "").strip()
        if title:
            out["title"] = title
        out["source"] = seed.get("source", "")
    except Exception as e:
        print(f"[topics] could not ground {topic!r} ({e}) — using it as typed")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--niche", default="money_history")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    a = ap.parse_args()
    for o in suggest(a.niche, a.n):
        print(f"  {o['title']}\n    {o['why']}")
