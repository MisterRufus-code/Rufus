#!/usr/bin/env python3
"""
competitors.py — what the neighbouring channels did, and which of it worked.

WHY RAW VIEWS ARE NOT EVIDENCE. Fifty thousand views on a channel that averages
two hundred thousand is a flop. Twenty thousand on a channel that averages three
thousand is the most interesting thing that happened this week. Sorting a mixed
list of other people's videos by view count therefore tells you which channels
are big, which you already knew, and nothing about which IDEAS worked.

So every video here is scored against ITS OWN channel:

    outperformance = this video's views / the median views of that channel's
                     recent uploads

A ratio near 1.0 is a normal day for them. Above about 2.0 is a video that did
something their own audience does not usually reward them for — and that is a
thing worth reading the title of.

WHY NAMED CHANNELS AND NOT SEARCH. An open keyword search finds whatever
YouTube feels like surfacing, burns quota fast, and cannot tell a genuine trend
from one viral fluke. Fifteen channels the owner chose, watched over weeks, is
a smaller and far more honest signal: the same faces, the same format, and a
baseline for each one.

    config/competitors.json     {"channels": ["UC...", "UC..."]}

QUOTA. playlistItems.list and videos.list are 1 unit per call, and this makes
two calls per channel per pass. Fifteen channels four times a day is 120 units
against a 10,000/day default — the daily video upload costs 1,600 on its own.
This is not the expensive part of anything.

CONTRACT: fail-open and never raises into a scheduled task. No key, no quota,
a dead channel id, a channel with two uploads — all return an empty list and a
printed reason.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
COMPETITORS_FILE = CONFIG_DIR / "competitors.json"

# How many recent uploads to pull per channel. Enough for a stable median and
# still one page of the API.
RECENT_PER_CHANNEL = 25

# Below this many uploads there is no median worth computing — one video is
# trivially its own median and would score 1.0 forever.
MIN_FOR_BASELINE = 5

# What counts as "this did better than they usually do". Deliberately high:
# a threshold that flags a third of every channel's uploads is describing
# normal variance, and this repo has walked back two checks that fired on
# most of what they looked at.
OUTPERFORMANCE = 2.0


def _is_placeholder(channel_id: str) -> bool:
    """An id straight out of competitors.json.example.

    Copying the example and not filling it in is the likeliest first-run state
    there is, and without this it costs an API call each to discover and comes
    back as "no such channel" — which reads like the channels were deleted
    rather than never entered.

    DELIBERATELY NARROW: a real id's full 24-character length AND every
    character after the UC being the same one. The first draft checked only
    the repeated character and started rejecting short ids in test fixtures —
    which are not real ids either, but they are not the example's placeholder,
    and a check that quietly drops a channel the owner chose would be a worse
    bug than the one it fixes. "This cannot be a channel id" is a different
    question and not one this needs to answer.
    """
    if len(channel_id) != 24 or channel_id[:2].upper() != "UC":
        return False
    return len(set(channel_id[2:].lower())) == 1


def channels() -> list[str]:
    """The channel ids to watch. [] with a reason when there are none."""
    if not COMPETITORS_FILE.exists():
        print(f"[competitors] no {COMPETITORS_FILE.name} — copy "
              f"competitors.json.example and put 5-15 channel ids in it")
        return []
    try:
        raw = json.loads(COMPETITORS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[competitors] {COMPETITORS_FILE.name} is unreadable ({e})")
        return []
    listed = [str(c).strip() for c in (raw.get("channels") or []) if str(c).strip()]
    out = [c for c in listed if not _is_placeholder(c)]
    if len(out) < len(listed):
        print(f"[competitors] {COMPETITORS_FILE.name} still has "
              f"{len(listed) - len(out)} placeholder id(s) from the example in "
              f"it — replace them with real channel ids (they start with UC "
              f"and are 24 characters)")
    if not out:
        print(f"[competitors] {COMPETITORS_FILE.name} has no real channels in "
              f"it, so there is nothing to watch")
    return out


def _service():
    """An authenticated YouTube client, or None with a reason.

    Reuses analytics_fetcher's auth rather than growing a second token flow:
    the scopes, the token file and the "no browser here" behaviour are already
    solved there, and two auth paths against one token file is how a scheduled
    task ends up sitting on an interactive OAuth prompt at 2am.
    """
    try:
        from googleapiclient.discovery import build
        from analytics_fetcher import _auth
        from channel_config import load_channel
        return build("youtube", "v3", credentials=_auth(load_channel()))
    except Exception as e:
        print(f"[competitors] no YouTube access ({e}) — skipping this pass")
        return None


def _uploads_playlist(yt, channel_id: str) -> str:
    """The channel's uploads playlist id, or ""."""
    try:
        resp = yt.channels().list(part="contentDetails",
                                  id=channel_id).execute()
        items = resp.get("items") or []
        if not items:
            print(f"[competitors] {channel_id}: no such channel")
            return ""
        return (items[0]["contentDetails"]["relatedPlaylists"]
                .get("uploads", ""))
    except Exception as e:
        print(f"[competitors] {channel_id}: channel lookup failed ({e})")
        return ""


def _recent_video_ids(yt, playlist_id: str, limit: int) -> list[str]:
    try:
        resp = yt.playlistItems().list(part="contentDetails",
                                       playlistId=playlist_id,
                                       maxResults=min(50, limit)).execute()
        return [i["contentDetails"]["videoId"]
                for i in (resp.get("items") or [])
                if i.get("contentDetails", {}).get("videoId")][:limit]
    except Exception as e:
        print(f"[competitors] playlist {playlist_id} failed ({e})")
        return []


def _stats(yt, video_ids: list[str]) -> list[dict]:
    """title/published/views/duration per video. One call for up to 50."""
    if not video_ids:
        return []
    try:
        resp = yt.videos().list(part="snippet,statistics,contentDetails",
                                id=",".join(video_ids[:50])).execute()
    except Exception as e:
        print(f"[competitors] stats failed ({e})")
        return []
    out = []
    for item in resp.get("items") or []:
        sn = item.get("snippet") or {}
        st = item.get("statistics") or {}
        out.append({
            "video_id": item.get("id", ""),
            "channel_id": sn.get("channelId", ""),
            "channel_title": sn.get("channelTitle", ""),
            "title": sn.get("title", ""),
            "published_at": sn.get("publishedAt", ""),
            "views": int(st.get("viewCount", 0) or 0),
            "duration": (item.get("contentDetails") or {}).get("duration", ""),
        })
    return out


def score(videos: list[dict]) -> list[dict]:
    """Add `outperformance` to each video, against its own channel's median.

    Pure, so the arithmetic that decides what the scout reads can be checked
    without a network. A channel with fewer than MIN_FOR_BASELINE uploads gets
    no ratio at all rather than a made-up one — with two videos, each is either
    the median or twice it, and both numbers are noise.
    """
    by_channel: dict[str, list[dict]] = {}
    for v in videos:
        by_channel.setdefault(v.get("channel_id", ""), []).append(v)

    out: list[dict] = []
    for _cid, group in by_channel.items():
        counts = [v["views"] for v in group if v.get("views")]
        median = statistics.median(counts) if len(counts) >= MIN_FOR_BASELINE else 0
        for v in group:
            v = dict(v)
            v["channel_median"] = int(median)
            v["outperformance"] = (round(v["views"] / median, 2)
                                   if median else 0.0)
            out.append(v)
    return out


def outperformers(videos: list[dict],
                  threshold: float = OUTPERFORMANCE) -> list[dict]:
    """The ones that beat their own channel, strongest first."""
    hits = [v for v in videos if v.get("outperformance", 0) >= threshold]
    return sorted(hits, key=lambda v: v["outperformance"], reverse=True)


def observe(limit_per_channel: int = RECENT_PER_CHANNEL) -> list[dict]:
    """One pass over every watched channel. [] on any failure, with a reason."""
    ids = channels()
    if not ids:
        return []
    yt = _service()
    if yt is None:
        return []

    collected: list[dict] = []
    for cid in ids:
        playlist = _uploads_playlist(yt, cid)
        if not playlist:
            continue
        vids = _recent_video_ids(yt, playlist, limit_per_channel)
        got = _stats(yt, vids)
        if len(got) < MIN_FOR_BASELINE:
            print(f"[competitors] {cid}: only {len(got)} upload(s) — not "
                  f"enough for a baseline, skipping")
            continue
        collected += got
    return score(collected)


def describe(videos: list[dict], top: int = 10) -> str:
    """One readable block for a log or a dashboard card."""
    hits = outperformers(videos)
    if not videos:
        return "nothing observed"
    if not hits:
        return (f"{len(videos)} video(s) observed, none above "
                f"{OUTPERFORMANCE:g}x its own channel's median — a quiet week "
                f"is a real answer, not a failure")
    lines = [f"{len(hits)} of {len(videos)} beat their own channel:"]
    for v in hits[:top]:
        lines.append(f"  {v['outperformance']:.1f}x  {v['views']:,} views  "
                     f"{v['channel_title'][:22]:<22}  {v['title'][:60]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(observe()))
