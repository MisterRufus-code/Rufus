#!/usr/bin/env python3
"""
feedback_analyzer.py
Analyzes video performance per channel and writes that channel's learnings.json
so script_writer.py can inject winning patterns into future GPT prompts.

Run after analytics_fetcher.py:
    python scripts/feedback_analyzer.py             # all channels
    python scripts/feedback_analyzer.py --channel main_en
"""

import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "rufus.db"


def _conn():
    return sqlite3.connect(str(DB_FILE))


def analyze(channel_id: str = None):
    from channel_config import load_channel, list_channels

    channels = [channel_id] if channel_id else list_channels()
    for cid in channels:
        channel = load_channel(cid)
        _analyze_channel(channel)


def _analyze_channel(channel):
    with _conn() as c:
        rows = c.execute("""
            SELECT v.id, v.niche, v.script_hook, v.title, v.scene_desc,
                   m.views, m.watch_pct, m.ctr, m.likes
            FROM videos v
            JOIN (
                SELECT video_id, views, watch_pct, ctr, likes
                FROM metrics
                WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY video_id)
            ) m ON m.video_id = v.id
            WHERE (v.channel = ? OR v.channel IS NULL)
              -- A YOUTUBE ID SHARED BY TWO ROWS CANNOT BE ATTRIBUTED, and a
              -- row that cannot be attributed must not vote. Six videos in
              -- this channel's database carried the id kGVAHaObJ38 — one link
              -- recorded against six different scripts — so all six joined to
              -- a SEVENTH video's metrics: the same view count, the same
              -- likes, and a watch percentage of zero.
              --
              -- Zero watch_pct is fatal here specifically, because engagement
              -- is watch_pct × log(likes) × log(views). Zero times anything is
              -- zero, so every one of them sorted to the bottom and every one
              -- was written into losing_hooks. All four entries in that list
              -- were these rows. The flywheel was not learning that those
              -- hooks failed; it was learning that a data-entry mistake looks
              -- like failure, and then feeding that back into the hook prompt.
              AND (v.youtube_id IS NULL OR v.youtube_id = '' OR v.youtube_id IN (
                    SELECT youtube_id FROM videos
                    WHERE youtube_id IS NOT NULL AND youtube_id <> ''
                    GROUP BY youtube_id HAVING COUNT(*) = 1))
            ORDER BY v.id DESC
        """, (channel.id,)).fetchall()

    # SAID OUT LOUD, because a silently smaller sample is how this went
    # unnoticed for a week. The rows are still in the database and still fixable
    # — the owner clears the wrong link and they come back into the learning.
    try:
        import db_manager
        dupes = db_manager.duplicate_youtube_ids()
    except Exception:
        dupes = []
    if dupes:
        n = sum(d["count"] for d in dupes)
        print(f"[feedback] ⚠ {n} video(s) excluded: {len(dupes)} YouTube id(s) "
              f"are claimed by more than one row, so their views belong to "
              f"whichever video really owns the link.")
        for d in dupes:
            print(f"[feedback]   {d['youtube_id']} → videos "
                  f"{', '.join(str(i) for i in d['video_ids'])}")
        print(f"[feedback]   Clear the wrong ones and they rejoin the learning.")

    if len(rows) < 3:
        print(f"[feedback] {channel.id}: not enough data ({len(rows)} videos with "
              f"metrics, need ≥3)")
        return
    if len(rows) < 5:
        print(f"[feedback] {channel.id}: ⚠ small sample ({len(rows)}) – patterns may be noisy")

    # Engagement score = watch_pct × log(likes+2), views-weighted (log-damped
    # so one viral outlier doesn't drown everything).
    #
    # CTR was REMOVED from the formula: the fetcher's CTR source
    # (annotationClickThroughRate) was retired by YouTube in 2019 and returned
    # 0 for every video — multiplying by it zeroed EVERY score, the sort
    # became a stable no-op, and "winning_hooks" injected into the hook
    # prompt were just the newest 20% of videos. The flywheel was silently
    # learning noise.
    scored = []
    for row in rows:
        vid_id, niche, hook, title, scene, views, watch_pct, ctr, likes = row
        engagement = watch_pct * math.log(likes + 2) * math.log((views or 0) + 2)
        scored.append({
            "id":         vid_id,
            "niche":      niche,
            "hook":       hook or "",
            "title":      title or "",
            "scene":      scene or "",
            "engagement": engagement,
        })

    scored.sort(key=lambda x: x["engagement"], reverse=True)
    n      = len(scored)
    top_n  = max(1, n // 5)
    top_20 = scored[:top_n]
    bot_20 = scored[-top_n:]

    winning_hooks  = [v["hook"] for v in top_20 if v["hook"]][:5]
    losing_hooks   = [v["hook"] for v in bot_20 if v["hook"]][:5]
    winning_titles = [v["title"] for v in top_20 if v["title"]][:5]
    losing_titles  = [v["title"] for v in bot_20 if v["title"]][:5]

    # Average engagement per niche
    niche_scores: dict[str, list] = {}
    for v in scored:
        niche_scores.setdefault(v["niche"], []).append(v["engagement"])
    avg_by_niche = {k: round(sum(vs) / len(vs), 4) for k, vs in niche_scores.items()}
    best_niches  = sorted(avg_by_niche, key=avg_by_niche.__getitem__, reverse=True)

    learnings = {
        "winning_hooks":         winning_hooks,
        "losing_hooks":          losing_hooks,
        "winning_titles":        winning_titles,
        "losing_titles":         losing_titles,
        "best_niches":           best_niches,
        "avg_score_by_niche":    avg_by_niche,
        "total_videos_analyzed": n,
    }

    path = channel.learnings_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(learnings, indent=2), encoding="utf-8")

    print(f"[feedback] {channel.id}: learnings updated → {path} ({n} videos)")
    print(f"[feedback] best niches:  {best_niches}")
    print(f"[feedback] top hooks:    {winning_hooks[:3]}")
    print(f"[feedback] avoid hooks:  {losing_hooks[:3]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="Analyze one channel (default: all)")
    args = ap.parse_args()
    analyze(args.channel)
