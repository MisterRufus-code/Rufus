#!/usr/bin/env python3
"""
feedback_analyzer.py
Analyzes video performance and writes config/learnings.json so script_writer.py
can inject winning patterns into future GPT prompts.

Run after analytics_fetcher.py:
    python scripts/feedback_analyzer.py
"""

import json
import math
import sqlite3
from pathlib import Path

ROOT           = Path(__file__).parent.parent
DB_FILE        = ROOT / "rufus.db"
LEARNINGS_FILE = ROOT / "config" / "learnings.json"


def _conn():
    return sqlite3.connect(str(DB_FILE))


def analyze():
    with _conn() as c:
        rows = c.execute("""
            SELECT v.id, v.niche, v.script_hook, v.scene_desc,
                   m.views, m.watch_pct, m.ctr, m.likes
            FROM videos v
            JOIN (
                SELECT video_id, views, watch_pct, ctr, likes
                FROM metrics
                WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY video_id)
            ) m ON m.video_id = v.id
            ORDER BY v.id DESC
        """).fetchall()

    if len(rows) < 3:
        print(f"Not enough data ({len(rows)} videos with metrics). Need at least 3.")
        return
    if len(rows) < 5:
        print(f"[feedback] ⚠ small sample ({len(rows)} videos) – patterns may be noisy")

    # Engagement score = CTR × watch_pct × log(likes+2)
    scored = []
    for row in rows:
        vid_id, niche, hook, scene, views, watch_pct, ctr, likes = row
        engagement = ctr * watch_pct * math.log(likes + 2)
        scored.append({
            "id":         vid_id,
            "niche":      niche,
            "hook":       hook or "",
            "scene":      scene or "",
            "engagement": engagement,
        })

    scored.sort(key=lambda x: x["engagement"], reverse=True)
    n      = len(scored)
    top_n  = max(1, n // 5)
    top_20 = scored[:top_n]
    bot_20 = scored[-top_n:]

    winning_hooks = [v["hook"] for v in top_20 if v["hook"]][:5]
    losing_hooks  = [v["hook"] for v in bot_20 if v["hook"]][:5]

    # Average engagement per niche
    niche_scores: dict[str, list] = {}
    for v in scored:
        niche_scores.setdefault(v["niche"], []).append(v["engagement"])
    avg_by_niche = {k: round(sum(vs) / len(vs), 4) for k, vs in niche_scores.items()}
    best_niches  = sorted(avg_by_niche, key=avg_by_niche.__getitem__, reverse=True)

    learnings = {
        "winning_hooks":         winning_hooks,
        "losing_hooks":          losing_hooks,
        "best_niches":           best_niches,
        "avg_score_by_niche":    avg_by_niche,
        "total_videos_analyzed": n,
    }

    LEARNINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNINGS_FILE.write_text(json.dumps(learnings, indent=2))

    print(f"[feedback] learnings.json updated – {n} videos analyzed")
    print(f"[feedback] best niches:  {best_niches}")
    print(f"[feedback] top hooks:    {winning_hooks[:3]}")
    print(f"[feedback] avoid hooks:  {losing_hooks[:3]}")


if __name__ == "__main__":
    analyze()
