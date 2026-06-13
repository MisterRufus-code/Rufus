# Rufus Operations — running the machine unattended

The pipeline runs itself via cron (see `docs/CRONTAB.md`). Your job is a short
daily glance and a weekly steer. The system already enforces the quality gate,
the analytics→learnings flywheel, anti-repetition memory, and a single-instance
lock — so most days there is nothing to do.

## Daily (≈5 minutes)

1. `venv/bin/python scripts/report.py` — last 7 days at a glance. Check:
   - `live` count matches how many videos you expected (2/day default).
   - any `upload-failed` rows → open YouTube Studio and verify before re-uploading
     (a failed upload may have partially gone through — never blind-retry).
   - `held-for-review` rows → scripts that scored below the gate; they're in
     `media_library/output/<channel>/` if you want to salvage one manually.
2. In YouTube Studio: confirm the scheduled videos published, and **pin the CTA
   comment** Rufus posted (the API can't pin Shorts comments — this is the one
   manual touch). Reply to 1–3 real comments — early engagement is a ranking signal.
3. Glance at `logs/rufus_YYYYMMDD.log` for any `✗` lines.

## Weekly (≈15 minutes)

1. `venv/bin/python scripts/feedback_analyzer.py` (cron does this Sunday; run it
   manually if you skipped). Then read `config/learnings.json` (or the channel's
   `config/channels/<id>/learnings.json`): which hooks/titles won, which lost,
   best niches. The script writer already injects these into new prompts.
2. `venv/bin/python scripts/report.py --weeks 4 --correlate` — confirm higher
   script scores actually correlate with more views (validates the quality gate).
3. Review held videos; delete dead weight in `media_library/output/`.

## Monthly

- Check OpenAI spend (≈$5–13/mo at 2 videos/day).
- Nudge the niche `schedule` in `niches.json` (or the channel) toward the
  `best_niches` from learnings.

## When you add a second channel (see the scaling plan)

1. Create the YouTube account + Google Cloud OAuth client (or reuse the project
   for channels 1–3 — quota is ~6 uploads/day/project).
2. Copy `config/channels.json.example` → `config/channels.json`, add a block.
3. Put that channel's `client_secrets.json` under `config/channels/<id>/` and run
   one manual upload to OAuth (creates `youtube_token.json`).
4. Add one cron line: `rufus_daily.sh <id>`, staggered ≥1h from other channels.

## Anti-termination rules (do not skip these)

- One Google account per channel. Never the same/near-identical video on two
  channels (the global `blacklist.json` + `used_seeds.json` guard this — keep
  them global).
- Warm up a new channel: full branding before video 1, 1/day week 1, ramp to
  2/day after. Never launch two channels the same week.
- Cap 3/day/channel. Honest titles/thumbnails (clickbait that mismatches content
  craters watch%, which is the metric that gets channels swept).
- Never buy subs/views/engagement, never cross-comment between your own channels.
