# Rufus crontab — fully unattended daily production

Install with `crontab -e` and paste (adjust `~` paths if Rufus lives elsewhere):

```cron
# ── Render + upload: 2 videos/day. Render times are staggered so each video
#    lands in a different peak publish slot (publishAt picks the NEXT peak hour).
30 6  * * *  ~/Rufus/scripts/cron/rufus_daily.sh
30 14 * * *  ~/Rufus/scripts/cron/rufus_daily.sh

# ── Metrics in, learnings out
0  10 * * *  cd ~/Rufus && venv/bin/python scripts/analytics_fetcher.py >> logs/cron_analytics.log 2>&1
15 10 * * 0  cd ~/Rufus && venv/bin/python scripts/feedback_analyzer.py >> logs/cron_feedback.log  2>&1

# ── Pre-flight: catches dead A1111, missing keys, low disk before render time
0  6  * * *  cd ~/Rufus && venv/bin/python scripts/health_check.py >> logs/cron_health.log 2>&1
```

## Multi-channel (when you add channel #2)

One wrapper line per channel per slot, **staggered ≥1 hour apart** (one GPU =
one render at a time, and staggering keeps upload patterns human):

```cron
30 6  * * *  ~/Rufus/scripts/cron/rufus_daily.sh main_en
30 8  * * *  ~/Rufus/scripts/cron/rufus_daily.sh second_channel
30 14 * * *  ~/Rufus/scripts/cron/rufus_daily.sh main_en
30 16 * * *  ~/Rufus/scripts/cron/rufus_daily.sh second_channel
```

## Requirements for unattended runs

- A1111 must auto-start with the machine (`./webui.sh --api --xformers --medvram`)
  — add it to your desktop session autostart or a systemd user unit. If it's
  down, Rufus falls back to Pexels stock automatically (run still completes).
- One manual upload first so `config/.../youtube_token.json` exists — cron can
  refresh tokens but cannot do the first browser OAuth.
- Quota: YouTube allows ~6 uploads/day per Google Cloud project (1,600 units ×
  6 = 9,600 of 10,000). Stay at ≤3/day/channel regardless — anti-spam.
