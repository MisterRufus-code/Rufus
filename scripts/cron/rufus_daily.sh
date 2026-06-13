#!/usr/bin/env bash
# rufus_daily.sh — cron entry point for one scheduled Rufus run.
#
#   ./rufus_daily.sh              # default channel
#   ./rufus_daily.sh main_en      # explicit channel (multi-channel cron lines)
#
# flock is belt-and-suspenders over Rufus's own pidfile lock: if a previous
# run is still rendering, this invocation exits quietly instead of queuing.
set -u

RUFUS_DIR="${RUFUS_DIR:-$HOME/Rufus}"
CHANNEL="${1:-}"

export RUFUS_VIDEO_SOURCE="${RUFUS_VIDEO_SOURCE:-sd}"
export RUFUS_MIN_UPLOAD_SCORE="${RUFUS_MIN_UPLOAD_SCORE:-8}"

cd "$RUFUS_DIR" || exit 1

ARGS=(--scheduled)
if [ -n "$CHANNEL" ]; then
    ARGS+=(--channel "$CHANNEL")
fi

exec flock -n "/tmp/rufus.cron.lock" \
    "$RUFUS_DIR/venv/bin/python" scripts/main.py "${ARGS[@]}"
