#!/bin/bash
set -euo pipefail

# Only run in remote (web) sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Pull latest code from the current tracking branch
echo "[session-start] pulling latest code..."
git pull origin "$(git rev-parse --abbrev-ref HEAD)" || echo "[session-start] git pull failed — continuing anyway"

# Install/sync Python dependencies
if [ -f requirements.txt ]; then
  echo "[session-start] installing dependencies..."
  pip install -q -r requirements.txt
fi

echo "[session-start] ready"
