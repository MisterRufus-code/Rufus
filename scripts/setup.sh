#!/usr/bin/env bash
# Rufus — one-shot local setup (run from anywhere inside the repo)
#   bash scripts/setup.sh
#
# Installs system deps, creates a venv, installs Python requirements, downloads
# the caption font, initializes the SQLite DB, and scaffolds config/keys.json.
# Idempotent: safe to re-run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"

echo "=== [1/6] System packages (ffmpeg, python venv, fonts) ==="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo apt-get install -y ffmpeg python3-venv python3-pip fonts-dejavu-core
else
  echo "  ⚠ apt-get not found — install ffmpeg + python3-venv manually for your distro"
fi

echo "=== [2/6] Media directories ==="
mkdir -p "$REPO_DIR/media_library"/{output,temp,music,cache}

echo "=== [3/6] Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/scripts/requirements.txt"

echo "=== [4/6] config/keys.json ==="
if [ ! -f "$REPO_DIR/config/keys.json" ]; then
  cp "$REPO_DIR/config/keys.json.template" "$REPO_DIR/config/keys.json"
  echo "  → created config/keys.json from template — fill in your real keys"
else
  echo "  → config/keys.json already exists (left untouched)"
fi

echo "=== [5/6] Initialize database ==="
"$VENV_DIR/bin/python" "$REPO_DIR/scripts/db_manager.py"

echo "=== [6/6] Health check ==="
"$VENV_DIR/bin/python" "$REPO_DIR/scripts/health_check.py" || true

cat <<EOF

=== Setup complete ===
Next steps:
  1. Edit config/keys.json — add your OpenAI key (required) and Pexels key.
  2. Activate the venv:   source venv/bin/activate
  3. Test render:         python scripts/main.py --skip-upload
  4. Optional engines:
       RUFUS_RENDERER=remotion   (cd remotion && npm install first)
       RUFUS_VIDEO_SOURCE=sd     (run Automatic1111 with --api)
       RUFUS_TTS=xtts            (pip install TTS for local voice)
EOF
