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

echo "=== [1/7] System packages (ffmpeg, python venv, fonts) ==="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo apt-get install -y ffmpeg python3-venv python3-pip fonts-dejavu-core
else
  echo "  ⚠ apt-get not found — install ffmpeg + python3-venv manually for your distro"
fi

echo "=== [2/7] Media directories ==="
mkdir -p "$REPO_DIR/media_library"/{output,temp,music,cache}

echo "=== [3/7] Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/scripts/requirements.txt"

echo "=== [4/7] config/keys.json ==="
if [ ! -f "$REPO_DIR/config/keys.json" ]; then
  cp "$REPO_DIR/config/keys.json.template" "$REPO_DIR/config/keys.json"
  echo "  → created config/keys.json from template — fill in your real keys"
else
  echo "  → config/keys.json already exists (left untouched)"
fi

echo "=== [5/7] Initialize database ==="
"$VENV_DIR/bin/python" "$REPO_DIR/scripts/db_manager.py"

echo "=== [6/7] Remotion renderer (optional) ==="
if command -v npm >/dev/null 2>&1; then
  if [ -d "$REPO_DIR/remotion/node_modules" ]; then
    echo "  → remotion/node_modules already installed — reusing"
  else
    echo "  → npm found — installing Remotion dependencies..."
    if (cd "$REPO_DIR/remotion" && npm install); then
      echo "  → Remotion ready. Enable it with: RUFUS_RENDERER=remotion"
    else
      echo "  ⚠ npm install failed — Remotion renderer won't be available (ffmpeg still works)"
    fi
  fi
else
  echo "  ⚠ npm not found — Remotion renderer skipped (ffmpeg still works)"
  echo "    Want it later? install Node.js, then: cd remotion && npm install"
fi

echo "=== [7/7] Health check ==="
"$VENV_DIR/bin/python" "$REPO_DIR/scripts/health_check.py" || true

# PRESENT IS NOT THE SAME AS WORKING, and a setup script that ends on a
# presence check has not finished the job. This encodes a real second of
# H.264, builds a real subtitle file, writes and reads a real row and serves a
# real page — in a few seconds, and without spending a penny.
echo
echo "=== Does it actually work? ==="
"$VENV_DIR/bin/python" "$REPO_DIR/scripts/smoke.py" || true

cat <<EOF

=== Setup complete ===
Next steps:
  1. Edit config/keys.json — add your OpenAI key (required) and Pexels key.
  2. Activate the venv:   source venv/bin/activate
  3. Test render:         python scripts/main.py --skip-upload
  4. Optional engines:
       RUFUS_RENDERER=remotion   (step 6 above already installed it if npm was found)
       RUFUS_VIDEO_SOURCE=sd     (run Automatic1111 with --api)
       RUFUS_TTS=xtts            (pip install TTS for local voice)
EOF
