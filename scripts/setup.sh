#!/usr/bin/env bash
# Rufus – one-shot server setup
# Run as: bash setup.sh
set -euo pipefail

MEDIA_DIR="$HOME/media_library"
SCRIPTS_DIR="$MEDIA_DIR/scripts"

echo "=== [1/5] System packages ==="
sudo apt-get update -q
sudo apt-get install -y ffmpeg python3-venv python3-pip openssh-server

echo "=== [2/5] Directory layout ==="
mkdir -p "$MEDIA_DIR"/{backgrounds,output,temp}
mkdir -p "$MEDIA_DIR/config"

echo "=== [3/5] Copying scripts ==="
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp -v "$REPO_DIR/scripts/audio_gen.py" "$SCRIPTS_DIR/"
cp -v "$REPO_DIR/scripts/blacklist.py" "$SCRIPTS_DIR/"
chmod +x "$SCRIPTS_DIR/audio_gen.py" "$SCRIPTS_DIR/blacklist.py"

echo "=== [4/5] Python venv ==="
python3 -m venv "$MEDIA_DIR/venv"
"$MEDIA_DIR/venv/bin/pip" install --upgrade pip -q
"$MEDIA_DIR/venv/bin/pip" install -r "$REPO_DIR/scripts/requirements.txt"

echo "=== [5/5] SSH key for n8n ==="
KEY="$HOME/.ssh/rufus_n8n_ed25519"
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "n8n-rufus"
  cat "$KEY.pub" >> "$HOME/.ssh/authorized_keys"
  chmod 600 "$HOME/.ssh/authorized_keys"
  echo ""
  echo ">>> Paste this private key into n8n credential 'rufusSsh':"
  cat "$KEY"
else
  echo "Key already exists at $KEY"
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Drop a background MP4 into $MEDIA_DIR/backgrounds/next_bg.mp4"
echo "  2. Import n8n/shorts_workflow.json into n8n"
echo "  3. Fill in: Pixabay API key, OpenAI API key, Telegram bot token"
echo "  4. In n8n, create SSH credential 'rufusSsh' with the key printed above"
