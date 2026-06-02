#!/usr/bin/env bash
#
# setup_cloud.sh — One-paste setup for a fresh GPU instance (Vast.ai / RunPod).
#
# Installs ComfyUI + VideoHelperSuite, downloads the Wan2.1 text-to-video models,
# and prepares Rufus so you can produce viral Shorts end-to-end on the rented GPU.
#
# Usage on a fresh instance:
#     wget -qO- https://raw.githubusercontent.com/MisterRufus-code/Rufus/claude/nifty-darwin-6MeTJ/setup_cloud.sh | bash
#   or, if Rufus is already cloned:
#     bash setup_cloud.sh
#
# Idempotent: re-running skips anything already done (downloads resume with -c).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY_DIR="${COMFY_DIR:-$WORKSPACE/ComfyUI}"
RUFUS_DIR="${RUFUS_DIR:-$WORKSPACE/Rufus}"
RUFUS_REPO="https://github.com/MisterRufus-code/Rufus.git"
RUFUS_BRANCH="${RUFUS_BRANCH:-claude/nifty-darwin-6MeTJ}"

HF="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"

say() { echo -e "\n\033[1;36m== $* ==\033[0m"; }

dl() {  # dl <url> <dest>  — resumable, skips if already complete
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -gt 100000000 ]; then
    echo "  ✓ exists: $(basename "$dest")"
  else
    echo "  ↓ $(basename "$dest")"
    wget -c -q --show-progress "$url" -O "$dest"
  fi
}

mkdir -p "$WORKSPACE"

# ── 1. ComfyUI ───────────────────────────────────────────────────────────────
say "ComfyUI"
if [ ! -d "$COMFY_DIR" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi
pip install -q -r "$COMFY_DIR/requirements.txt"

# ── 2. VideoHelperSuite (mp4 output node used by the Rufus workflow) ──────────
say "VideoHelperSuite custom node"
VHS_DIR="$COMFY_DIR/custom_nodes/ComfyUI-VideoHelperSuite"
if [ ! -d "$VHS_DIR" ]; then
  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$VHS_DIR"
fi
[ -f "$VHS_DIR/requirements.txt" ] && pip install -q -r "$VHS_DIR/requirements.txt" || true

# ── 3. Wan2.1 models (14B — better quality; ~18GB total) ──────────────────────
say "Wan2.1 models (~18GB, resumable — 14B for better quality)"
dl "$HF/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"      "$COMFY_DIR/models/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"
dl "$HF/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"  "$COMFY_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
dl "$HF/vae/wan_2.1_vae.safetensors"                           "$COMFY_DIR/models/vae/wan_2.1_vae.safetensors"

# ── 4. Rufus ─────────────────────────────────────────────────────────────────
say "Rufus"
if [ ! -d "$RUFUS_DIR" ]; then
  git clone -b "$RUFUS_BRANCH" "$RUFUS_REPO" "$RUFUS_DIR"
fi
pip install -q -r "$RUFUS_DIR/requirements.txt"
# ffmpeg (NVENC build ships with most CUDA images; install if missing)
command -v ffmpeg >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg)

# ── 5. Keys template ─────────────────────────────────────────────────────────
if [ ! -f "$RUFUS_DIR/config/keys.json" ]; then
  say "Creating config/keys.json template — FILL IN YOUR KEYS"
  cat > "$RUFUS_DIR/config/keys.json" <<'JSON'
{
  "openai_api_key": "sk-...",
  "pexels_api_key": "",
  "youtube_client_secret_file": "config/client_secret.json"
}
JSON
fi

cat <<EOF

\033[1;32m✓ Setup complete.\033[0m

Next:
  1. Edit keys:        nano $RUFUS_DIR/config/keys.json
  2. Start ComfyUI:    cd $COMFY_DIR && python main.py --listen 0.0.0.0 --port 8188 &
  3. Produce a Short:  cd $RUFUS_DIR && \\
                       RUFUS_GPU=1 RUFUS_VIDEO_SOURCE=comfy python scripts/main.py --skip-upload

When the videos look right, drop --skip-upload to auto-upload the good ones to YouTube.
EOF
