#!/usr/bin/env python3
"""
health_check.py
Verifies all critical dependencies before a pipeline run.

Usage:
    python scripts/health_check.py

Exit 0 = all critical checks pass (warnings are printed but non-fatal).
Exit 1 = at least one critical check failed.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
KEYS_FILE   = CONFIG_DIR / "keys.json"
NICHES_FILE = CONFIG_DIR / "niches.json"


def run() -> None:
    ok_lines   : list[str] = []
    warn_lines : list[str] = []
    err_lines  : list[str] = []

    def ok(label: str) -> None:
        ok_lines.append(f"  ✓  {label}")

    def warn(label: str, msg: str) -> None:
        warn_lines.append(f"  ⚠  {label}: {msg}")

    def err(label: str, msg: str) -> None:
        err_lines.append(f"  ✗  {label}: {msg}")

    # ── FFmpeg ──────────────────────────────────────────────────────────────────
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if r.returncode == 0:
        ok("FFmpeg installed")
    else:
        err("FFmpeg installed", "ffmpeg not found in PATH")

    r = subprocess.run(["ffprobe", "-version"], capture_output=True)
    if r.returncode == 0:
        ok("ffprobe installed")
    else:
        err("ffprobe installed", "ffprobe not found in PATH")

    # ── config/keys.json ────────────────────────────────────────────────────────
    if not KEYS_FILE.exists():
        err("config/keys.json", "file not found – create it from config/keys.json.example")
    else:
        ok("config/keys.json exists")
        try:
            keys = json.loads(KEYS_FILE.read_text())
            oai  = keys.get("openai", "")
            if oai and not oai.startswith("YOUR_") and oai.startswith("sk-"):
                ok("OpenAI key set")
            else:
                err("OpenAI key", "not set or invalid – must start with 'sk-'")

            pexels = keys.get("pexels", "")
            if pexels and not pexels.startswith("YOUR_"):
                ok("Pexels key set")
            else:
                warn("Pexels key", "not set – Pexels source disabled")
        except Exception as e:
            err("config/keys.json parse", str(e))

    # ── config/niches.json ──────────────────────────────────────────────────────
    if not NICHES_FILE.exists():
        err("config/niches.json", "file not found")
    else:
        try:
            data   = json.loads(NICHES_FILE.read_text())
            active = data["active"]
            if active in data["niches"]:
                ok(f"config/niches.json valid  (active: {active})")
            else:
                err("config/niches.json", f"active='{active}' not in niches dict")
        except Exception as e:
            err("config/niches.json", str(e))

    # ── Disk space ──────────────────────────────────────────────────────────────
    usage   = shutil.disk_usage(str(ROOT))
    free_gb = usage.free / (1024 ** 3)
    if free_gb >= 1.0:
        ok(f"Disk space  ({free_gb:.1f} GB free)")
    else:
        err("Disk space", f"only {free_gb:.1f} GB free – need ≥1 GB")

    # ── Python packages ─────────────────────────────────────────────────────────
    missing = []
    for pkg in ["edge_tts", "faster_whisper", "openai", "google.auth"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if not missing:
        ok("Python packages  (edge_tts, faster_whisper, openai, google-auth)")
    else:
        err("Python packages", f"missing: {', '.join(missing)}  →  pip install -r requirements.txt")

    # ── YouTube OAuth token ─────────────────────────────────────────────────────
    token = CONFIG_DIR / "youtube_token.json"
    if token.exists():
        ok("YouTube OAuth token")
    else:
        warn("YouTube OAuth token", "not found – run main.py once manually to complete OAuth flow")

    # ── Print results ────────────────────────────────────────────────────────────
    print("\nRufus Health Check")
    print("=" * 42)
    for line in ok_lines:
        print(line)

    if warn_lines:
        print()
        for line in warn_lines:
            print(line)

    if err_lines:
        print()
        for line in err_lines:
            print(line)
        print("\n  Fix the errors above before running main.py\n")
        sys.exit(1)
    else:
        print(f"\n  All critical checks passed.  Ready to run.\n")


if __name__ == "__main__":
    run()
