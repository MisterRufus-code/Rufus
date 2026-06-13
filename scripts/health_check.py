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
import os
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
    def _binary_ok(name: str) -> bool:
        if shutil.which(name) is None:
            return False
        try:
            return subprocess.run([name, "-version"], capture_output=True).returncode == 0
        except (FileNotFoundError, OSError):
            return False

    for binary in ("ffmpeg", "ffprobe"):
        if _binary_ok(binary):
            ok(f"{binary} installed")
        else:
            err(f"{binary} installed", f"{binary} not found in PATH — run: sudo apt install ffmpeg")

    # ── config/keys.json ────────────────────────────────────────────────────────
    if not KEYS_FILE.exists():
        err("config/keys.json", "file not found – create it from config/keys.json.template")
    else:
        ok("config/keys.json exists")
        try:
            keys = json.loads(KEYS_FILE.read_text())
            oai  = keys.get("openai", "")
            if oai.startswith("sk-") and len(oai) > 20 and not oai.startswith("YOUR_"):
                ok("OpenAI key set")
            else:
                err("OpenAI key", "not set or placeholder – paste your real sk-... key into config/keys.json")

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
    required = ["edge_tts", "faster_whisper", "openai", "httpx", "PIL", "google.auth"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if not missing:
        ok(f"Python packages  ({', '.join(required)})")
    else:
        err("Python packages", f"missing: {', '.join(missing)}  →  pip install -r requirements.txt")

    # ── Optional backends (informational) ─────────────────────────────────────────
    try:
        __import__("TTS")
        ok("XTTS v2 available  (RUFUS_TTS=xtts for local voice)")
    except ImportError:
        warn("XTTS v2", "not installed – using Edge TTS (pip install TTS for local voice cloning)")

    # ── HyperFrames (only relevant if a niche uses it) ────────────────────────────
    try:
        import json as _json
        niches = _json.loads((CONFIG_DIR / "niches.json").read_text()).get("niches", {})
        uses_hf = any(n.get("video_source") == "hyperframes" for n in niches.values())
        if uses_hf:
            import subprocess as _sp
            cmd = os.environ.get("HYPERFRAMES_CMD", "npx --yes hyperframes").split()
            try:
                r = _sp.run(cmd + ["--version"], capture_output=True, text=True, timeout=120)
                if r.returncode == 0:
                    ok("HyperFrames available  (motion-graphic niches)")
                else:
                    warn("HyperFrames", "not runnable – those niches fall back to SD/Pexels "
                                        "(install Node 22+; `npx hyperframes` auto-fetches)")
            except Exception:
                warn("HyperFrames", "Node/npx not found – motion-graphic niches fall back "
                                    "to SD/Pexels (install Node.js 22+)")
    except Exception:
        pass

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
