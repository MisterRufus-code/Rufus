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
            return subprocess.run([name, "-version"], capture_output=True, timeout=10).returncode == 0
        except (FileNotFoundError, OSError):
            return False

    _ffmpeg_hint = (
        "not found in PATH — download from https://www.gyan.dev/ffmpeg/builds/, unzip, "
        "and add the \\bin folder to PATH (then reopen the terminal)"
        if os.name == "nt" else
        "not found in PATH — run: sudo apt install ffmpeg"
    )
    for binary in ("ffmpeg", "ffprobe"):
        if _binary_ok(binary):
            ok(f"{binary} installed")
        else:
            err(f"{binary} installed", f"{binary} {_ffmpeg_hint}")

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

            reddit_id = keys.get("reddit_client_id", "")
            if reddit_id and not reddit_id.startswith("YOUR_"):
                ok("Reddit API key set  (richer research seeds)")
            else:
                warn("Reddit key", "not set – Reddit source falls back to StackExchange/HN/RSS "
                                  "(add reddit_client_id + reddit_client_secret to keys.json)")
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

    # ── Voice backends (informational) ─────────────────────────────────────────────
    tts_backend = os.environ.get("RUFUS_TTS", "edge").strip().lower()
    try:
        keys = json.loads(KEYS_FILE.read_text()) if KEYS_FILE.exists() else {}
    except Exception:
        keys = {}
    eleven = keys.get("elevenlabs", "")
    if eleven and not eleven.startswith("YOUR_"):
        ok(f"ElevenLabs key set  (RUFUS_TTS=elevenlabs for best voice){'  ← ACTIVE' if tts_backend == 'elevenlabs' else ''}")
    elif tts_backend == "elevenlabs":
        err("ElevenLabs key", "RUFUS_TTS=elevenlabs but no key in keys.json — will fall back to Edge TTS")
    else:
        warn("ElevenLabs key", "not set – set it + RUFUS_TTS=elevenlabs for the most natural voice (~$0.10/video)")

    try:
        import kokoro as _kokoro_mod  # noqa: F401
        ok(f"Kokoro TTS available  (RUFUS_TTS=kokoro, voice: {os.environ.get('RUFUS_KOKORO_VOICE','am_adam')}){'  ← ACTIVE' if tts_backend == 'kokoro' else ''}")
    except ImportError:
        if tts_backend == "kokoro":
            err("Kokoro TTS", "RUFUS_TTS=kokoro but kokoro not installed — pip install kokoro>=0.9.2 soundfile")
        else:
            warn("Kokoro TTS", "not installed — pip install kokoro>=0.9.2 soundfile for a free local voice (RUFUS_TTS=kokoro)")

    try:
        __import__("TTS")
        ok(f"XTTS v2 available  (RUFUS_TTS=xtts for free local voice){'  ← ACTIVE' if tts_backend == 'xtts' else ''}")
    except ImportError:
        warn("XTTS v2", "not installed – pip install TTS for free local voice cloning (RUFUS_TTS=xtts)")

    # ── MusicGen (informational — optional AI music backend) ────────────────────
    try:
        from audiocraft.models import MusicGen as _MG  # noqa: F401
        ok("MusicGen available  (Meta AudioCraft — AI-generated on-tone music)")
    except ImportError:
        warn("MusicGen", "not installed — pip install audiocraft for AI music generation (optional)")

    # ── Diffusers (in-process SD — no A1111 server needed) ──────────────────────
    video_src = os.environ.get("RUFUS_VIDEO_SOURCE", "sd").lower()
    try:
        import diffusers as _diff  # noqa: F401
        model_key = os.environ.get("RUFUS_DIFFUSERS_MODEL", "sdxl-turbo")
        ok(f"Diffusers available  ({model_key}, RUFUS_VIDEO_SOURCE=diffusers){'  ← ACTIVE' if video_src == 'diffusers' else ''}")
    except ImportError:
        if video_src == "diffusers":
            err("Diffusers", "RUFUS_VIDEO_SOURCE=diffusers but diffusers not installed — "
                "pip install diffusers transformers accelerate")
        else:
            warn("Diffusers", "not installed — pip install diffusers transformers accelerate "
                 "(enables RUFUS_VIDEO_SOURCE=diffusers, no A1111 server needed)")

    # ── pytrends (Google Trends trending signal — optional) ──────────────────────
    try:
        from pytrends.request import TrendReq as _TR  # noqa: F401
        ok("pytrends available  (Google Trends signal — timely hooks)")
    except ImportError:
        warn("pytrends", "not installed — pip install pytrends>=4.9.0 for trending-topic hook boost (optional)")

    # ── ComfyUI + FLUX (only relevant if a niche uses it or it's selected) ───────
    try:
        _niches = json.loads(NICHES_FILE.read_text()).get("niches", {})
        uses_comfy = (video_src == "comfy"
                      or any(n.get("video_source") == "comfy" for n in _niches.values()))
        if uses_comfy:
            import requests as _rq
            chost = os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")
            try:
                r = _rq.get(f"{chost}/system_stats", timeout=5)
                if r.status_code == 200:
                    ok(f"ComfyUI available ({chost})  ← FLUX engine"
                       + ("  ← ACTIVE" if video_src == "comfy" else ""))
                    # Server up ≠ model loadable — verify the configured
                    # checkpoint is actually in ComfyUI's list (the classic
                    # first-run failure is the file in the wrong folder).
                    try:
                        sys.path.insert(0, str(Path(__file__).parent))
                        from comfy_client import list_checkpoints as _lc
                        _active = json.loads(NICHES_FILE.read_text())
                        _acfg   = _active["niches"].get(
                            os.environ.get("RUFUS_NICHE_OVERRIDE") or _active["active"], {})
                        _model  = (os.environ.get("COMFY_MODEL")
                                   or _acfg.get("comfy_model")
                                   or "flux1-dev-fp8.safetensors")
                        _ckpts = _lc()
                        if _ckpts and _model in _ckpts:
                            ok(f"FLUX checkpoint loadable  ({_model})")
                        elif _ckpts:
                            msg = (f"'{_model}' not in ComfyUI's checkpoint list "
                                   f"(sees: {', '.join(_ckpts[:4])}) — put the file in "
                                   f"ComfyUI\\models\\checkpoints\\ or set COMFY_MODEL")
                            err("FLUX checkpoint", msg) if video_src == "comfy" \
                                else warn("FLUX checkpoint", msg)
                    except Exception:
                        pass
                else:
                    warn("ComfyUI", f"responded {r.status_code} at {chost} — check startup")
            except Exception:
                msg = (f"not running at {chost} — start ComfyUI with --listen "
                       f"(comfy niches fall back to SD/Pexels)")
                err("ComfyUI", msg) if video_src == "comfy" else warn("ComfyUI", msg)
    except Exception:
        pass

    # ── Kokoro-FastAPI (only checked when selected) ──────────────────────────────
    if tts_backend == "kokoro_api":
        import requests as _rq
        kurl = os.environ.get("KOKORO_API_URL", "http://localhost:8880").rstrip("/")
        try:
            r = _rq.get(f"{kurl}/health", timeout=5)
            if r.status_code == 200:
                ok(f"Kokoro-FastAPI reachable ({kurl})  ← ACTIVE")
            else:
                warn("Kokoro-FastAPI", f"responded {r.status_code} at {kurl} — check the container")
        except Exception:
            err("Kokoro-FastAPI", f"RUFUS_TTS=kokoro_api but nothing at {kurl} — start it: "
                f"docker run -d -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2")

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

    # ── Stable Diffusion host (non-fatal — A1111 may not be running yet) ─────────
    try:
        import urllib.request as _urlreq
        sd_host = os.environ.get("SD_HOST", "http://127.0.0.1:7860")
        _urlreq.urlopen(f"{sd_host}/sdapi/v1/options", timeout=3)
        ok(f"Stable Diffusion API reachable  ({sd_host})")
    except Exception:
        warn("Stable Diffusion API", f"not reachable at {os.environ.get('SD_HOST','http://127.0.0.1:7860')} "
                                     "– start A1111 before running or set SD_HOST env var "
                                     "(pipeline falls back to Pexels)")

    # ── Database writability ─────────────────────────────────────────────────────
    try:
        import sqlite3 as _sqlite3
        db_path = ROOT / "rufus.db"
        conn = _sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _hc_probe (x INTEGER)")
        conn.execute("DROP TABLE IF EXISTS _hc_probe")
        conn.close()
        ok(f"Database writable  ({db_path.name})")
    except Exception as e:
        err("Database", f"not writable: {e} — check file permissions on rufus.db")

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
