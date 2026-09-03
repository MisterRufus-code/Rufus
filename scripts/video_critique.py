#!/usr/bin/env python3
"""
video_critique.py — free, local AI critique of a finished Rufus video.

Standalone tool, not wired into the pipeline: run it by hand after a video
renders, get a plain-text quality report. No OpenAI/Claude API call, no
per-video cost — uses a LOCAL vision-language model via Ollama (free,
zero-marginal-cost, runs on the same RTX 3090 already doing everything
else). Same philosophy as the rest of this project (local Whisper, local
Realistic Vision/Z-Image, local Edge/Kokoro TTS): pay once in setup time,
never per run.

Why Ollama specifically: it's the simplest way to run a real vision model
locally on Windows — one installer, one `ollama pull <model>`, then a
plain REST API on localhost. No Python ML stack to wrangle, no manual
checkpoint placement.

Setup (one-time, free):
  1. Install Ollama: https://ollama.com/download (Windows installer)
  2. Pull a vision model — recommended, in order:
       ollama pull llama3.2-vision      (11B, strong description quality,
                                          fits a 24GB card comfortably)
       ollama pull llava                (lighter/faster if VRAM is tight
                                          or ComfyUI is running alongside)
  3. Ollama runs as a background service after install — nothing else to
     start. Point OLLAMA_HOST at it if it's not on localhost.

Usage:
  python scripts/video_critique.py media_library/output/some.mp4
  python scripts/video_critique.py some.mp4 "the full script text"

Prints a structured report to stdout and saves it next to the video as
<video>.critique.txt. Returns None (prints setup instructions instead of a
report) if Ollama isn't running or the model hasn't been pulled — this
must never be mistaken for "the video is fine," so the caller always sees
which of the two happened.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

FRAME_COUNT = 6          # sampled evenly across the video's duration
OLLAMA_TIMEOUT = 180      # a multi-image vision prompt is slow on CPU fallback
REPORT_SECTIONS = (
    "HOOK (first frame)", "PACING / VISUAL VARIETY",
    "IMAGE-NARRATION MATCH", "VISIBLE AI ARTIFACTS",
    "CAPTION LEGIBILITY", "OVERALL VERDICT",
)


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _model() -> str:
    return os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")


def is_available() -> bool:
    try:
        r = requests.get(f"{_host()}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _pulled_models() -> list[str]:
    try:
        r = requests.get(f"{_host()}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def _model_ready(model: str) -> bool:
    """True if `model` (or model:tag) is pulled. Ollama lists exact tags
    ('llama3.2-vision:latest') so a bare name must match by prefix too."""
    pulled = _pulled_models()
    return any(p == model or p.startswith(f"{model}:") for p in pulled)


def _probe_duration(video_path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                d = float(stream.get("duration", 0))
                if d > 0:
                    return d
    except Exception:
        pass
    return 20.0


def extract_frames(video_path: Path, count: int = FRAME_COUNT) -> list[Path]:
    """count frames sampled evenly across the video (excludes the very first
    and last instants, which are often a hard cut/fade)."""
    duration = _probe_duration(video_path)
    frames: list[Path] = []
    for i in range(count):
        ts = duration * (i + 1) / (count + 1)
        tmp = Path(tempfile.mkstemp(suffix=f"_{i}.jpg")[1])
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ts:.2f}",
             "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(tmp)],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
            frames.append(tmp)
        else:
            tmp.unlink(missing_ok=True)
    return frames


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_prompt(script: str) -> str:
    script_block = f'\nNARRATION SCRIPT (for context — judge the IMAGES against this):\n"""\n{script.strip()}\n"""\n' if script.strip() else ""
    sections = "\n".join(f"- {s}" for s in REPORT_SECTIONS)
    return (
        "You are reviewing frames sampled evenly across a finished vertical "
        "YouTube Short (a faceless, AI-generated explainer video) for a "
        "channel owner who wants an honest quality critique before "
        "publishing more like it.\n"
        f"{script_block}\n"
        "Write a short report with EXACTLY these section headers, each "
        "1-3 sentences, blunt and specific (name what you actually see in "
        "the frames, not generic advice):\n"
        f"{sections}\n\n"
        "Be honest about weaknesses — this is for improving future videos, "
        "not for reassurance."
    )


def critique_video(video_path: Path, script: str = "", model: str = None,
                   frame_count: int = FRAME_COUNT) -> dict | None:
    """Sample frames from video_path, send them + script to a local Ollama
    vision model, return {"report": str, "model": str, "frame_count": int}.
    Returns None (after printing exactly why) if Ollama isn't reachable or
    the model isn't pulled — never returns a fabricated report."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[critique] video not found: {video_path}")
        return None

    if not is_available():
        print(f"[critique] Ollama not reachable at {_host()} — install it "
              f"(https://ollama.com/download) and make sure it's running, "
              f"or set OLLAMA_HOST. See this file's module docstring.")
        return None

    model = model or _model()
    if not _model_ready(model):
        print(f"[critique] model '{model}' not pulled yet — run:\n"
              f"    ollama pull {model}\n"
              f"(one-time, free — the whole point of this tool). "
              f"Or set OLLAMA_VISION_MODEL to a model you've already pulled.")
        return None

    frames = extract_frames(video_path, count=frame_count)
    if not frames:
        print(f"[critique] couldn't extract any frames from {video_path.name} "
              f"(ffmpeg/ffprobe failed) — check the file isn't corrupt.")
        return None

    try:
        images_b64 = [_encode(f) for f in frames]
        r = requests.post(
            f"{_host()}/api/generate",
            json={"model": model, "prompt": _build_prompt(script),
                  "images": images_b64, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        report = (r.json().get("response") or "").strip()
    except Exception as e:
        print(f"[critique] Ollama request failed: {e}")
        return None
    finally:
        for f in frames:
            f.unlink(missing_ok=True)

    if not report:
        print("[critique] Ollama returned an empty response — try again "
              "or a different OLLAMA_VISION_MODEL.")
        return None

    return {"report": report, "model": model, "frame_count": len(frames)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/video_critique.py <video.mp4> [\"<script text>\"]")
        sys.exit(1)

    video = Path(sys.argv[1])
    script_text = sys.argv[2] if len(sys.argv) > 2 else ""

    result = critique_video(video, script=script_text)
    if result is None:
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AI CRITIQUE — {video.name}  (model: {result['model']}, "
          f"{result['frame_count']} frames)")
    print(f"{'='*60}\n")
    print(result["report"])

    out_path = video.with_suffix(".critique.txt")
    try:
        out_path.write_text(result["report"], encoding="utf-8")
        print(f"\n[critique] saved → {out_path}")
    except OSError as e:
        print(f"\n[critique] couldn't save report file: {e}")
