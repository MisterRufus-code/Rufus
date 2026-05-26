#!/usr/bin/env python3
"""
llava_tagger.py
Extracts a frame from a video and sends it to LLaVA (via Ollama)
to get a scene description used for script writing.

Usage:
    python llava_tagger.py /path/to/video.mp4
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

OLLAMA_URL  = "http://localhost:11434/api/generate"
LLAVA_MODEL = "llava"          # change to "llava:13b" if you have it
FRAME_TIME  = "00:00:03"       # grab frame at 3 seconds in


def extract_frame(video_path: Path) -> Path:
    """Use ffmpeg to pull one frame as JPEG."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-ss", FRAME_TIME,
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        tmp.name
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: try frame at 0 seconds
        cmd[3] = "00:00:00"
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extract failed: {result.stderr.decode()}")

    return Path(tmp.name)


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def describe_frame(image_b64: str, context_prompt: str) -> str:
    """Send frame to LLaVA and get scene description."""
    payload = {
        "model":  LLAVA_MODEL,
        "prompt": context_prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temp = factual description
            "num_predict": 200,
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def tag_video(video_path: Path, llava_context: str) -> str:
    """
    Full pipeline: video → frame → LLaVA → scene description string.
    Returns the description text.
    """
    print("[llava] extracting frame...")
    frame_path = extract_frame(video_path)

    try:
        print("[llava] sending to Ollama...")
        image_b64   = image_to_base64(frame_path)
        description = describe_frame(image_b64, llava_context)
        print(f"[llava] description: {description}")
        return description
    finally:
        os.unlink(frame_path)   # clean up temp frame


def load_niche_context() -> str:
    config_dir  = Path(__file__).parent.parent / "config"
    niches      = json.loads((config_dir / "niches.json").read_text())
    active      = niches["active"]
    return niches["niches"][active]["llava_context"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llava_tagger.py <video_path>")
        sys.exit(1)

    video      = Path(sys.argv[1])
    context    = load_niche_context()
    description = tag_video(video, context)
    print(f"\nDESCRIPTION={description}")
