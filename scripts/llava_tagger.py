#!/usr/bin/env python3
"""
llava_tagger.py
Extracts frames from videos and sends them to GPT-4o Vision for scene descriptions.
Includes multi-candidate selection: describe all 5, GPT picks the most viral one.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_client() -> OpenAI:
    keys = json.loads((CONFIG_DIR / "keys.json").read_text())
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("OpenAI key not set in config/keys.json")
    return OpenAI(api_key=key)


def extract_frame(video_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()

    for seek in ("00:00:03", "00:00:01", "00:00:00"):
        cmd = [
            "ffmpeg", "-y", "-ss", seek,
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and Path(tmp.name).stat().st_size > 1000:
            return Path(tmp.name)

    raise RuntimeError(f"ffmpeg frame extract failed for {video_path.name}")


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def describe_frame(image_b64: str, context_prompt: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": context_prompt},
            {"type": "image_url", "image_url": {
                "url":    f"data:image/jpeg;base64,{image_b64}",
                "detail": "low",   # ~$0.0004/image – sufficient for scene ID
            }},
        ]}],
        max_tokens=200,
        temperature=0.3,
    )
    result = resp.choices[0].message.content.strip()
    if not result:
        raise ValueError("GPT-4o Vision returned empty response")
    return result


def tag_video(video_path: Path, llava_context: str, client: OpenAI = None) -> str:
    """Extract one frame and describe it with GPT-4o Vision."""
    if client is None:
        client = _load_client()

    print(f"[vision] extracting frame from {video_path.name}...")
    frame_path = extract_frame(video_path)
    try:
        print("[vision] describing with GPT-4o...")
        image_b64   = image_to_base64(frame_path)
        description = describe_frame(image_b64, llava_context, client)
        short       = description[:120] + "..." if len(description) > 120 else description
        print(f"[vision] {short}")
        return description
    finally:
        try:
            os.unlink(frame_path)
        except Exception:
            pass


def pick_best_video(candidates: list[Path], llava_context: str) -> tuple[Path, str]:
    """
    Describe all candidate videos, ask GPT which is most viral.
    Returns (chosen_path, scene_description).
    """
    client = _load_client()

    niches     = json.loads((CONFIG_DIR / "niches.json").read_text())
    active     = niches["active"]
    niche_name = niches["niches"][active]["display_name"]

    descriptions: list[str] = []
    valid_paths: list[Path]  = []

    for i, path in enumerate(candidates):
        try:
            print(f"[vision] candidate {i+1}/{len(candidates)}")
            desc = tag_video(path, llava_context, client=client)
            descriptions.append(desc)
            valid_paths.append(path)
        except Exception as e:
            print(f"[vision] candidate {i+1} skipped: {e}")

    if not descriptions:
        raise RuntimeError("GPT-4o Vision failed to describe any candidate")

    if len(descriptions) == 1:
        return valid_paths[0], descriptions[0]

    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    prompt   = (
        f"You are choosing the best background video for a viral {niche_name} YouTube Short.\n\n"
        f"Here are {len(descriptions)} video scene descriptions:\n{numbered}\n\n"
        "Which would make the most engaging, viral Short for this niche?\n"
        "Reply with ONLY: NUMBER|REASON (e.g. '3|Trading screen – creates instant tension')"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=80,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"[gpt] video pick: {answer}")

    try:
        idx = int(answer.split("|")[0].strip()) - 1
        idx = max(0, min(idx, len(valid_paths) - 1))
    except Exception:
        idx = 0

    return valid_paths[idx], descriptions[idx]


def load_niche_context() -> str:
    niches = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = niches["active"]
    return niches["niches"][active]["llava_context"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llava_tagger.py <video_path>")
        sys.exit(1)
    video       = Path(sys.argv[1])
    context     = load_niche_context()
    description = tag_video(video, context)
    print(f"\nDESCRIPTION={description}")
