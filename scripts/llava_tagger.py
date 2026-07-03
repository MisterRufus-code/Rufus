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


def _active_niche(data: dict) -> str:
    return os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]


def _load_client() -> OpenAI:
    keys = json.loads((CONFIG_DIR / "keys.json").read_text())
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("OpenAI key not set in config/keys.json")
    return OpenAI(api_key=key)


def extract_frame(video_path: Path) -> Path:
    """Single-frame fallback: try t=3s, t=1s, t=0s."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()

    for seek in ("00:00:03", "00:00:01", "00:00:00"):
        cmd = [
            "ffmpeg", "-y", "-ss", seek,
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and Path(tmp.name).stat().st_size > 1000:
            return Path(tmp.name)

    raise RuntimeError(f"ffmpeg frame extract failed for {video_path.name}")


def extract_frames(video_path: Path, count: int = 3) -> list[Path]:
    """Extract `count` frames at 20/50/80% of the video's duration."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    duration = 15.0
    try:
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                d = float(stream.get("duration", 0))
                if d > 0:
                    duration = d
                    break
    except Exception:
        pass

    fractions = [0.2, 0.5, 0.8][:count]
    frames: list[Path] = []
    for frac in fractions:
        ts  = max(0.5, duration * frac)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.2f}",
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and Path(tmp.name).stat().st_size > 1000:
            frames.append(Path(tmp.name))

    if not frames:
        frames.append(extract_frame(video_path))

    return frames


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def describe_frame(image_b64: str, context_prompt: str, client: OpenAI) -> str:
    """Single-frame describe (kept for backward compatibility)."""
    return describe_frames([image_b64], context_prompt, client)


def describe_frames(image_b64_list: list[str], context_prompt: str, client: OpenAI) -> str:
    """Describe multiple frames in one Vision call for richer scene understanding."""
    n       = len(image_b64_list)
    prefix  = (
        f"{context_prompt}\n\n"
        f"You are seeing {n} frame{'s' if n > 1 else ''} from this video"
        + (": beginning, middle, and end. Synthesize all frames into one description." if n > 1 else ".")
    )
    content = [{"type": "text", "text": prefix}]
    for b64 in image_b64_list:
        content.append({"type": "image_url", "image_url": {
            "url":    f"data:image/jpeg;base64,{b64}",
            "detail": "low",
        }})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=250,
        temperature=0.3,
    )
    result = resp.choices[0].message.content.strip()
    if not result:
        raise ValueError("GPT-4o Vision returned empty response")
    return result


def tag_video(video_path: Path, llava_context: str, client: OpenAI = None) -> str:
    """Extract 3 frames and describe them with GPT-4o Vision."""
    if client is None:
        client = _load_client()

    print(f"[vision] extracting frames from {video_path.name}...")
    frames = extract_frames(video_path)
    try:
        print(f"[vision] describing {len(frames)} frames with GPT-4o...")
        b64_list    = [image_to_base64(f) for f in frames]
        description = describe_frames(b64_list, llava_context, client)
        short       = description[:120] + "..." if len(description) > 120 else description
        print(f"[vision] {short}")
        return description
    finally:
        for f in frames:
            try:
                os.unlink(f)
            except Exception:
                pass


def pick_best_video(candidates: list[Path], llava_context: str,
                    seed: dict = None, analysis: str = None) -> tuple[Path, str]:
    """
    Describe all candidate videos, ask GPT which best matches the script topic.
    Pass analysis (from preanalyze()) for hook-angle-level matching; falls back
    to raw seed content when analysis is not available.
    Returns (chosen_path, scene_description).
    """
    client = _load_client()

    niches     = json.loads((CONFIG_DIR / "niches.json").read_text())
    active     = _active_niche(niches)
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

    # Build context for the picker — analysis gives hook angle + biographical facts
    # which is much more specific than raw seed title/content
    if analysis:
        seed_ctx = f"\nSCRIPT ANGLE (hook + core message from pre-analysis):\n{analysis[:500]}"
    elif seed:
        stype = seed.get("type", "")
        if stype == "reddit":
            seed_ctx = (
                f"\nSCRIPT TOPIC: \"{seed.get('title', '')}\"\n"
                f"{(seed.get('content') or '')[:250]}"
            )
        elif stype == "wisdom":
            seed_ctx = (
                f"\nSCRIPT TOPIC: Quote from {seed.get('source', 'Unknown')}:\n"
                f"\"{(seed.get('content') or '')[:200]}\""
            )
        elif stype in ("hackernews", "stackexchange"):
            seed_ctx = (
                f"\nSCRIPT TOPIC: \"{seed.get('title', '')}\"\n"
                f"{(seed.get('content') or '')[:250]}"
            )
        else:
            seed_ctx = ""
    else:
        seed_ctx = ""

    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    prompt   = (
        f"You are choosing the best background video for a viral {niche_name} YouTube Short.\n"
        f"{seed_ctx}\n\n"
        f"VIDEO OPTIONS:\n{numbered}\n\n"
        "SELECTION RULES — apply in order:\n"
        "1. STORY SUBJECT first: match the specific topic, not just the niche. "
        "A market crash story → trading screens with red; a frugality story → tools/repair work; "
        "a discipline story → athlete grinding; a wealth trap story → bills/debt, NOT a yacht.\n"
        "2. CONTRADICT the story if needed: a story about losing wealth should show loss, not luxury.\n"
        "3. MOOD match: tension/conflict → darker/dramatic footage; inspiration → bright/dynamic.\n"
        "4. REJECT any video whose visuals directly contradict the story's emotional tone.\n\n"
        "Reply with ONLY: NUMBER|REASON (e.g. '3|Trading screens with red — matches the crash story')"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=80,
        timeout=45,
    )
    answer = (resp.choices[0].message.content or "").strip()
    print(f"[gpt] video pick: {answer}")

    try:
        raw_idx = answer.split("|")[0].strip()
        idx = int("".join(c for c in raw_idx if c.isdigit()) or "1") - 1
        idx = max(0, min(idx, len(valid_paths) - 1))
    except Exception:
        idx = 0

    return valid_paths[idx], descriptions[idx]


def load_niche_context() -> str:
    niches = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = _active_niche(niches)
    return niches["niches"][active]["llava_context"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llava_tagger.py <video_path>")
        sys.exit(1)
    video       = Path(sys.argv[1])
    context     = load_niche_context()
    description = tag_video(video, context)
    print(f"\nDESCRIPTION={description}")
