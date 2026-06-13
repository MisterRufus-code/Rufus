#!/usr/bin/env python3
"""
hyperframes_client.py — motion-graphic video source for Rufus via HeyGen's
HyperFrames (open-source HTML→MP4 renderer, Apache-2.0).

An LLM (GPT, reusing the existing OpenAI key) writes a self-contained animated
HTML scene from each script-derived scene description; HyperFrames renders it to
a 1080×1920 MP4 with headless Chrome + FFmpeg. Output drops into the same
pipeline contract as sd_client:

    generate_clips(queries, n, clip_duration) -> list[Path]   # 1080×1920 mp4s

so audio_gen.render() composites voice + captions + ducked music on top exactly
as it does for SD or Pexels clips.

Why this source: CPU-only (no GPU contention with SD), $0 (no credits), fast,
and deterministic. Animated charts/gradients/counters read as "agency-grade"
for data niches (finance/business/mindset).

Requirements on the host: Node.js 22+ (the project already needs Node for the
remotion engine) and FFmpeg. HyperFrames itself is fetched on demand via npx.

Env:
  HYPERFRAMES_CMD   space-separated launcher (default: "npx --yes hyperframes")
  HF_RENDER_TIMEOUT seconds per scene render (default 180)
"""

import json
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"
TMP_DIR    = Path(__file__).parent.parent / "media_library" / "temp" / "hf"

OUT_W, OUT_H = 1080, 1920
MODEL        = "gpt-4o-mini"
MIN_BYTES    = 50_000


def _launcher() -> list[str]:
    return os.environ.get("HYPERFRAMES_CMD", "npx --yes hyperframes").split()


@lru_cache(maxsize=1)
def is_available() -> bool:
    """True if HyperFrames can be launched (Node present + package resolvable)."""
    try:
        r = subprocess.run(_launcher() + ["--version"],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def _load_key() -> str:
    try:
        key = json.loads(KEYS_FILE.read_text()).get("openai", "")
        if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
            return key
    except Exception:
        pass
    return ""


def _extract_html(raw: str) -> str | None:
    """Pull an HTML document out of an LLM reply; None if it isn't one."""
    raw = re.sub(r"^```(?:html)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    low = raw.lower()
    if "<html" not in low and "<!doctype" not in low:
        return None
    return raw


def _scene_prompt(scene_desc: str, niche_cfg: dict, duration: float) -> str:
    accent = niche_cfg.get("accent_color", "#FFD23F")
    mood   = niche_cfg.get("style_suffix", "cinematic, high contrast")
    return (
        "Write ONE self-contained HTML document that animates a looping motion-"
        "graphic BACKGROUND for a vertical short-form video. Output ONLY the HTML.\n\n"
        f"Scene concept: {scene_desc}\n"
        f"Mood/palette: {mood}. Accent color: {accent}. Dark, cinematic base.\n\n"
        "HARD RULES:\n"
        f"- Root stage element must carry: id=\"stage\" data-composition-id=\"scene\" "
        f"data-start=\"0\" data-duration=\"{duration:.1f}\" data-width=\"{OUT_W}\" "
        f"data-height=\"{OUT_H}\", and CSS sizing it to exactly {OUT_W}x{OUT_H}px.\n"
        "- Use ONLY inline CSS @keyframes / transforms / gradients / SVG. NO external "
        "scripts, NO CDN links, NO <img> with remote URLs (must render offline, "
        "deterministically).\n"
        "- ABSTRACT/ATMOSPHERIC ONLY — gradients, drifting shapes, particles, grids, "
        "soft glows, or simple animated bars/lines for data niches. Motion must be "
        "smooth and subtle, never strobing.\n"
        "- NO large text, NO words, NO headlines, NO logos. Spoken-word captions are "
        "overlaid by a later stage; on-screen text here would collide with them.\n"
        f"- Animation should fill the full {duration:.1f}s and hold gracefully at the end.\n"
        "- Everything in one file: <!DOCTYPE html><html>…</html> with a <style> block."
    )


def _scene_to_html(scene_desc: str, niche_cfg: dict, duration: float) -> str | None:
    key = _load_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _scene_prompt(scene_desc, niche_cfg, duration)}],
            max_tokens=1600,
            temperature=0.7,
        )
        return _extract_html(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[hf] scene HTML generation failed: {e}")
        return None


def _render_html(html: str, work_dir: Path, out_mp4: Path) -> bool:
    """Render one index.html composition to out_mp4 via the HyperFrames CLI."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "index.html").write_text(html, encoding="utf-8")
    timeout = int(os.environ.get("HF_RENDER_TIMEOUT", "180"))
    try:
        r = subprocess.run(
            _launcher() + ["render", "--output", str(out_mp4)],
            cwd=str(work_dir), capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            print(f"[hf] render failed: {r.stderr[-300:].strip()}")
            return False
        return out_mp4.exists() and out_mp4.stat().st_size >= MIN_BYTES
    except Exception as e:
        print(f"[hf] render error: {e}")
        return False


def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0,
                   niche_cfg: dict | None = None) -> list[Path]:
    """Generate up to n motion-graphic mp4 clips (1080×1920). [] if unavailable."""
    if not is_available():
        print("[hf] HyperFrames not available (need Node 22+; `npx hyperframes`). "
              "Falling back.")
        return []

    niche_cfg = niche_cfg or {}
    prompts   = list(queries or ["abstract cinematic background"])
    while len(prompts) < n:
        prompts.append(prompts[len(prompts) % len(prompts)])
    prompts = prompts[:n]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    clips: list[Path] = []

    for i, desc in enumerate(prompts):
        print(f"[hf] {i+1}/{len(prompts)}: {desc[:70]}")
        html = _scene_to_html(desc, niche_cfg, clip_duration)
        if not html:
            print(f"[hf] no HTML for scene {i+1} — skipping")
            continue
        work_dir = TMP_DIR / f"{stamp}_{i}"
        out_mp4  = TMP_DIR / f"{stamp}_{i}.mp4"
        ok = _render_html(html, work_dir, out_mp4)
        shutil.rmtree(work_dir, ignore_errors=True)
        if ok:
            clips.append(out_mp4)
            print(f"[hf] clip {i+1} ready")
        else:
            print(f"[hf] clip {i+1} failed — skipping")

    print(f"[hf] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    qs = sys.argv[1:] or ["glowing market chart rising on a dark grid",
                          "abstract gold particles drifting in the dark"]
    print("available:", is_available())
    for p in generate_clips(qs, n=len(qs), niche_cfg={"accent_color": "#FFC53D"}):
        print("CLIP=", p)
