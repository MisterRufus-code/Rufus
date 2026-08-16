#!/usr/bin/env python3
"""
hyperframes_client.py — motion-graphic video source for Rufus via HeyGen's
HyperFrames (open-source HTML→MP4 renderer, Apache-2.0).

Two rendering modes:

  CSS-only  — GPT writes a self-contained animated HTML scene (charts, particles,
              gradients, counters).  No external assets needed.  Best when SD is
              not running.

  Hybrid    — SD supplies a photorealistic base image; GPT writes HTML that embeds
              it (via base64 data URI) with a CSS Ken Burns animation + a data
              overlay (chart / counter / particle) on top.  Best quality.

Both modes implement the same contract as sd_client:

    generate_clips(queries, n, clip_duration, niche_cfg, image_paths) -> list[Path]

so audio_gen.render() composites voice + captions + music on top unchanged.

Env:
  HYPERFRAMES_CMD   space-separated launcher (default: "npx --yes hyperframes")
  HF_RENDER_TIMEOUT seconds per scene render (default 180)
"""

import base64
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

# The scene size handed to the HTML renderer, from the active format profile.
# It is written into the prompt AND into the fallback scene's own CSS, so a
# hard-coded pair here does not merely mis-size the output — it tells the model
# in words to build a 1080×1920 page for a 1920×1080 video.
import video_format as _vf
OUT_W, OUT_H = _vf.dimensions()
MODEL        = "gpt-4o-mini"
MIN_BYTES    = 50_000


def _launcher() -> list[str]:
    """The launcher argv, with the executable resolved to an absolute path.

    Same Windows trap as remotion_renderer._npx: npx is `npx.cmd` there, and
    CreateProcess does not apply PATHEXT, so a bare "npx" raises WinError 2 and
    `is_available()` reports False. HyperFrames then looks uninstalled on a box
    where it is installed and working.
    """
    argv = os.environ.get("HYPERFRAMES_CMD", "npx --yes hyperframes").split()
    if argv:
        resolved = shutil.which(argv[0])
        if resolved:
            argv[0] = resolved
    return argv


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
        key = json.loads(KEYS_FILE.read_text(encoding="utf-8")).get("openai", "")
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


# ── Visual-style guidance (niche-aware) ────────────────────────────────────────

def _niche_visual_guide(niche_cfg: dict) -> str:
    """Return niche-specific visual animation instructions for the GPT scene prompt."""
    tags = (niche_cfg.get("name", "") + " " + niche_cfg.get("style_suffix", "")).lower()

    if any(k in tags for k in ("finance", "business", "mindset", "stock", "wealth", "data")):
        return (
            "VISUAL STYLE — data/finance. Build ONE of:\n"
            "  (A) SVG line chart: a <path> whose stroke draws itself left→right via "
            "stroke-dashoffset animation over the full duration. Include light grid lines "
            "and axis tick marks. Glowing accent-color data line.\n"
            "  (B) Counter ticking up: large numeric display animating 0 → a realistic "
            "target (e.g. $847K or 94%) using CSS @property or step() timing.\n"
            "  (C) Bar chart: 5–7 vertical bars growing upward from 0, staggered delays, "
            "accent color fills.\n"
            "Dark near-black background, subtle grid. The chart/counter occupies the "
            "lower 40% of the stage so it reads clearly behind spoken captions."
        )
    if any(k in tags for k in ("motivation", "personal", "development", "energy", "sunrise")):
        return (
            "VISUAL STYLE — motivation/energy. ONE of:\n"
            "  (A) Rising particles: 25–40 small circles drifting upward from the bottom "
            "with gentle horizontal sway, accent-color gradient opacity.\n"
            "  (B) Light burst: 8–12 rays expanding from dead-centre with slow rotation, "
            "high-opacity core fading to transparent at the edges.\n"
            "  (C) Pulse rings: concentric circles expanding from centre, staggered delays, "
            "accent color, low opacity — heartbeat feel.\n"
            "Energetic but smooth — never strobing or jarring."
        )
    return (
        "VISUAL STYLE — bold geometric. Slowly panning gradient across the frame, OR "
        "drifting translucent polygons/circles with subtle rotation. Dramatic, cinematic."
    )


def _overlay_hint(niche_cfg: dict, accent: str) -> str:
    """Short overlay description for hybrid mode (lower-third, above the SD photo)."""
    tags = (niche_cfg.get("name", "") + " " + niche_cfg.get("style_suffix", "")).lower()
    if any(k in tags for k in ("finance", "business", "mindset", "stock", "wealth", "data")):
        return (
            f"an SVG line chart whose path draws itself over the clip duration. "
            f"Accent color {accent}. Dark semi-transparent panel (rgba 0,0,0,0.55). "
            "Subtle grid. Positioned at the bottom 35% of the frame."
        )
    if any(k in tags for k in ("motivation", "personal", "development", "energy", "sunrise")):
        return (
            f"25–35 small rising particles (accent color {accent}, opacity 0.5–0.8) "
            "drifting upward from the bottom of the frame."
        )
    return f"a soft radial glow pulse from bottom-centre, accent color {accent}, opacity 0.3."


# ── Prompt builders ────────────────────────────────────────────────────────────

def _scene_prompt(scene_desc: str, niche_cfg: dict, duration: float) -> str:
    """CSS-only mode: GPT writes a full animated HTML scene from scratch."""
    accent = niche_cfg.get("accent_color", "#FFD23F")
    mood   = niche_cfg.get("style_suffix", "cinematic, high contrast")
    guide  = _niche_visual_guide(niche_cfg)
    return (
        "Write ONE self-contained HTML document for an animated motion-graphic BACKGROUND "
        "(vertical short-form video). Output ONLY the HTML — no explanation.\n\n"
        f"Scene concept: {scene_desc}\n"
        f"Palette: {mood}. Accent color: {accent}. Dark near-black base (#0a0a0f).\n\n"
        f"{guide}\n\n"
        "HARD RULES:\n"
        f'- Root element must have: id="stage" data-composition-id="scene" data-start="0" '
        f'data-duration="{duration:.1f}" data-width="{OUT_W}" data-height="{OUT_H}" '
        f'and CSS: width:{OUT_W}px; height:{OUT_H}px; overflow:hidden; position:relative.\n'
        "- ONLY inline CSS @keyframes / SVG / transforms / gradients. NO external scripts, "
        "NO CDN links, NO <img src> with http/https URLs — must render offline and "
        "deterministically.\n"
        "- NO large text, NO words, NO headlines, NO logos. Spoken-word captions are "
        "overlaid by a later stage; on-screen text here would collide with them.\n"
        f"- Animations fill the full {duration:.1f}s and hold gracefully at the final frame.\n"
        "- One file: <!DOCTYPE html><html><head><style>…</style></head><body>…</body></html>"
    )


def _scene_prompt_hybrid(scene_desc: str, niche_cfg: dict, duration: float) -> str:
    """Hybrid mode: GPT writes HTML with __IMAGE_DATA_URI__ placeholder + CSS overlay."""
    accent  = niche_cfg.get("accent_color", "#FFD23F")
    overlay = _overlay_hint(niche_cfg, accent)
    return (
        "Write ONE self-contained HTML document for a vertical short-form video background. "
        "A photorealistic background image will be injected AFTER generation — write the "
        "literal string __IMAGE_DATA_URI__ as the src of <img id='bg'> and do NOT change it.\n\n"
        f"Scene concept: {scene_desc}\n"
        f"Accent color: {accent}\n\n"
        "EXACT REQUIRED STRUCTURE:\n"
        f'<div id="stage" data-composition-id="scene" data-start="0" '
        f'data-duration="{duration:.1f}" data-width="{OUT_W}" data-height="{OUT_H}" '
        f'style="width:{OUT_W}px;height:{OUT_H}px;overflow:hidden;position:relative;">\n'
        f'  <img id="bg" src="__IMAGE_DATA_URI__" style="position:absolute;top:0;left:0;'
        f'width:100%;height:100%;object-fit:cover;'
        f'animation:kenburns {duration:.1f}s ease-in-out forwards;">\n'
        "  <!-- CSS animated overlay elements here -->\n"
        "</div>\n\n"
        "@keyframes kenburns must be: 0%{transform:scale(1.1) translate(1%,1%)} "
        f"100%{{transform:scale(1.0) translate(-1%,-1%)}}\n\n"
        f"OVERLAY: {overlay}\n"
        "Use ONLY inline CSS @keyframes. NO external URLs, NO CDN scripts, "
        f"NO words or logos. All animations fill {duration:.1f}s."
    )


# ── GPT call ───────────────────────────────────────────────────────────────────

def _call_gpt(prompt: str, max_tokens: int = 2000) -> str | None:
    """Call GPT once, return extracted HTML or None on failure."""
    key = _load_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return _extract_html(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[hf] HTML generation failed: {e}")
        return None


def _scene_to_html(scene_desc: str, niche_cfg: dict, duration: float) -> str | None:
    return _call_gpt(_scene_prompt(scene_desc, niche_cfg, duration))


def _scene_to_html_hybrid(scene_desc: str, niche_cfg: dict, duration: float,
                           image_b64: str) -> str | None:
    """Generate hybrid HTML and inject the actual base64 image after GPT responds."""
    html = _call_gpt(_scene_prompt_hybrid(scene_desc, niche_cfg, duration), max_tokens=2400)
    if not html:
        return None
    return html.replace("__IMAGE_DATA_URI__", f"data:image/png;base64,{image_b64}")


# ── Renderer ───────────────────────────────────────────────────────────────────

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


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0,
                   niche_cfg: dict | None = None,
                   image_paths: list[Path] | None = None) -> list[Path]:
    """Generate up to n motion-graphic mp4 clips (1080×1920). Returns [] if unavailable.

    image_paths — optional SD-generated PNG paths for hybrid mode.  When an image
    is provided for scene i, GPT embeds it as a base64 background with CSS Ken Burns
    + an animated data overlay on top.  Scenes without a matching image fall back to
    pure CSS animation automatically.
    """
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

        img_path = (image_paths[i] if image_paths and i < len(image_paths) else None)
        if img_path and img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            html    = _scene_to_html_hybrid(desc, niche_cfg, clip_duration, img_b64)
            mode    = "hybrid"
        else:
            html = _scene_to_html(desc, niche_cfg, clip_duration)
            mode = "css-only"

        if not html:
            print(f"[hf] no HTML for scene {i+1} ({mode}) — skipping")
            continue

        print(f"[hf] rendering scene {i+1} ({mode})...")
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
