#!/usr/bin/env python3
"""
comfy_client.py — Generate Rufus background clips with a local ComfyUI server.

This replaces the Pexels stock-footage source with AI-generated video (Wan2.1 or
any text-to-video workflow) running on a rented GPU. Enable it with:

    export RUFUS_VIDEO_SOURCE=comfy
    export COMFY_URL=http://127.0.0.1:8188      # default
    export RUFUS_GPU=1                          # so render uses NVENC too

How it works
------------
ComfyUI exposes an HTTP API. We:
  1. Load a workflow you exported from ComfyUI via  Save (API Format)  →  a JSON
     file placed at config/comfy_workflows/<name>.json, with placeholder tokens.
  2. Substitute the prompt / seed / size / frame-count tokens for each clip.
  3. POST it to /prompt, poll /history/{id}, then download the produced mp4 via
     /view into media_library/comfy/.

Placeholder tokens the workflow JSON may contain (all optional):
    __PROMPT__   positive prompt text
    __NEG__      negative prompt text
    __SEED__     integer seed
    __WIDTH__    output width   (default 1080 for 9:16)
    __HEIGHT__   output height  (default 1920)
    __FRAMES__   number of frames to generate
    __FILENAME__ output filename prefix

Design note: we deliberately do NOT hardcode node names. Whatever workflow the
ComfyUI template ships with, you export it once with these tokens in the relevant
fields and Rufus parameterizes it. Graceful: any failure returns [] so the caller
can fall back to Pexels.
"""

import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import requests

ROOT          = Path(__file__).parent.parent
CONFIG_DIR    = ROOT / "config"
WORKFLOW_DIR  = CONFIG_DIR / "comfy_workflows"
OUT_DIR       = ROOT / "media_library" / "comfy"

COMFY_URL     = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_NAME = os.environ.get("COMFY_WORKFLOW", "wan_t2v")
# Wan2.1's sweet spot is 720x1280 (9:16); Rufus upscales/crops to 1080x1920 during
# the Ken-Burns pass, so generating smaller is faster with no visible quality loss.
CLIP_W        = int(os.environ.get("COMFY_WIDTH",  "720"))
CLIP_H        = int(os.environ.get("COMFY_HEIGHT", "1280"))
CLIP_FRAMES   = int(os.environ.get("COMFY_FRAMES", "81"))     # ~5s @ 16fps for Wan2.1
POLL_TIMEOUT  = int(os.environ.get("COMFY_TIMEOUT", "900"))   # 15 min per clip ceiling
NEG_PROMPT    = os.environ.get(
    "COMFY_NEG",
    "blurry, low quality, watermark, text, distorted, deformed, static, still image",
)

# Cinematic styling appended to every prompt so generated footage looks pro.
STYLE_SUFFIX = ("cinematic, shallow depth of field, professional color grading, "
                "vertical 9:16, dynamic camera motion, 4k, highly detailed")


def is_available() -> bool:
    """Return True if a ComfyUI server is reachable."""
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def _load_workflow() -> dict:
    path = WORKFLOW_DIR / f"{WORKFLOW_NAME}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"ComfyUI workflow not found: {path}\n"
            f"Export one from ComfyUI (Save → API Format) with placeholder tokens "
            f"(__PROMPT__, __SEED__, …) and save it there."
        )
    return json.loads(path.read_text())


def _inject(workflow: dict, repl: dict) -> dict:
    """Recursively replace placeholder tokens in every string value of the graph.

    If a value is EXACTLY a token (e.g. "__WIDTH__"), substitute the typed value so
    numeric fields stay ints. Otherwise do in-string text replacement (prompts).
    """
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            if node in repl:                 # whole value is a token → keep its type
                return repl[node]
            out = node
            for token, value in repl.items():
                if token in out:
                    out = out.replace(token, str(value))
            return out
        return node
    return walk(workflow)


def _submit(workflow: dict) -> str:
    """Queue a workflow. Returns the prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["prompt_id"]


def _wait_for_outputs(prompt_id: str) -> list[dict]:
    """Poll /history until the prompt finishes. Returns list of output-file dicts."""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            hist = r.json().get(prompt_id)
        except Exception:
            hist = None
        if hist:
            status = hist.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for {prompt_id}")
            files = []
            for node_out in hist.get("outputs", {}).values():
                # Video nodes emit 'gifs'/'videos'; image nodes emit 'images'.
                for key in ("gifs", "videos", "images"):
                    files.extend(node_out.get(key, []))
            if files:
                return files
        time.sleep(2)
    raise TimeoutError(f"ComfyUI clip timed out after {POLL_TIMEOUT}s")


def _download(file_info: dict, dest: Path) -> bool:
    params = urllib.parse.urlencode({
        "filename":  file_info.get("filename", ""),
        "subfolder": file_info.get("subfolder", ""),
        "type":      file_info.get("type", "output"),
    })
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/view?{params}", timeout=120) as resp:
            data = resp.read()
        if len(data) < 10_000:          # too small to be a real clip
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"[comfy] download failed: {e}")
        return False


def generate_clip(prompt: str, index: int = 0) -> Path | None:
    """Generate a single clip for one prompt. Returns the mp4 path or None."""
    full_prompt = f"{prompt.strip()}, {STYLE_SUFFIX}"
    seed = random.randint(1, 2**31 - 1)
    repl = {
        "__PROMPT__":   full_prompt,
        "__NEG__":      NEG_PROMPT,
        "__SEED__":     seed,
        "__WIDTH__":    CLIP_W,
        "__HEIGHT__":   CLIP_H,
        "__FRAMES__":   CLIP_FRAMES,
        "__FILENAME__": f"rufus_{int(time.time())}_{index}",
    }
    workflow = _inject(_load_workflow(), repl)
    print(f"[comfy] generating clip {index + 1}: \"{prompt[:60]}\"")
    pid   = _submit(workflow)
    files = _wait_for_outputs(pid)
    dest  = OUT_DIR / f"clip_{int(time.time())}_{index}.mp4"
    if _download(files[0], dest):
        print(f"[comfy] ✓ {dest.name}")
        return dest
    return None


def generate_clips(prompts: list[str], n: int = 5) -> list[Path]:
    """Generate up to n clips. Cycles prompts if fewer than n were supplied.

    Returns whatever succeeded — never raises — so main.py can fall back to
    Pexels if ComfyUI is down or every clip failed.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not prompts:
        print("[comfy] no prompts supplied — nothing to generate")
        return []
    if not is_available():
        print(f"[comfy] server not reachable at {COMFY_URL} — skipping")
        return []

    clips: list[Path] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        try:
            clip = generate_clip(prompt, index=i)
            if clip:
                clips.append(clip)
        except Exception as e:
            print(f"[comfy] clip {i + 1} failed: {e}")
    print(f"[comfy] generated {len(clips)}/{n} clips")
    return clips


if __name__ == "__main__":
    import sys
    test_prompts = sys.argv[1:] or [
        "an elderly investor studying financial charts at a wooden desk",
        "stacks of gold coins slowly rotating on a dark reflective surface",
    ]
    print(f"COMFY_URL={COMFY_URL}  available={is_available()}")
    out = generate_clips(test_prompts, n=len(test_prompts))
    print("GENERATED:", [str(p) for p in out])
