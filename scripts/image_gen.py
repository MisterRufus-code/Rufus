#!/usr/bin/env python3
"""
image_gen.py — generate a standalone image from a text prompt on this PC's GPU.

Distinct from thumbnail_gen.py, which BRANDS a frame taken out of an already
rendered video. This one makes a brand-new image from a description, for the
case Rufus had no answer for: someone who is NOT at the keyboard wants a
picture. They describe it, the RTX 3090 renders it through the same ComfyUI
stills workflow the video pipeline uses (config/stills_api.json), and the PNG
lands somewhere the dashboard can hand straight to their phone.

Landscape by default (1280x720) because that's YouTube's thumbnail frame — the
video pipeline's own stills are 1080x1920 portrait, the wrong shape here. Pass
--portrait for a vertical image instead.

Usage:
    python scripts/image_gen.py "a cracked hourglass spilling gold coins"
    python scripts/image_gen.py "..." --out my_image.png --seed 42
    python scripts/image_gen.py "..." --portrait

Requires ComfyUI running with an exported stills workflow — the same setup the
`comfy` video source needs (see comfy_client.py). No ComfyUI, no image: this
prints why and exits non-zero rather than silently producing something that
looks unlike the rest of the channel.

Environment:
  COMFY_HOST              http://localhost:8188
  RUFUS_THUMBNAIL_DIR     where PNGs land (default <media>/thumbnails)
  RUFUS_STILLS_DETAIL     photographic direction appended to the prompt
"""

import argparse
import random
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths

# YouTube's thumbnail frame: 1280x720 is the documented minimum-recommended
# size and the 16:9 ratio every surface crops toward.
THUMB_W, THUMB_H = 1280, 720
PORTRAIT_W, PORTRAIT_H = 1080, 1920


def _slugify(text: str, limit: int = 40) -> str:
    """A filename that says what the image IS — a folder of image_1.png tells
    you nothing three weeks later."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:limit].rstrip("_") or "image"


def _apply_image_dims(graph: dict, width: int, height: int) -> None:
    """Force width/height on the latent-sizing nodes of an IMAGE workflow.

    comfy_template.prepare() only substitutes dims on nodes that ALSO carry
    `length` or `duration` — it was written for video graphs. An image
    workflow's EmptyLatentImage has just width/height, so prepare() leaves the
    export's own resolution untouched, which for a stills template means
    1080x1920 portrait: exactly the wrong shape for a thumbnail. This covers
    the image case, and skips anything video-shaped so a mixed graph is safe.
    """
    for node in graph.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "length" in inputs or "duration" in inputs:
            continue          # a video node — not ours to resize
        if "width" in inputs and "height" in inputs:
            inputs["width"], inputs["height"] = width, height


def generate_image(prompt: str, out_path: Path | None = None, *,
                   width: int = THUMB_W, height: int = THUMB_H,
                   seed: int | None = None,
                   add_detail: bool = True) -> Path | None:
    """Render one image → the saved PNG path, or None on any failure.

    Never raises: callers include a Flask route, where an exception is a 500
    page on someone's phone. Failure is None plus a printed reason.
    """
    import comfy_client
    import comfy_template

    if not comfy_client.is_available():
        print(f"[image] ComfyUI not reachable at {comfy_client._host()} — start it first")
        return None

    tpl = comfy_client._stills_template()
    if tpl is None:
        print("[image] no stills workflow exported to config/stills_api.json "
              "— export one from ComfyUI (see comfy_client.py header)")
        return None

    full_prompt = comfy_client._with_detail(prompt) if add_detail else prompt
    seed = random.randint(1, 2**31 - 1) if seed is None else seed

    graph = comfy_template.prepare(tpl, prompt=full_prompt, seed=seed,
                                   save_prefix="rufus_image")
    _apply_image_dims(graph, width, height)

    client_id = uuid.uuid4().hex
    print(f"[image] rendering {width}x{height} (seed {seed}) …")
    started = time.time()

    pid = comfy_client._submit(graph, client_id)
    if not pid:
        print("[image] ComfyUI rejected the workflow")
        return None

    png_bytes = comfy_client._await_image(pid)
    if not png_bytes:
        print("[image] render produced no image")
        return None

    if out_path is None:
        out_dir = paths.thumbnails_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(time.time())}_{_slugify(prompt)}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)

    # The prompt that made it, stored beside it — so a good image can be
    # reproduced or nudged instead of re-guessed from memory.
    try:
        out_path.with_suffix(".txt").write_text(
            f"PROMPT: {prompt}\nSEED: {seed}\nSIZE: {width}x{height}\n"
            f"FULL PROMPT: {full_prompt}\n", encoding="utf-8")
    except OSError:
        pass

    print(f"[image] {out_path}  ({len(png_bytes)//1024}KB, {time.time()-started:.1f}s)")
    return out_path


def recent_images(limit: int = 40) -> list[dict]:
    """Newest-first list of generated images, for the dashboard gallery."""
    d = paths.thumbnails_dir()
    if not d.exists():
        return []
    pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in pngs[:limit]:
        meta = p.with_suffix(".txt")
        prompt = ""
        if meta.exists():
            try:
                first = meta.read_text(encoding="utf-8").splitlines()[0]
                prompt = first.replace("PROMPT:", "").strip()
            except (OSError, IndexError):
                pass
        st = p.stat()
        out.append({"name": p.name, "prompt": prompt,
                    "mtime": st.st_mtime, "kb": st.st_size // 1024})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate an image from a text prompt on the local GPU")
    ap.add_argument("prompt", help="What the image should show")
    ap.add_argument("--out", "-o", help="Output PNG path (default: media_library/thumbnails/)")
    ap.add_argument("--seed", type=int, help="Reuse a seed to reproduce an image")
    ap.add_argument("--portrait", action="store_true",
                    help=f"Render {PORTRAIT_W}x{PORTRAIT_H} vertical instead of {THUMB_W}x{THUMB_H}")
    ap.add_argument("--width", type=int, help="Explicit width (overrides --portrait)")
    ap.add_argument("--height", type=int, help="Explicit height")
    ap.add_argument("--no-detail", action="store_true",
                    help="Skip the photographic detail suffix — use the prompt verbatim")
    args = ap.parse_args()

    width  = args.width  or (PORTRAIT_W if args.portrait else THUMB_W)
    height = args.height or (PORTRAIT_H if args.portrait else THUMB_H)

    path = generate_image(args.prompt, Path(args.out) if args.out else None,
                          width=width, height=height, seed=args.seed,
                          add_detail=not args.no_detail)
    if path is None:
        return 1
    print(f"OUTPUT={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
