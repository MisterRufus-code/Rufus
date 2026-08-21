#!/usr/bin/env python3
"""
image_gen.py — generate a standalone image from a text prompt on this PC's GPU.

Distinct from thumbnail_gen.py, which BRANDS a frame taken out of an already
rendered video. This one makes a brand-new image from a description, for the
case Rufus had no answer for: someone who is NOT at the keyboard wants a
picture. They describe it, the RTX 3090 renders it through the same ComfyUI
stills workflow the video pipeline uses (config/stills_api.json), and the PNG
lands somewhere the dashboard can hand straight to their phone.

Landscape by default (1280x720) because that's YouTube's thumbnail frame, which
is not necessarily the video's — a Short renders 1080x1920, the wrong shape for
a thumbnail. Pass --frame for an image at the active format's frame size.

Deduplicated against recent output (video frames AND other thumbnails, one
shared history — see generate_image()'s docstring): a near-duplicate gets one
automatic regeneration with a new seed before being accepted.

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
# The other offer: the shape of the video itself, so an image made here can be
# dropped into a run. Called PORTRAIT_W/H and fixed at 1080×1920 while there
# was only one format — on a long-form channel that option produced a vertical
# image labelled "matches the video frame" that matched nothing.
import video_format as _vf
FRAME_W, FRAME_H = _vf.dimensions()

# Cap for a browser-initiated render. The dashboard runs threaded=False, so a
# request that blocks holds up EVERY other user — including the owner trying
# to approve a video. A still is ~20-30s on a 3090 when the GPU is free; past
# this something else is occupying it, and failing fast with "try again" beats
# freezing the page for five minutes.
WEB_TIMEOUT = 90


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
                   add_detail: bool = True,
                   timeout: float | None = None) -> Path | None:
    """Render one image → the saved PNG path, or None on any failure.

    Never raises: callers include a Flask route, where an exception is a 500
    page on someone's phone. Failure is None plus a printed reason.

    `timeout` caps the wait for ComfyUI. The dashboard passes a short one
    because it renders inline on a single-threaded server — see WEB_TIMEOUT.

    DEDUPLICATION: this module previously had none at all, unlike the video
    pipeline (comfy_client.py), which perceptual-hashes every still and
    regenerates on a near-duplicate. The reported symptom was exactly what
    that gap predicts — asking for a thumbnail repeatedly landed on the same
    or a near-identical image (a coin close-up, session after session) with
    nothing to notice or push back on it. This now reuses the SAME hash
    check and the SAME persisted history file the video pipeline reads and
    writes (comfy_client._load_prior_hashes/_save_hashes), so a thumbnail
    that looks like a recent VIDEO frame is caught too, not just other
    thumbnails — one shared "recently generated" pool, not two blind ones.
    Skipped when the caller passed an explicit `seed`: that means they're
    deliberately trying to reproduce a specific past image, and silently
    swapping the seed on them would defeat the point.
    """
    import comfy_client
    import comfy_template
    from sd_client import DUP_THRESHOLD, MAX_DUP_RETRIES, _avg_hash, _hamming

    if not comfy_client.is_available():
        print(f"[image] ComfyUI not reachable at {comfy_client._host()} — start it first")
        return None

    tpl = comfy_client._stills_template()
    if tpl is None:
        print("[image] no stills workflow exported to config/stills_api.json "
              "— export one from ComfyUI (see comfy_client.py header)")
        return None

    full_prompt = comfy_client._with_detail(prompt) if add_detail else prompt
    explicit_seed = seed is not None
    seed = random.randint(1, 2**31 - 1) if seed is None else seed

    check_freshness = (not explicit_seed) and comfy_client._fresh_images_enabled()
    accepted_hashes = comfy_client._load_prior_hashes() if check_freshness else []

    client_id = uuid.uuid4().hex
    started = time.time()
    png_bytes: bytes | None = None
    img_hash: int | None = None
    max_tries = (MAX_DUP_RETRIES + 1) if check_freshness else 1

    for attempt in range(max_tries):
        this_seed = seed if attempt == 0 else random.randint(1, 2**31 - 1)
        graph = comfy_template.prepare(tpl, prompt=full_prompt, seed=this_seed,
                                       save_prefix="rufus_image",
                                       negative=comfy_client._stills_negative())
        _apply_image_dims(graph, width, height)

        print(f"[image] rendering {width}x{height} (seed {this_seed}) …")
        pid = comfy_client._submit(graph, client_id)
        if not pid:
            print("[image] ComfyUI rejected the workflow")
            return None

        candidate = comfy_client._await_image(pid, timeout=timeout)
        if not candidate:
            print("[image] render produced no image")
            return None

        if not check_freshness:
            png_bytes, seed = candidate, this_seed
            break

        # _avg_hash reads from a path, not raw bytes — a temp file round-trip
        # is the cheapest way to reuse the exact hash function comfy_client
        # already uses, rather than a second, possibly-drifting implementation.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(candidate)
            tmp_path = Path(tf.name)
        try:
            h = _avg_hash(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        is_dup = (h is not None and accepted_hashes
                 and min(_hamming(h, p) for p in accepted_hashes) < DUP_THRESHOLD)
        if is_dup and attempt < max_tries - 1:
            print(f"[image] near-duplicate of a recent image → regenerating "
                  f"(retry {attempt + 1}/{max_tries - 1})")
            continue

        png_bytes, seed, img_hash = candidate, this_seed, h
        if is_dup:
            print(f"[image] still near-dup after {max_tries - 1} retries — keeping it anyway")
        break

    if not png_bytes:
        print("[image] render produced no image")
        return None

    if check_freshness and img_hash is not None:
        accepted_hashes.append(img_hash)
        comfy_client._save_hashes(accepted_hashes)

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


# The composed thumbnail sits beside its background rather than replacing it.
# Overwriting the png would mean the headline could be typed exactly once —
# and the whole point of typing it on the page is trying five of them against
# the same picture without paying for the GPU again.
COMPOSED_SUFFIX = ".thumb.jpg"


def composed_path(png_path: Path) -> Path:
    return Path(png_path).with_suffix(COMPOSED_SUFFIX)


def _sidecar_set(png_path: Path, key: str, value: str) -> None:
    """Replace one KEY: line in the sidecar, keeping the rest."""
    meta = Path(png_path).with_suffix(".txt")
    try:
        lines = meta.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    lines = [ln for ln in lines if not ln.startswith(f"{key}:")]
    lines.append(f"{key}: {value}")
    try:
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def set_headline(png_path: Path, headline: str) -> Path | None:
    """Compose `headline` onto this background and remember it.

    ONE IMPLEMENTATION, TWO CALLERS: the CLI (so a background job can render
    and compose in one process) and the dashboard's recompose route (so
    retyping the words costs ~100ms of Pillow and no GPU at all). A second copy
    of "compose, then record what was composed" would drift, and the half that
    drifts is whichever one is not being looked at.
    """
    png_path = Path(png_path)
    if not png_path.exists():
        return None
    try:
        import thumbnail_gen
        out = thumbnail_gen.compose(png_path, headline, composed_path(png_path))
    except Exception as e:
        print(f"[image] could not compose the headline: {e}")
        return None
    _sidecar_set(png_path, "HEADLINE", headline.replace("\n", " "))
    return out


def recent_images(limit: int = 40) -> list[dict]:
    """Newest-first list of generated images, for the dashboard gallery."""
    d = paths.thumbnails_dir()
    if not d.exists():
        return []
    pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in pngs[:limit]:
        meta = p.with_suffix(".txt")
        prompt, headline = "", ""
        if meta.exists():
            try:
                for line in meta.read_text(encoding="utf-8").splitlines():
                    if line.startswith("PROMPT:") and not prompt:
                        prompt = line.split(":", 1)[1].strip()
                    elif line.startswith("HEADLINE:"):
                        headline = line.split(":", 1)[1].strip()
            except OSError:
                pass
        st = p.stat()
        composed = composed_path(p)
        out.append({"name": p.name, "prompt": prompt, "headline": headline,
                    "composed": composed.name if composed.exists() else "",
                    "mtime": st.st_mtime, "kb": st.st_size // 1024})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate an image from a text prompt on the local GPU")
    ap.add_argument("prompt", help="What the image should show")
    ap.add_argument("--out", "-o", help="Output PNG path (default: media_library/thumbnails/)")
    ap.add_argument("--seed", type=int, help="Reuse a seed to reproduce an image")
    ap.add_argument("--portrait", "--frame", dest="frame", action="store_true",
                    help=f"Render at the video frame {FRAME_W}x{FRAME_H} "
                         f"instead of {THUMB_W}x{THUMB_H}")
    ap.add_argument("--width", type=int, help="Explicit width (overrides --frame)")
    ap.add_argument("--height", type=int, help="Explicit height")
    ap.add_argument("--no-detail", action="store_true",
                    help="Skip the photographic detail suffix — use the prompt verbatim")
    ap.add_argument("--headline", default="",
                    help="Compose these words onto the image as a YouTube "
                         "thumbnail headline (thumbnail_gen.compose)")
    ap.add_argument("--count", type=int, default=1,
                    help="Render this many variants of the same prompt. Real "
                         "thumbnail work is picking one of several, not "
                         "accepting the first.")
    args = ap.parse_args()

    width  = args.width  or (FRAME_W if args.frame else THUMB_W)
    height = args.height or (FRAME_H if args.frame else THUMB_H)

    made = 0
    for n in range(max(1, args.count)):
        # --out names ONE file, so it only makes sense for a single render;
        # asking for five into one path would leave one image and four
        # overwritten ones, which is worse than refusing.
        out = Path(args.out) if (args.out and args.count <= 1) else None
        path = generate_image(args.prompt, out, width=width, height=height,
                              seed=args.seed, add_detail=not args.no_detail)
        if path is None:
            continue
        made += 1
        if args.headline.strip():
            composed = set_headline(path, args.headline.strip())
            if composed:
                print(f"COMPOSED={composed}")
        print(f"OUTPUT={path}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
