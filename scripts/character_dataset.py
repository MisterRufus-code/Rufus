#!/usr/bin/env python3
"""
character_dataset.py — generate a LoRA training set for a niche's recurring
character, stills only.

Why this exists: text-level character consistency (character_engine.py) can
hold ONE fixed look, because the repeated wardrobe description IS the thing
tying the frames together. The moment you want the same character in DIFFERENT
clothing per scene, text alone cannot do it — nothing would be left connecting
the images. That needs a model that encodes the character's identity
independently of what they wear, and for a flat-2D-illustration niche running
Z-Image-Turbo the practical route is a character LoRA:

  - IPAdapter FaceID Plus V2 is SD1.5/SDXL only and PuLID targets FLUX, so
    neither is a mature fit for the Z-Image stills model here.
  - Both rely on InsightFace, which detects PHOTOGRAPHIC faces. money_history
    renders flat vector illustration with simplified facial shapes, which that
    detector is not built for.
  - A LoRA trains on the character itself, works with stylized art, and leaves
    the wardrobe free to be prompted per scene.

A LoRA needs 20-40 images of the character. Producing those by running full
video pipelines is the slow way — a run spends most of its wall clock on the
motion pass (Hunyuan/Wan/LTX, ~10 min/video) and on scripting/rendering, none
of which a training set uses. This script skips all of it: stills only, one
ComfyUI call per image, no script, no motion, no render.

Usage:
  python scripts/character_dataset.py --niche money_history --count 24
  python scripts/character_dataset.py --niche money_history --count 8 --start-index 24

Output (kohya_ss / sd-scripts layout — image + caption sharing a basename):
  media_library/character_datasets/<niche>/
      chronicler_000.png
      chronicler_000.txt      <- CAPTION, not metadata
      ...
      _manifest.jsonl         <- prompt/seed per image, for reproducing one

Then: curate by hand (delete anything off-model), train the LoRA, add a
LoraLoaderModelOnly to the ComfyUI stills workflow and re-export it to
config/stills_api.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths

# LoRA convention: square, and the same size for every image in the set.
DEFAULT_SIZE = 1024
DEFAULT_COUNT = 24

# A LoRA learns whatever stays CONSTANT across the set and treats what varies
# as incidental. So the character description is held fixed while framing,
# angle, action and background are deliberately rotated — otherwise the LoRA
# bakes in one pose (or one background) as part of the character's identity.
_FRAMINGS = [
    "full body, head to feet",
    "three-quarter length, head to knees",
    "waist-up",
    "head and shoulders",
]
_ANGLES = [
    "facing the viewer straight on",
    "turned three-quarters to the left",
    "turned three-quarters to the right",
    "in profile, facing left",
    "in profile, facing right",
]
_ACTIONS = [
    "standing still, arms at their sides",
    "mid-stride, walking forward",
    "one hand raised, gesturing outward",
    "holding the lantern up at shoulder height",
    "seated on a low stone step",
    "turning to look back over one shoulder",
    "both hands clasped in front",
    "leaning slightly forward, listening",
]
# Kept plain and low-detail on purpose: a busy, distinctive background gets
# learned as part of the character. Simple fields and flat shapes do not.
_BACKGROUNDS = [
    "a plain flat cream background",
    "a plain flat sepia background",
    "a bare wall with generous negative space",
]

# List LENGTHS are deliberately 4/5/8/3 — pairwise-varied rather than all the
# same. When every list was length 4 they advanced in lockstep, so a 24-image
# set collapsed to 8 distinct setups repeated three times. With these lengths
# the combination repeats only every lcm(4,5,8,3) = 120 images, so any
# realistic set size gets a genuinely different setup per image.
_VARIATION_PERIOD = 120


def _variation(i: int) -> tuple[str, str]:
    """(prompt fragment, caption fragment) for image `i`. Deterministic."""
    framing = _FRAMINGS[i % len(_FRAMINGS)]
    angle = _ANGLES[i % len(_ANGLES)]
    action = _ACTIONS[i % len(_ACTIONS)]
    background = _BACKGROUNDS[i % len(_BACKGROUNDS)]
    text = f"{framing}, {angle}, {action}, {background}"
    return text, text


def build_prompt(niche: str, index: int) -> tuple[str, str] | None:
    """(render prompt, caption) for image `index`, or None if the niche has no
    character configured.

    The RENDER prompt carries the full character description — this is the one
    place detail is wanted, unlike the per-beat prompts, which must stay compact
    (see character_engine.short_ref).

    The CAPTION deliberately does NOT describe the character. In LoRA training,
    whatever you caption is treated as variable and whatever you leave uncaptioned
    is absorbed into the trigger word. Captioning the cloak would teach the model
    that "cloak" is an interchangeable detail — the exact opposite of what a
    character LoRA is for."""
    import character_engine

    cfg = character_engine.niche_character(niche)
    if not cfg:
        return None

    try:
        niches = json.loads(character_engine.NICHES_FILE.read_text(encoding="utf-8"))
        style = niches["niches"].get(niche, {}).get("style_suffix", "")
    except Exception:
        style = ""

    trigger = cfg.get("lora_trigger") or _default_trigger(cfg)
    variation, caption_variation = _variation(index)
    name = cfg.get("name") or "the character"

    prompt = (f"{name}: {cfg['description']} {variation}. "
              f"Consistent character design in every image.")
    if style:
        prompt = f"{prompt} {style}"

    return prompt, f"{trigger}, {caption_variation}"


def _default_trigger(cfg: dict) -> str:
    """A LoRA trigger token from the character's name. Lowercased, underscored,
    and suffixed so it is unlikely to collide with a real word the base model
    already knows — a trigger that IS an ordinary word gets diluted by
    everything the model already associates with it."""
    name = (cfg.get("name") or "character").lower()
    slug = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    slug = "_".join(p for p in slug.split("_") if p and p != "the")
    return f"{slug or 'character'}_v1"


def build_dataset(niche: str, count: int = DEFAULT_COUNT,
                  out_dir: Path | None = None, size: int = DEFAULT_SIZE,
                  start_index: int = 0, seed_base: int | None = None) -> list[Path]:
    """Render `count` training images. Returns the paths actually written.

    Never raises on a per-image failure — a ComfyUI hiccup 18 images into a 24
    image set must not throw away the 17 that worked."""
    import character_engine
    import image_gen

    if not character_engine.niche_character(niche):
        print(f"[dataset] niche '{niche}' has no character block in niches.json")
        return []

    out_dir = Path(out_dir) if out_dir else paths.character_dataset_dir() / niche
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = character_engine.niche_character(niche)
    trigger = cfg.get("lora_trigger") or _default_trigger(cfg)
    stem = trigger.rsplit("_v", 1)[0] or "character"
    if seed_base is None:
        seed_base = random.randint(1, 2_000_000_000)

    print(f"[dataset] {niche} → {out_dir}")
    print(f"[dataset] trigger word: {trigger}   (use this in prompts after training)")
    print(f"[dataset] {count} images at {size}x{size}, seed base {seed_base}")

    manifest = out_dir / "_manifest.jsonl"
    written: list[Path] = []

    for n in range(count):
        index = start_index + n
        built = build_prompt(niche, index)
        if built is None:
            break
        prompt, caption = built
        out_path = out_dir / f"{stem}_{index:03d}.png"
        # An EXPLICIT seed is load-bearing, not just for reproducibility:
        # image_gen skips its perceptual-dedup pass when one is given. Without
        # that, a training set — which is by definition many near-identical
        # images of one character — would be fought by the very check meant to
        # stop repeated video frames, and would also poison the shared
        # freshness pool that the video pipeline reads.
        seed = (seed_base + index * 7919) % (2**31 - 1)
        print(f"[dataset] {n + 1}/{count}  {out_path.name}")
        got = image_gen.generate_image(prompt, out_path, width=size, height=size,
                                       seed=seed)
        if got is None:
            print(f"[dataset] image {index} failed — continuing")
            continue

        # image_gen writes a "PROMPT:/SEED:" sidecar at this exact path, which a
        # LoRA trainer would read as the caption. Overwrite it with the real
        # caption and keep the generation metadata in the manifest instead.
        try:
            got.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        except OSError as e:
            print(f"[dataset] caption write failed for {got.name}: {e}")
        try:
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"file": got.name, "index": index,
                                    "seed": seed, "caption": caption,
                                    "prompt": prompt}) + "\n")
        except OSError:
            pass
        written.append(got)

    print(f"\n[dataset] {len(written)}/{count} images written to {out_dir}")
    if written:
        print(f"[dataset] NEXT: delete any image that is off-model, then train a "
              f"LoRA on this folder with trigger '{trigger}'.")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--niche", default="money_history")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--start-index", type=int, default=0,
                    help="continue an existing set instead of overwriting it")
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    written = build_dataset(args.niche, count=args.count, size=args.size,
                            out_dir=args.out_dir, start_index=args.start_index,
                            seed_base=args.seed_base)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
