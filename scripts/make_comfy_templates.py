#!/usr/bin/env python3
"""
make_comfy_templates.py — derive img2img ComfyUI templates from the stills one.

WHY THIS IS NOT THE THING comfy_template.py FORBIDS. That module's docstring
bans building a graph *from documentation* — the Wan 2.2 integration proved
that guessing node wiring and sampler settings gets them subtly wrong, and only
an export of a proven run on the real rig gets them right. This script does not
guess anything: it reads the channel owner's ALREADY-PROVEN
config/stills_api.json and rewrites the four things img2img actually requires,
leaving every verified setting — loaders, weight dtype, sampler, scheduler,
cfg, shift, VAE — byte-identical to the working export.

The four changes:
  1. add a LoadImage node (the init frame)
  2. add a VAEEncode, reusing THE SAME vae reference VAEDecode already uses
     (not a guessed loader id)
  3. point the sampler's latent_image at that VAEEncode instead of the empty latent
  4. set denoise < 1.0, and raise steps so denoise*steps stays workable

Point 4 matters on a distilled model: Z-Image-Turbo's 9 steps at denoise 0.4
leaves ~4 effective steps. Steps are raised so the picture can actually move.

Outputs:
  config/stills_i2i_api.json      denoise 0.40 — frame-to-frame continuity
                                  (RUFUS_BEAT_MOTION=i2i)
  config/character_stills_api.json denoise 0.55 — carry a reference character
                                  into a new scene. NOTE: this is plain img2img
                                  conditioning, NOT an identity lock. IPAdapter
                                  FaceID/PuLID are stronger but are SD1.5/SDXL/
                                  FLUX-targeted and rely on InsightFace, which
                                  detects photographic faces — a poor fit for a
                                  flat-2D-illustration niche on Z-Image. A
                                  character LoRA remains the strong option.

Usage:
  python scripts/make_comfy_templates.py
  python scripts/make_comfy_templates.py --i2i-denoise 0.45 --steps 14

Verify in ComfyUI before trusting a run: load the produced JSON, drop in any
image, queue it, and check the result actually changes but stays recognisable.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "config"

LOAD_ID = "rufus_init_image"
ENCODE_ID = "rufus_init_encode"


def _find(graph: dict, predicate) -> str | None:
    for nid, node in graph.items():
        if isinstance(node, dict) and predicate(node):
            return nid
    return None


def _sampler_id(graph: dict) -> str | None:
    """The node doing the denoising. Matched by INPUTS, not class name, so a
    custom or renamed sampler still resolves."""
    return _find(graph, lambda n: "denoise" in (n.get("inputs") or {})
                 and "latent_image" in (n.get("inputs") or {}))


def _vae_ref(graph: dict):
    """The exact vae wire VAEDecode uses. Reusing the reference rather than
    hunting for a VAELoader keeps this correct when the VAE comes bundled out
    of a checkpoint loader instead."""
    dec = _find(graph, lambda n: n.get("class_type") == "VAEDecode")
    if dec is None:
        return None
    return (graph[dec].get("inputs") or {}).get("vae")


def derive_i2i(graph: dict, *, denoise: float, steps: int | None,
               save_prefix: str) -> dict:
    """Return an img2img graph derived from a proven txt2img graph."""
    g = copy.deepcopy(graph)

    ks = _sampler_id(g)
    if ks is None:
        raise ValueError("no sampler with denoise+latent_image found — is "
                         "config/stills_api.json a real API export?")
    vae = _vae_ref(g)
    if vae is None:
        raise ValueError("no VAEDecode found, so the VAE wire is unknown")

    old_latent = (g[ks].get("inputs") or {}).get("latent_image")

    g[LOAD_ID] = {"class_type": "LoadImage",
                  "inputs": {"image": "rufus_init.png", "upload": "image"}}
    g[ENCODE_ID] = {"class_type": "VAEEncode",
                    "inputs": {"pixels": [LOAD_ID, 0], "vae": vae}}

    g[ks]["inputs"]["latent_image"] = [ENCODE_ID, 0]
    g[ks]["inputs"]["denoise"] = denoise
    if steps is not None:
        g[ks]["inputs"]["steps"] = steps

    # Drop the empty-latent node now that nothing feeds from it. ComfyUI prunes
    # unreached nodes anyway, but leaving a dangling size source invites
    # someone to "fix" the resolution there later and wonder why it does
    # nothing — the size now comes from the init image.
    if isinstance(old_latent, list) and old_latent:
        orphan = str(old_latent[0])
        still_used = any(
            isinstance(v, list) and v and str(v[0]) == orphan
            for nid, n in g.items() if nid != orphan
            for v in (n.get("inputs") or {}).values())
        if not still_used:
            g.pop(orphan, None)

    for node in g.values():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = save_prefix
    return g


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(CONFIG / "stills_api.json"))
    ap.add_argument("--i2i-denoise", type=float, default=0.40)
    ap.add_argument("--character-denoise", type=float, default=0.55)
    ap.add_argument("--steps", type=int, default=14,
                    help="raised from the txt2img value so denoise*steps stays "
                         "workable on a distilled model (default 14)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing template (they may be hand-tuned)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"[templates] {src} not found — export your stills workflow from "
              f"ComfyUI first (see README's 'Swappable stills model').")
        return 1
    try:
        graph = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[templates] {src} is not valid JSON: {e}")
        return 1
    if isinstance(graph.get("prompt"), dict):
        graph = graph["prompt"]

    if "RUFUS_PROMPT" not in json.dumps(graph):
        print(f"[templates] {src} has no RUFUS_PROMPT placeholder — set the "
              f"positive prompt to that literal string and re-export.")
        return 1

    targets = [
        (CONFIG / "stills_i2i_api.json", args.i2i_denoise, "rufus_i2i"),
        (CONFIG / "character_stills_api.json", args.character_denoise, "rufus_character"),
    ]
    written = 0
    for path, denoise, prefix in targets:
        if path.exists() and not args.force:
            print(f"[templates] {path.name} already exists — leaving it alone "
                  f"(--force to overwrite)")
            continue
        try:
            out = derive_i2i(graph, denoise=denoise, steps=args.steps,
                             save_prefix=prefix)
        except ValueError as e:
            print(f"[templates] cannot derive {path.name}: {e}")
            return 1
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[templates] wrote {path.name}  (denoise {denoise}, steps {args.steps})")
        written += 1

    if written:
        print("\nVERIFY BEFORE A REAL RUN: open each file in ComfyUI "
              "(Workflow → Open), drop any image into the LoadImage node, and "
              "queue it once. The output should change but stay recognisable. "
              "If it barely moves, raise denoise; if it loses the scene, lower it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
