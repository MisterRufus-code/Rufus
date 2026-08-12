#!/usr/bin/env python3
"""fetch_wan_lora.py — put the Wan 4-step speed LoRAs where ComfyUI reads them.

WHY A SCRIPT AND NOT A LINK. The 4-step lightx2v/Lightning distillation LoRAs
are the single biggest speed lever for Wan 2.2 on a 24GB card — measured on
this rig (wan_client.py), sampling runs ~57s per step, so 20 steps is ~19
minutes a clip and 4 steps is ~4. But there are dozens of near-identically
named variants published across several repos, they come as a HIGH-NOISE and
LOW-NOISE PAIR that must match, and dropping them anywhere other than
ComfyUI's own models/loras leaves them invisible — which is the exact failure
comfy_doctor exists to diagnose.

IT SEARCHES, IT DOES NOT HARDCODE. Repo layouts for these files have moved
more than once. Pinning a path here would rot, and a rotted path 404s in a way
that reads like "the file is gone" rather than "this script is out of date".
So it lists what each candidate repo actually contains and matches on the
filename, then tells you exactly what it found when it cannot decide.

    python scripts/fetch_wan_lora.py --dest "C:/ComfyUI/models/loras"
    python scripts/fetch_wan_lora.py --dest ... --i2v      # image-to-video pair
    python scripts/fetch_wan_lora.py --dest ... --dry-run  # look, don't fetch

After it finishes, the LoRA still has to be TURNED ON in ComfyUI and the
workflow re-exported: comfy_template.prepare() substitutes prompt, image, seed
and dims only, so the step count and the LoRA toggle are frozen into whatever
was exported. Downloading the file changes nothing on its own.
"""

import argparse
import os
import sys
from pathlib import Path

# Repos that have carried these files. Ordered by how canonical they are; the
# first one holding a matching pair wins.
CANDIDATE_REPOS = (
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    "Kijai/WanVideo_comfy",
    "lightx2v/Wan2.2-Lightning",
)

# A file is one of these LoRAs if it is a safetensors LoRA whose name carries a
# distillation marker. Substring matching for the same reason comfy_doctor uses
# it: these get renamed constantly and an exact list would rot in a week.
_SPEED_MARKERS = ("lightx2v", "lightning", "4step", "4-step", "4steps")


def _is_speed_lora(name: str) -> bool:
    low = name.lower()
    return (low.endswith(".safetensors")
            and any(m in low for m in _SPEED_MARKERS))


def _for_mode(name: str, i2v: bool) -> bool:
    """Whether this file belongs to the image-to-video or text-to-video pair.

    "ti2v" CONTAINS "i2v" — the same trap comfy_doctor hit — so the 5B model's
    files must never be counted as the 14B image-to-video pair.
    """
    low = name.lower()
    if "ti2v" in low:
        return False
    return ("i2v" in low) if i2v else ("t2v" in low)


def _half(name: str) -> str | None:
    """Which expert this file is for. The pair must match; one alone is
    useless, because the two experts split the sampling schedule between
    them."""
    low = name.lower()
    if "high" in low:
        return "high"
    if "low" in low:
        return "low"
    return None


def find_pair(files: list[str], i2v: bool) -> dict[str, str]:
    """The {high, low} pair among `files`, or as much of it as exists."""
    out: dict[str, str] = {}
    for name in sorted(files):
        if not _is_speed_lora(name) or not _for_mode(name, i2v):
            continue
        half = _half(name)
        if half and half not in out:
            out[half] = name
    return out


def _resolve_dest(explicit: str | None) -> Path | None:
    """Where ComfyUI reads LoRAs from.

    No auto-magic guessing of an install location: a wrong guess writes 600MB
    into a folder nothing reads, and the symptom is identical to not having
    downloaded it at all.
    """
    raw = explicit or os.environ.get("RUFUS_COMFY_LORAS", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", help="ComfyUI's models/loras directory")
    ap.add_argument("--i2v", action="store_true",
                    help="fetch the image-to-video pair instead of text-to-video")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be downloaded, fetch nothing")
    args = ap.parse_args(argv)

    dest = _resolve_dest(args.dest)
    if dest is None and not args.dry_run:
        print("Need the ComfyUI LoRA folder. Pass --dest, e.g.\n"
              "  python scripts/fetch_wan_lora.py --dest "
              "\"C:/ComfyUI/models/loras\"\n"
              "or set RUFUS_COMFY_LORAS. It must already exist — this script "
              "will not create a folder ComfyUI has never heard of, because "
              "a file in the wrong place looks exactly like a file that was "
              "never downloaded.")
        return 2

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed in this environment:\n"
              "  .venv/Scripts/python.exe -m pip install huggingface_hub")
        return 2

    api = HfApi()
    want = "i2v" if args.i2v else "t2v"
    for repo in CANDIDATE_REPOS:
        try:
            files = list(api.list_repo_files(repo))
        except Exception as e:
            print(f"[lora] {repo}: unreachable ({e})")
            continue

        pair = find_pair(files, args.i2v)
        if len(pair) < 2:
            found = ", ".join(sorted(pair.values())) or "nothing matching"
            print(f"[lora] {repo}: no complete {want} pair — found {found}")
            continue

        print(f"[lora] {repo}: found the {want} pair")
        for half in ("high", "low"):
            print(f"    {half}-noise: {pair[half]}")
        if args.dry_run:
            return 0

        for half in ("high", "low"):
            name = pair[half]
            out = dest / Path(name).name
            if out.exists():
                print(f"[lora] {out.name} already present — skipping")
                continue
            print(f"[lora] downloading {Path(name).name}…")
            hf_hub_download(repo_id=repo, filename=name,
                            local_dir=str(dest), local_dir_use_symlinks=False)
        print(f"\nDone. Both files are in {dest}.\n"
              f"NOW RE-EXPORT: downloading the LoRA changes nothing by itself.\n"
              f"  1. ComfyUI → open the Wan 2.2 text-to-video template\n"
              f"  2. Turn the 4-step LoRA toggle ON (steps → 4, cfg → 1.0)\n"
              f"  3. Set the positive prompt to exactly RUFUS_PROMPT\n"
              f"  4. Run it once, then Workflow → Export (API) →\n"
              f"     config/wan_t2v_api.json\n"
              f"Verify with:  python scripts/comfy_doctor.py wan_t2v")
        return 0

    print(f"\nNo complete {want} LoRA pair found in any known repo. The layout "
          f"has probably moved again — search HuggingFace for "
          f"\"wan2.2 {want} lightx2v 4steps\" and drop the high-noise and "
          f"low-noise files into ComfyUI's models/loras by hand. They must be "
          f"a matching pair; one alone does nothing, because the two experts "
          f"split the sampling schedule.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
