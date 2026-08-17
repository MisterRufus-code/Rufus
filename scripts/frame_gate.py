#!/usr/bin/env python3
"""
frame_gate.py — should this frame be rendered again?

THE GAP THIS CLOSES, and it is the one the owner has been feeling for months.
Making a video by hand means re-rolling a picture until it is good. This
pipeline renders each picture ONCE and ships it. The retry loop is already
there — comfy_client.generate_clips runs up to MAX_DUP_RETRIES attempts with a
fresh seed each time — but the only thing that can reject a frame is a
perceptual-duplicate check. A frame that is a correct, non-duplicate drawing
and is still unusable gets accepted on the first try:

  - a six-panel contact sheet of the same figure, captioned (seen twice in one
    real gallery of sixteen)
  - a figure floating on blank paper with the scene missing
  - a picture that does not show what its prompt asked for
  - readable lettering, which is the clearest sign a machine made the video

So the loop is not the missing piece. Its standard is one check wide, and this
module is the rest of the standard.

CHEAPEST FIRST, because the expensive one shares a GPU with the renderer that
is mid-run. Two pixel checks that need nothing but PIL, then — only if it is
switched on — one question to a vision model.

    RUFUS_FRAME_GATE=1     turn the pixel checks on (off by default)
    RUFUS_VISION_GATE=1    also ask the vision model, per frame. Costs seconds
                           a frame and wants the card ComfyUI is holding.

THRESHOLDS ARE DELIBERATELY CONSERVATIVE, and the reason is written into this
repo's history twice: a check that fires on good frames is a check people
learn to scroll past, and here it costs GPU time as well. Everything below is
tuned to catch a frame nobody would defend, not a frame somebody might not
love. `--dry-run` against a folder of real keyframes is how the numbers get
set; guessing them is how this becomes the drift warning that fired on seven
shots in ten.

WHAT IT CANNOT SEE. The pale, washed-out gallery — every background beige,
every frame technically complete — is not in here. That is the checkpoint
overriding an explicit instruction, and no pixel threshold separates "the
style is washed out" from "the style is pale on purpose"; ink_woodcut is
monochrome on white paper by design. That one is a model or LoRA decision.

CONTRACT: never raises, never blocks a render on its own. A frame it cannot
read is a frame it passes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── the thresholds ───────────────────────────────────────────────────────────

# Share of near-white pixels past which the frame is bare paper rather than a
# picture. 0.90 and not 0.75: ink_woodcut is a monochrome engraving on white
# and a legitimate frame in that style can be three-quarters white. This
# catches "there is essentially nothing here", which is what the white-void bug
# produced at its worst, and deliberately misses the milder version of it.
BLANK_SHARE = 0.90
_NEAR_WHITE = 235

# A gutter row/column: near-white and near-uniform all the way across.
_GUTTER_MEAN = 242
_GUTTER_RANGE = 12
# Ignore the outer eighth — a frame with a white sky at the top or a bright
# margin at the edge is not a grid, and every drawing has quiet edges.
_EDGE_MARGIN = 0.12
# A contact sheet needs a horizontal AND a vertical interior gutter. One alone
# is a horizon, a tabletop or a wall.
_GRID_MIN_BANDS = 1

_ANALYSIS_PX = 128


def enabled() -> bool:
    return os.environ.get("RUFUS_FRAME_GATE", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def vision_enabled() -> bool:
    """The per-frame vision question, which is the expensive one.

    Separate from RUFUS_VISION (the after-the-fact review of a finished run)
    because they cost differently and compete differently: the review runs when
    the GPU is free, this runs while ComfyUI is holding it.
    """
    return os.environ.get("RUFUS_VISION_GATE", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def _gray(path: Path, size: int = _ANALYSIS_PX):
    """Downscaled greyscale pixels as a list of rows. None if unreadable."""
    try:
        from PIL import Image
        img = Image.open(str(path)).convert("L").resize(
            (size, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.tobytes())
    return [px[r * size:(r + 1) * size] for r in range(size)]


def blank_share(path: Path) -> float:
    """Share of the frame that is near-white. 0.0 if it cannot be read."""
    rows = _gray(path)
    if not rows:
        return 0.0
    total = sum(len(r) for r in rows)
    white = sum(1 for r in rows for p in r if p >= _NEAR_WHITE)
    return round(white / total, 3) if total else 0.0


def _bands(lines: list[list[int]]) -> int:
    """Interior gutter bands among these lines (rows or columns).

    A band is one or more consecutive near-white, near-uniform lines. Counting
    BANDS rather than lines is what makes the number mean "how many times the
    picture is divided" instead of "how thick the divider is".
    """
    n = len(lines)
    lo, hi = int(n * _EDGE_MARGIN), int(n * (1 - _EDGE_MARGIN))
    bands, in_band = 0, False
    for i in range(lo, hi):
        line = lines[i]
        gutter = (sum(line) / len(line) >= _GUTTER_MEAN
                  and max(line) - min(line) <= _GUTTER_RANGE)
        if gutter and not in_band:
            bands += 1
        in_band = gutter
    return bands


# A frame emptier than this cannot be a contact sheet, because there are no
# panels for the gutters to separate. Without this guard a nearly blank frame
# reads as a grid in both directions — every row and column of it is near-white
# and near-uniform, which is the definition of a gutter — and the re-roll would
# be told "never draw a grid" about a picture whose problem is that there is
# nothing in it. Caught by the test for the white-void frame.
_GRID_MAX_WHITE = 0.80


def grid_bands(path: Path) -> tuple[int, int]:
    """(horizontal, vertical) interior gutter bands — a contact sheet's tell."""
    rows = _gray(path)
    if not rows:
        return (0, 0)
    total = sum(len(r) for r in rows)
    white = sum(1 for r in rows for p in r if p >= _NEAR_WHITE)
    if total and white / total > _GRID_MAX_WHITE:
        return (0, 0)
    cols = [[rows[r][c] for r in range(len(rows))] for c in range(len(rows[0]))]
    return (_bands(rows), _bands(cols))


# ── the verdict ──────────────────────────────────────────────────────────────

# Each reason carries the clause that goes back into the retry prompt. A
# re-roll with the same prompt and a new seed is the same prompt — a person
# who re-rolls says what was wrong with the last one.
_RETRY_HINTS = {
    "contact_sheet": ("ONE SINGLE SCENE seen from ONE camera at ONE moment — "
                      "never a grid, never side-by-side panels, never the same "
                      "figure repeated in a row, never a character sheet."),
    "blank_frame": ("Draw the WHOLE place: a ground under the subject, what is "
                    "behind it, and the things that say where this is. Never a "
                    "subject floating on blank paper."),
    "misses_the_prompt": "",      # filled from the model's own answer
    "lettering": ("No words, no captions, no labels, no signs, no numbers "
                  "anywhere in the picture. Any surface that would carry "
                  "writing is drawn blank."),
}


def retry_hint(reason: str, detail: str = "") -> str:
    """The sentence to append to the prompt on the next attempt."""
    base = _RETRY_HINTS.get(reason, "")
    if reason == "misses_the_prompt" and detail:
        return f"The last attempt was missing this, so make it clearly visible: {detail}."
    return base


def check(path: Path, prompt: str = "", client=None) -> tuple[bool, str, str]:
    """(ok, reason, detail) for one rendered frame. Never raises.

    ok=True on anything it cannot read or cannot judge — a frame that could not
    be looked at is not a frame that failed.
    """
    try:
        # EMPTY FIRST. An almost blank frame satisfies the gutter test in both
        # directions trivially, and answering "contact_sheet" would send the
        # next attempt a note about grids when the problem is that nothing is
        # in the picture.
        share = blank_share(path)
        if share >= BLANK_SHARE:
            return (False, "blank_frame",
                    f"{share:.0%} of the frame is bare paper")

        h, v = grid_bands(path)
        if h >= _GRID_MIN_BANDS and v >= _GRID_MIN_BANDS:
            return (False, "contact_sheet",
                    f"{h}x{v} interior gutters — this is a panel layout")

        if vision_enabled() and prompt:
            import vision_review
            seen = vision_review.look(path, prompt, client=client)
            if seen:
                if not seen.get("shows_it"):
                    return (False, "misses_the_prompt",
                            seen.get("missing") or "")
                if seen.get("lettering"):
                    return (False, "lettering",
                            seen.get("lettering_note") or "")
    except Exception as e:
        print(f"[gate] {Path(path).name}: could not be checked ({e}) — kept")
    return (True, "", "")


# ── the dry run ──────────────────────────────────────────────────────────────

def dry_run(folder: Path) -> dict:
    """What the gate WOULD have rejected in a folder of finished frames.

    The step that sets the thresholds honestly. A gate costs GPU time on every
    rejection, so it has to be measured against real output before it is
    switched on — against the frames of a run whose gallery is already known to
    be good or bad, where the answer can be checked by looking.
    """
    folder = Path(folder)
    frames = sorted(p for p in folder.rglob("*.png") if p.is_file())
    out = {"looked_at": len(frames), "rejected": [], "folder": str(folder)}
    for f in frames:
        h, v = grid_bands(f)
        share = blank_share(f)
        reason = ""
        if share >= BLANK_SHARE:
            reason = f"blank_frame ({share:.0%} white)"
        elif h >= _GRID_MIN_BANDS and v >= _GRID_MIN_BANDS:
            reason = f"contact_sheet ({h}x{v} gutters)"
        if reason:
            out["rejected"].append({"frame": f.name, "reason": reason})
        print(f"  {'REJECT' if reason else '  ok  '}  {f.name:<28} "
              f"gutters {h}x{v}  white {share:.0%}  {reason}")
    n = out["looked_at"]
    share = len(out["rejected"]) / n if n else 0.0
    print(f"\n{len(out['rejected'])} of {n} would be re-rolled ({share:.0%}).")
    if share > 0.25:
        print("That is too many. A gate that rejects a quarter of a good "
              "gallery costs an hour of GPU and teaches nobody anything — "
              "raise the thresholds before switching it on.")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not args:
        print("Usage: python scripts/frame_gate.py --dry-run <folder-of-pngs>")
        raise SystemExit(1)
    dry_run(Path(args[0]))
