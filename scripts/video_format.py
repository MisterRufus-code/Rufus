#!/usr/bin/env python3
"""
video_format.py — what shape of video this run is making.

WHY THIS EXISTS. Every number that makes a Short a Short was written into
whichever module needed it: 1080×1920 in audio_gen and again in Short.tsx and
again in comfy_client's portrait fit, a 115-word cap in script_standards.json,
a 10–30 beat range in main._target_beats, a 1.6s minimum shot in audio_gen, a
180-second "broken render" ceiling in qc_check. Nothing was wrong with any of
them. They simply all encoded ONE format, in seven places, none of which
mentioned the others.

Asking for a second format is therefore not asking for a setting. It is asking
those seven numbers to move together, which they can only do if they live in
one place first. That is all this file is: the profile, and the readers import
it instead of writing the number down again.

    RUFUS_FORMAT=short   40-second vertical Short   (default, unchanged)
    RUFUS_FORMAT=long    long-form landscape video

WHAT A PROFILE DOES NOT DECIDE. The look (config/styles.json), the niche, the
voice, the renderer. Those are already single-sourced and are orthogonal to
shape — a stickman long-form video and a stickman Short are the same look at
different lengths, which is the point.

CONTRACT: pure, cheap, and never raises. An unknown format name falls back to
`short` and says so, because a typo that silently changed the aspect ratio of
a nine-minute render would be an expensive way to learn about it.
"""

from __future__ import annotations

import os

# ── the profiles ─────────────────────────────────────────────────────────────
#
# Every field is a number some module used to hard-code. The comments say where
# each one came from, because a profile is only trustworthy if you can see that
# it did not invent its values.
PROFILES: dict[str, dict] = {
    "short": {
        "id": "short",
        "label": "Short (vertical, ~40s)",
        # The render. 1080×1920 is the shape audio_gen, Short.tsx and
        # comfy_client's _fit_to_portrait each declared separately.
        "width": 1080,
        "height": 1920,
        # What the stills model is asked for before the fit. 832×1472 is the
        # portrait size the exported ComfyUI workflow runs at.
        "still_width": 832,
        "still_height": 1472,
        # The script. From config/script_standards.json, where the note reads
        # "115 words ≈ 45s at +6% rate … >50s bleeds completion rate".
        "words_min": 80,
        "words_max": 115,
        # The pictures. main._target_beats: one per five spoken words, floored
        # at 10 so a short script is not a slideshow of three, ceilinged at 30
        # where the storyboard call starts losing the thread.
        "words_per_picture": 5,
        "beats_min": 10,
        "beats_max": 30,
        # The cut rhythm. audio_gen.MIN_SEG — raised from 1.2 after a real
        # 24-picture run put thirteen shots on the floor.
        "min_seg_s": 1.6,
        # qc_check.MIN_DUR / MAX_DUR: outside this, the render is broken.
        "qc_min_s": 10.0,
        "qc_max_s": 180.0,
        # Burned-in captions. 140px on a 1920-tall frame is 7% of the height —
        # big, because a Short is watched on a phone at arm's length and the
        # words are half the format. MarginV 600 sits ~31% up: below the face
        # zone, above the Shorts UI that covers the bottom fifth.
        "caption_size": 140,
        "caption_margin_v": 600,
        # The word-synced insert cutaway, on a 1080-wide frame.
        "insert_w": 460,
    },
    "long": {
        "id": "long",
        "label": "Long-form (landscape, ~9 min)",
        "width": 1920,
        "height": 1080,
        "still_width": 1472,
        "still_height": 832,
        # ~150 words/minute of narration: 1,350 words is roughly nine minutes,
        # which is the length the format is aimed at. The floor is a real
        # floor — below about six minutes this is neither a Short nor
        # long-form, and lands in the gap YouTube rewards least.
        "words_min": 900,
        "words_max": 1600,
        # A picture every ~9 spoken words is a shot of roughly 3.5 seconds,
        # which is the pace an explainer holds: long enough to read the
        # drawing, short enough that nothing sits. That is ~150 pictures for a
        # nine-minute script — hours of GPU on this box, and the reason
        # long-form is a deliberate choice rather than a default.
        "words_per_picture": 9,
        "beats_min": 40,
        "beats_max": 220,
        # Calmer than a Short by design. A 3.5s average with a 2.5s floor
        # leaves room for the cut planner to land on real pauses.
        "min_seg_s": 2.5,
        "qc_min_s": 240.0,
        "qc_max_s": 1500.0,
        # NOT the Shorts numbers scaled — a different viewing situation. 140px
        # on a 1080-tall frame would be 13% of the height, and long-form is
        # watched further away on a bigger screen where the picture is the
        # point and the caption is an aid. 58px is ~5.4%, the broadcast
        # subtitle proportion. MarginV 70 puts it near the bottom edge, where
        # there is no app UI to avoid and no reason to cover the frame.
        "caption_size": 58,
        "caption_margin_v": 70,
        # Proportionally smaller on a wider frame: 460 of 1080 is 43% of the
        # width and would swallow a landscape shot.
        "insert_w": 520,
    },
}

DEFAULT = "short"


def name() -> str:
    """The active format id, always one of PROFILES."""
    raw = (os.environ.get("RUFUS_FORMAT") or "").strip().lower()
    if not raw:
        return DEFAULT
    if raw in PROFILES:
        return raw
    # LOUD, because the alternative is a nine-minute render in the wrong
    # aspect ratio discovered at upload time.
    print(f"[format] RUFUS_FORMAT={raw!r} is not a known format "
          f"({', '.join(sorted(PROFILES))}) — using {DEFAULT}")
    return DEFAULT


def profile(fmt: str | None = None) -> dict:
    """The active profile as a plain dict. Never raises."""
    return dict(PROFILES.get(fmt or name(), PROFILES[DEFAULT]))


def get(key: str, default=None):
    """One field of the active profile."""
    return profile().get(key, default)


def is_long() -> bool:
    return name() == "long"


def dimensions() -> tuple[int, int]:
    p = profile()
    return int(p["width"]), int(p["height"])


def still_dimensions() -> tuple[int, int]:
    p = profile()
    return int(p["still_width"]), int(p["still_height"])


def is_portrait() -> bool:
    w, h = dimensions()
    return h >= w


def target_beats(word_count: int, fmt: str | None = None) -> int:
    """How many pictures a script of this length should become.

    The rule main._target_beats has always used, with the constants coming
    from the profile instead of from the function body. SD_CLIPS still
    overrides it — that is main's business, not this module's.
    """
    p = profile(fmt)
    per = max(1, int(p["words_per_picture"]))
    return max(int(p["beats_min"]),
               min(int(p["beats_max"]), round(word_count / float(per))))


def describe() -> str:
    """One line for the run header, so a surprising render has its cause in
    the log rather than in somebody's memory of what they clicked."""
    p = profile()
    return (f"{p['label']} — {p['width']}×{p['height']}, "
            f"{p['words_min']}–{p['words_max']} words, "
            f"{p['beats_min']}–{p['beats_max']} pictures")
