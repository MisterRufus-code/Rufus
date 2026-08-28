"""Named looks for the burned-in words, chosen before a render.

WHY THIS IS A CHOICE AND NOT A SETTING. The captions are half of what a Short
IS — one word at a time in capitals with a colour pop is a genre, not a detail
— and this channel has shipped exactly one of them because the numbers behind
it live in video_format.PROFILES, next to the frame size and the QC bounds,
where nobody would think to look and where changing them changes every future
video at once. The owner asked for "a few subtitles styles to choose from
before the rendering", which is the right shape: a per-video decision, made
where the other per-video decisions are made, on the page that renders it.

WHAT A PRESET IS ALLOWED TO CHANGE, and the honest limit. Two halves reach the
renderer by different roads:

  - the READING half — how many words to a card, and whether it shouts — is
    built in audio_gen._cluster_words, which BOTH renderers call. A preset
    changes it once and both engines obey.
  - the LOOK half — size, height up the frame, outline, box, colour — is
    written into the ASS style line that FFmpeg burns. The Remotion engine
    draws its own captions in TSX and does not read it.

So a preset is fully honoured on the default FFmpeg path and half honoured on
the opt-in Remotion one, and audio_gen says so out loud when the two disagree
rather than leaving a picked style silently ignored. Claiming otherwise here
would be this repo's oldest bug — built, wired, tested and never actually fed.

SHARES OF THE FRAME RATHER THAN PIXELS, deliberately. 140px is 7% of a
1920-tall vertical frame and 13% of a 1080-tall landscape one; a preset that
hard-coded it would be a different design in long-form than in Shorts. Sizes
and heights here are fractions of the FRAME HEIGHT, so "big" means big in
whichever shape is being rendered. Multiplying the format's own number would
have been the same mistake one level up — long-form already ships the
broadcast look, so scaling it down again to "make it broadcast" lands at 26px.

AND THE DEFAULT IS "WHATEVER THE FORMAT ASKS FOR", not one of the looks. The
per-format caption numbers in video_format are a real decision — one word
shouting is right for a phone at arm's length and wrong for nine minutes on a
television — so the preset that runs when nobody chose says nothing at all and
lets them stand. Every other preset is somebody overriding that on purpose.
"""

from __future__ import annotations

import os

# ASS colours are &HAABBGGRR — blue and red swapped against the hex everyone
# writes. Named here so a preset reads as a colour rather than as a hex puzzle.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
GOLD = "&H003FD2FF"        # #FFD23F, the channel accent
CREAM = "&H00E8F4FF"
SEMI_BLACK = "&HA0000000"  # ~63% opaque, for the boxed look

# BorderStyle in ASS: 1 draws an outline and a shadow, 3 fills an opaque box
# behind the text. There is no third option worth offering.
OUTLINE, BOX = 1, 3

# Every key a preset may set, and what it means. Anything absent falls through
# to the active format profile, so a preset says only what it changes.
PRESETS: dict[str, dict] = {
    "format": {
        "label": "Whatever this format asks for",
        "blurb": "The numbers the format was designed around — one shouted "
                 "word at a time for a Short, four-word phrases near the "
                 "bottom edge for long-form. This is what every video has "
                 "shipped with; pick something else only to override it on "
                 "purpose.",
    },
    "word_pop": {
        "label": "One word, shouting",
        "blurb": "A single word at a time, capitals, very large, high on the "
                 "frame, each one popping as it lands. Identical to the "
                 "default in a Short — choose it to force that treatment onto "
                 "long-form, where the format would otherwise use subtitles.",
        "words": 1, "upper": True,
        "height_pct": 0.073, "bottom_pct": 0.3125,
        "primary": WHITE, "outline": 4, "shadow": 2,
        "border_style": OUTLINE, "pop": True,
    },
    "phrase": {
        "label": "Short phrases, calmer",
        "blurb": "Three words to a card in ordinary case, a size down, same "
                 "height on the frame and the same pop. It stops shouting — "
                 "for a video whose script is doing the work.",
        "words": 3, "upper": False,
        "height_pct": 0.053, "bottom_pct": 0.31,
        "primary": WHITE, "outline": 3, "shadow": 2,
        "border_style": OUTLINE, "pop": True,
    },
    "broadcast": {
        "label": "Subtitles, out of the way",
        "blurb": "Four words in natural case, small, near the bottom edge, no "
                 "animation. The look television uses — it reads as an aid to "
                 "the picture rather than as the point of it. On a Short that "
                 "puts them under the app's own buttons, which is a real cost "
                 "and the reason it is not the default.",
        "words": 4, "upper": False,
        "height_pct": 0.054, "bottom_pct": 0.065,
        "primary": WHITE, "accent": CREAM, "outline": 2, "shadow": 1,
        "border_style": OUTLINE, "pop": False,
    },
    "boxed": {
        "label": "Two words on a black bar",
        "blurb": "Legible over anything. Two words at a time on a solid panel "
                 "instead of an outline — for pictures busy enough that "
                 "outlined white text disappears into them.",
        "words": 2, "upper": True,
        "height_pct": 0.045, "bottom_pct": 0.28,
        "primary": WHITE, "outline": 6, "shadow": 0,
        "border_style": BOX, "back": SEMI_BLACK, "pop": False,
    },
    "gold": {
        "label": "One word, all gold",
        "blurb": "The shouting look with the accent on every word rather than "
                 "only on the numbers and the money. Loudest of them, and the "
                 "one most likely to look like every other channel.",
        "words": 1, "upper": True,
        "height_pct": 0.073, "bottom_pct": 0.3125,
        "primary": GOLD, "accent": GOLD, "outline": 4, "shadow": 2,
        "border_style": OUTLINE, "pop": True,
    },
    "none": {
        "label": "No words on screen",
        "blurb": "Burns no captions at all. The voice still carries the "
                 "script and the pictures still cut to it — but a viewer "
                 "watching on mute gets nothing, which on Shorts is most of "
                 "them. Here because it is occasionally right, not because it "
                 "is ever safe.",
        "enabled": False, "pop": False,
    },
}

# What a run gets when nobody chose. NOT one of the looks: the per-format
# caption numbers are themselves a decision, and an unattended cron render
# should keep making the video this channel already makes.
DEFAULT = "format"

ENV_VAR = "RUFUS_CAPTIONS"


def names() -> list[str]:
    """Every preset id, in the order the picker should offer them."""
    return list(PRESETS)


def name() -> str:
    """The chosen preset id, always one of PRESETS.

    Loud on an unknown name for the same reason video_format.name() is: a
    typo that silently falls back is a video rendered in a look nobody picked,
    discovered at upload time.
    """
    raw = (os.environ.get(ENV_VAR) or "").strip().lower()
    if not raw:
        return DEFAULT
    if raw in PRESETS:
        return raw
    print(f"[captions] {ENV_VAR}={raw!r} is not a known style "
          f"({', '.join(names())}) — using {DEFAULT}")
    return DEFAULT


def preset(which: str | None = None) -> dict:
    """One preset by id, or the active one."""
    return PRESETS[which if which in PRESETS else name()]


def resolve(profile: dict, which: str | None = None,
            frame_height: int | None = None) -> dict:
    """The caption numbers for a render: the format's, with a preset over them.

    Returns absolute values — a size in pixels, a margin in pixels — because
    the shares only mean anything against a frame, and every caller wants the
    answer rather than the arithmetic. A preset that names no size keeps the
    format's own, which is what makes "format" a real default rather than a
    sixth look.
    """
    p = preset(which)
    size = float(profile.get("caption_size") or 140)
    margin = float(profile.get("caption_margin_v") or 600)
    if frame_height:
        if p.get("height_pct"):
            size = float(p["height_pct"]) * frame_height
        if p.get("bottom_pct"):
            margin = float(p["bottom_pct"]) * frame_height
    return {
        "style": which if which in PRESETS else name(),
        "enabled": bool(p.get("enabled", True)),
        "words": max(1, int(p.get("words") or profile.get("caption_words") or 1)),
        "upper": bool(p["upper"] if "upper" in p
                      else profile.get("caption_upper", True)),
        "size": max(8, round(size)),
        "margin_v": max(0, round(margin)),
        "primary": p.get("primary", WHITE),
        # None means "leave the channel's own accent alone". The niche's
        # accent_color is this channel's identity and a subtitle preset has no
        # business overwriting it — only the two presets whose whole point is
        # colour name one, and they say so by naming one.
        "accent": p.get("accent"),
        "back": p.get("back", "&H80000000"),
        "outline": int(p.get("outline", 4)),
        "shadow": int(p.get("shadow", 2)),
        "border_style": int(p.get("border_style", OUTLINE)),
        "pop": bool(p.get("pop", True)),
    }
