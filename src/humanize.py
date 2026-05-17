"""Humanization layer — adds controlled randomness to avoid platform detection.

YouTube and TikTok flag accounts with robotic patterns:
  - identical upload times every day
  - identical title casing
  - identical TTS speed
  - identical hook structures

This module injects variability so every output feels organic.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Upload timing
# ---------------------------------------------------------------------------

def random_upload_delay_seconds(niche: str = "general") -> float:
    """Return a random human-like delay (seconds) before uploading.

    Distribution is bimodal — either a short gap (post immediately-ish) or a
    longer gap (schedule for peak hours).  Never zero to avoid instant-upload
    fingerprint.
    """
    if random.random() < 0.25:
        # Short: 10–40 minutes
        return random.uniform(10 * 60, 40 * 60)
    # Long: 1–5 hours
    return random.uniform(3600, 5 * 3600)


def jittered_upload_time(base_dt: datetime, spread_minutes: int = 20) -> datetime:
    """Offset *base_dt* by a random amount within ±*spread_minutes*."""
    offset = random.randint(-spread_minutes * 60, spread_minutes * 60)
    return base_dt + timedelta(seconds=offset)


# ---------------------------------------------------------------------------
# Title / description naturalisation
# ---------------------------------------------------------------------------

_FILLER_PREFIXES = [
    "", "", "",          # most of the time: no prefix
    "Here's why ",
    "Wait — ",
    "This is ",
]

_EMPHASIS_SWAPS = {
    "very": ["really", "extremely", "incredibly", "genuinely"],
    "good": ["great", "solid", "strong", "excellent"],
    "bad": ["terrible", "awful", "rough", "poor"],
    "big": ["huge", "massive", "major", "significant"],
    "fast": ["quick", "rapid", "swift", "speedy"],
}


def naturalise_title(title: str) -> str:
    """Apply light randomisation to title phrasing without changing meaning."""
    # Randomly capitalise or not certain words
    words = title.split()
    result = []
    for word in words:
        clean = word.strip(".,!?\"'")
        lower = clean.lower()
        if lower in _EMPHASIS_SWAPS and random.random() < 0.3:
            replacement = random.choice(_EMPHASIS_SWAPS[lower])
            # preserve trailing punctuation
            punct = word[len(clean):]
            result.append(replacement + punct)
        else:
            result.append(word)
    title = " ".join(result)

    # Occasionally prepend a filler
    prefix = random.choice(_FILLER_PREFIXES)
    if prefix and not title[0].isupper():
        title = title[0].upper() + title[1:]
    return prefix + title


def vary_description(description: str) -> str:
    """Add a small random variation to description length to avoid fingerprinting."""
    endings = [
        "\n\n#shorts #viral",
        "\n\n👇 Drop your thoughts below.",
        "\n\n#trending",
        "",
        "\n\nLike & subscribe for more.",
    ]
    return description.rstrip() + random.choice(endings)


# ---------------------------------------------------------------------------
# TTS / voice variation
# ---------------------------------------------------------------------------

def random_tts_speed() -> float:
    """Return a minimal variation in TTS speed (0.98–1.02) for natural but steady delivery."""
    return round(random.gauss(1.0, 0.01), 3)


def random_pause_duration() -> float:
    """Return a natural pause length (seconds) between script sections."""
    return round(random.uniform(0.4, 1.2), 2)


# ---------------------------------------------------------------------------
# Hook / script style
# ---------------------------------------------------------------------------

HOOK_STYLES = [
    "question",       # "Have you ever wondered why…?"
    "stat",           # "9 out of 10 people don't know this…"
    "story",          # "Last year I lost everything trying to…"
    "controversy",    # "Everyone is wrong about…"
    "prediction",     # "In 2 years this will change everything…"
    "challenge",      # "Try doing this for 30 days and…"
    "revelation",     # "The secret that [authority] doesn't want you to know…"
]


def random_hook_style() -> str:
    return random.choice(HOOK_STYLES)


def random_cta_style() -> str:
    ctas = [
        "Subscribe for more.",
        "Like if this helped.",
        "Follow for weekly tips.",
        "Share this with someone who needs it.",
        "Comment your thoughts below.",
    ]
    return random.choice(ctas)


# ---------------------------------------------------------------------------
# Render variation
# ---------------------------------------------------------------------------

def random_subtitle_style() -> dict:
    """Return a slight variation in subtitle font size and colour."""
    sizes = [52, 54, 56, 58, 60]
    colours = ["white", "#FFFAFA", "#F5F5F5"]  # near-white variants
    return {
        "font_size": random.choice(sizes),
        "colour": random.choice(colours),
        "shadow_opacity": round(random.uniform(0.6, 0.9), 2),
    }


def random_bgm_volume() -> float:
    """Slight BGM volume variation (0.06–0.12) so edits don't sound identical."""
    return round(random.uniform(0.06, 0.12), 3)
