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


# ---------------------------------------------------------------------------
# Script humanizer — removes AI writing patterns from generated script text
# Based on Wikipedia's "Signs of AI writing" documented patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, replacement string)
_AI_VOCAB: list[tuple[re.Pattern, str]] = [
    # Signposting openers — announce before doing; just do it
    (re.compile(r"\bLet'?s (?:dive (?:in|into)|explore|break (?:this )?down|take a (?:look|deep dive))[^.!?]*[.!?]?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bHere'?s what you need to know[.:]*\s*", re.IGNORECASE), ""),
    (re.compile(r"\bWithout (?:further )?ado[,.:]*\s*", re.IGNORECASE), ""),
    (re.compile(r"\bNow let'?s (?:look at|turn to|examine)[^.!?]*[.!?]?\s*", re.IGNORECASE), ""),
    # Copula avoidance → simpler verbs
    (re.compile(r"\bserves as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bstands as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bmarks as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bfunctions as\b", re.IGNORECASE), "is"),
    # Promotional / inflated vocabulary
    (re.compile(r"\bgroundbreaking\b", re.IGNORECASE), "new"),
    (re.compile(r"\bpivotal\b", re.IGNORECASE), "key"),
    (re.compile(r"\bvibrant\b", re.IGNORECASE), "active"),
    (re.compile(r"\btapestry\b", re.IGNORECASE), "mix"),
    (re.compile(r"\blandscape\b(?= of| for)", re.IGNORECASE), "world"),
    (re.compile(r"\bdelve\b", re.IGNORECASE), "get"),
    (re.compile(r"\bunderscores?\b", re.IGNORECASE), "shows"),
    (re.compile(r"\bhighlights? the\b", re.IGNORECASE), "shows the"),
    (re.compile(r"\bfosters?\b", re.IGNORECASE), "builds"),
    (re.compile(r"\bfostering\b", re.IGNORECASE), "building"),
    (re.compile(r"\bshowcases?\b", re.IGNORECASE), "shows"),
    (re.compile(r"\btestament to\b", re.IGNORECASE), "proof of"),
    (re.compile(r"\bembark(?:s|ing|ed)? on\b", re.IGNORECASE), "start"),
    (re.compile(r"\bnestled\b", re.IGNORECASE), "located"),
    (re.compile(r"\bbreathtaking\b", re.IGNORECASE), "striking"),
    (re.compile(r"\bprofoundly?\b", re.IGNORECASE), "deeply"),
    (re.compile(r"\bintricate\b", re.IGNORECASE), "complex"),
    (re.compile(r"\bcrucially?\b", re.IGNORECASE), "importantly"),
    (re.compile(r"\bvitally?\b", re.IGNORECASE), "importantly"),
    (re.compile(r"\benhances?\b", re.IGNORECASE), "improves"),
    (re.compile(r"\benhancing\b", re.IGNORECASE), "improving"),
    (re.compile(r"\baligns? with\b", re.IGNORECASE), "fits"),
    (re.compile(r"\bgarners?\b", re.IGNORECASE), "gets"),
    # Excessive hedging
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bat this point in time\b", re.IGNORECASE), "now"),
    (re.compile(r"\bin the event that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bhas the ability to\b", re.IGNORECASE), "can"),
    (re.compile(r"\bit is (?:important|worth noting|crucial) to note that\b", re.IGNORECASE), ""),
    (re.compile(r"\bit (?:can|should) be noted that\b", re.IGNORECASE), ""),
    # Generic positive conclusions
    (re.compile(r"\bthe future (?:looks|is) bright[^.!?]*[.!?]", re.IGNORECASE), ""),
    (re.compile(r"\bexciting times (?:lie|are) ahead[^.!?]*[.!?]", re.IGNORECASE), ""),
    # Chatbot artifacts
    (re.compile(r"\bI hope this helps[.!]?\s*", re.IGNORECASE), ""),
    (re.compile(r"\blet me know if you(?:'?d like| have)[^.!?]*[.!?]\s*", re.IGNORECASE), ""),
    (re.compile(r"\bGreat question[.!]\s*", re.IGNORECASE), ""),
    (re.compile(r"\bOf course[.!]\s*", re.IGNORECASE), ""),
    (re.compile(r"\bCertainly[.!]\s*", re.IGNORECASE), ""),
]

_AI_SIGNPOST_REMOVE = re.compile(
    r"\b(?:In today['']s|In this) (?:video|episode)[^,;.!?]*[,;]?\s*",
    re.IGNORECASE,
)


def humanize_script_section(text: str) -> str:
    """
    Rule-based humanization for a single script section.

    Removes AI writing patterns (Wikipedia "Signs of AI writing"):
    signposting, AI vocabulary, copula avoidance, excessive hedging,
    promotional language, chatbot artifacts, generic conclusions.
    """
    if not text:
        return text

    text = _AI_SIGNPOST_REMOVE.sub("", text)

    for pattern, replacement in _AI_VOCAB:
        text = pattern.sub(replacement, text)

    # Reduce em dash overuse: word—word → word, word
    text = re.sub(r"(\w)\s*—\s*(\w)", r"\1, \2", text)

    # Clean up artifacts from removals
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()


def humanize_script(script) -> None:
    """
    In-place humanization of a VideoScript object.
    Applies humanize_script_section to hook, each section, and CTA.
    """
    if getattr(script, "hook", None):
        script.hook = humanize_script_section(script.hook)

    for section in getattr(script, "sections", []) or []:
        for key in ("script", "text"):
            if section.get(key):
                section[key] = humanize_script_section(section[key])

    if getattr(script, "call_to_action", None):
        script.call_to_action = humanize_script_section(script.call_to_action)
