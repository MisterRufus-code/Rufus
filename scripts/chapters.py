#!/usr/bin/env python3
"""
chapters.py — the timestamp list a long video is expected to carry.

WHY THIS IS NOT DECORATION. A nine-minute video without chapters is a nine
-minute video a viewer has to commit to blind. With them, the description
shows what is inside, the scrubber grows named segments, and the sections the
outline worked to build become visible before anybody presses play. Every
channel this format is aimed at has them; a channel that does not is
identifiably an automated one.

WHY IT CAN BE DONE HONESTLY HERE. Guessing timestamps by dividing the runtime
by the number of sections would be worse than having none — a chapter that
lands thirty seconds off is a promise the video breaks four times. This does
not guess. Whisper already transcribed the finished voice track with a
timestamp on every word, so each section's real start is found by matching its
opening words against that stream. A section whose opening cannot be found is
DROPPED rather than estimated.

YOUTUBE'S OWN RULES, which are the reason a partial list is not shipped:
  - the first timestamp must be 0:00
  - there must be at least three
  - each chapter must run at least ten seconds
Break any of them and YouTube does not show an incomplete list, it silently
shows none — so a list that cannot satisfy all three is not worth putting in
the description, and this returns nothing instead.

CONTRACT: pure, no network, never raises. No word timings, no titles, a
transcript that does not match — all return an empty list.
"""

from __future__ import annotations

import re

# YouTube's minimums, and they are the platform's rather than ours.
MIN_CHAPTERS = 3
MIN_CHAPTER_S = 10.0

# How many opening words to match a section by. Six is long enough to be
# unique in a 1,300-word script and short enough to survive one Whisper
# mishearing at the start of a sentence, which the descending retry covers.
_PROBE_LENS = (6, 5, 4, 3)

_INTRO_TITLE = "Intro"


def enabled() -> bool:
    """Chapters are a long-form thing. Three of them inside forty seconds
    would each be shorter than the platform's own ten-second floor."""
    try:
        import video_format
        return video_format.is_long()
    except Exception:
        return False


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _stamp(seconds: float) -> str:
    """0:00, 4:07, 1:02:05 — YouTube's own format, hours only when there are."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _find(probe: list[str], stream: list[tuple[float, str]], cursor: int) -> int:
    """Index in `stream` where `probe` starts, at or after cursor. -1 if absent.

    Two kinds of give, because the transcript is of SPOKEN audio and will not
    be a character-perfect copy of the script:

      - the probe shortens (6 words down to 3), for a mishearing late in it;
      - the probe may start one or two words in, for a mishearing at the head,
        which is where they cluster — a TTS engine runs the last word of the
        previous sentence into the first of the next often enough that a
        section's opening word is the least reliable word in it.

    Skipping a word or two costs under a second of precision on a mark that
    only has to be right to the second. Giving up costs the whole chapter.
    """
    if not probe:
        return -1
    for offset in (0, 1, 2):
        for k in _PROBE_LENS:
            if offset + k > len(probe):
                continue
            head = probe[offset:offset + k]
            for i in range(cursor, len(stream) - k + 1):
                if [w for _t, w in stream[i:i + k]] == head:
                    return i
    return -1


def locate(paragraphs: list[str], words: list[tuple[float, str]]) -> list[float]:
    """Each paragraph's real start time in the finished audio. -1.0 = not found.

    `words` is Whisper's word stream as (start_seconds, word) — audio_gen
    exports exactly that as LAST_WORDS, so this reads the timing of the audio
    that actually shipped rather than of the script that was requested. The
    two differ: a TTS engine drops a word, and every estimate after it is late.
    """
    stream = [(float(t), _norm(w)) for t, w in words if _norm(w)]
    out: list[float] = []
    cursor = 0
    for para in paragraphs:
        probe = [n for n in (_norm(w) for w in para.split()) if n]
        idx = _find(probe, stream, cursor)
        if idx < 0:
            out.append(-1.0)
            continue
        out.append(stream[idx][0])
        cursor = idx + 1
    return out


def build(script: str, words: list[tuple[float, str]],
          titles: list[str], duration: float = 0.0) -> list[tuple[float, str]]:
    """(start_seconds, title) per chapter, or [] if a valid list is impossible.

    `titles` names the sections of the script IN ORDER, aligned with the
    paragraphs after the opening — longform_writer returns the titles of the
    sections that actually survived into the script, which is not the same as
    the titles it planned.
    """
    if not script or not words or not titles:
        return []

    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return []

    # Paragraph 0 is the cold open and thesis; it always starts at 0:00 by
    # definition, so it is never matched and never dropped.
    body = paragraphs[1:]
    starts = locate(body, words)

    found = [(t, str(titles[i]).strip())
             for i, t in enumerate(starts)
             if i < len(titles) and t > 0 and str(titles[i]).strip()]

    chapters: list[tuple[float, str]] = [(0.0, _INTRO_TITLE)]
    for start, title in sorted(found, key=lambda x: x[0]):
        if start - chapters[-1][0] < MIN_CHAPTER_S:
            continue                       # too close to the last one to count
        if duration and duration - start < MIN_CHAPTER_S:
            continue                       # a chapter that ends before it reads
        chapters.append((start, title))

    if len(chapters) < MIN_CHAPTERS:
        # Not "ship what we have" — YouTube shows nothing for an invalid list,
        # so a short one is a description with orphan timestamps in it.
        return []
    return chapters


def as_lines(chapters: list[tuple[float, str]]) -> str:
    """The description block, exactly as YouTube parses it: one per line,
    timestamp first, ascending, starting at 0:00."""
    return "\n".join(f"{_stamp(t)} {title}" for t, title in chapters)
