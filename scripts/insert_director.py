#!/usr/bin/env python3
"""
insert_director.py — decide WHICH words get a picture, and exactly WHEN.

THE FORMAT THIS SERVES. The stickman/explainer shorts that do well on TikTok
are not one picture per sentence. They are a picture per NOUN: the narrator
says "palace" and a palace pops in on the beat, says "coin" and a coin pops in
half a second later. Twenty to forty small images across forty seconds, each
landing on the word that named it. The effect is that the video looks like it
is listening to itself.

WHY THIS PIPELINE CAN DO IT AND MOST CANNOT. The hard part of that style is
knowing the exact second "palace" is spoken. Editors hand-place it. Rufus
already has it for free: remotion_renderer.py transcribes the finished
voiceover with word-level timestamps and passes them to the renderer as

    words = [{"text": "PALACE", "start": 4.812, "end": 5.106}, ...]

so the timing question is already answered before this module runs. Everything
here is the other half — which words deserve a picture, and what to draw.

NO GPU, NO COMFYUI, NO NETWORK. This module plans; it never renders. That
separation is the point: a plan can be read, argued with and corrected in a
second, and only then is any GPU time spent drawing it. `python -m
insert_director "<script>"` prints the plan for a real script.

Env:
  RUFUS_INSERTS        1 (default) — 0 disables the whole layer
  RUFUS_INSERT_MAX     28  most inserts in one video
  RUFUS_INSERT_GAP     0.45 minimum seconds between two inserts
  RUFUS_INSERT_HOLD    0.70 seconds an insert stays on screen
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import storyboard  # noqa: E402  — reuse its concrete-noun vocabulary

DEFAULT_MAX = 28
DEFAULT_GAP = 0.45
DEFAULT_HOLD = 0.70

# Nouns worth a picture but too abstract for _content_words to keep, and nouns
# it keeps that are not worth interrupting the video for. The first list is
# empty on purpose: _content_words was tuned all through this project against
# real scripts, and duplicating its judgement here would give two places to fix
# the same mistake. The second exists because an INSERT has a higher bar than a
# shot description — "people" is filmable but drawing it adds nothing the beat
# image was not already showing.
_TOO_GENERIC = {
    "people", "person", "man", "woman", "men", "women", "thing", "things",
    "time", "times", "place", "places", "world", "life", "part", "parts",
    "side", "sides", "hand", "hands", "face", "faces", "eye", "eyes",
    "day", "days", "week", "month", "year", "years", "money",
    # Abstract nouns that survive every morphological rule because they are
    # short and ordinary. Observed leaking on real scripts: "a lack of foreign
    # reserves", "capital flight", "their jobs vanished".
    "lack", "capital", "value", "values", "flight", "growth", "trade", "jobs",
    "work", "wealth", "power", "control", "crisis", "panic", "risk", "risks",
    "chance", "future", "past", "history", "story", "reason", "result",
    "change", "point", "level", "rate", "rates", "cost", "costs", "price",
    "prices", "amount", "number", "numbers", "share", "shares", "stake",
}

# Common adjectives with no distinctive ending. "foreign" reads as a thing to a
# suffix rule and is a property to a viewer — the picture would be of whatever
# it modified.
_PLAIN_ADJECTIVES = {
    "foreign", "modern", "ancient", "common", "single", "double", "whole",
    "entire", "certain", "similar", "sudden", "quiet", "silent", "empty",
    "full", "clean", "clear", "sharp", "heavy", "light", "dark", "bright",
    "quick", "slow", "hard", "soft", "wide", "narrow", "deep", "high", "long",
    "short", "small", "large", "huge", "tiny", "rich", "poor", "young", "old",
    "real", "true", "false", "same", "other", "next", "last", "first", "final",
    "major", "minor", "total", "public", "private", "local", "global",
}

# AN INSERT HAS A HIGHER BAR THAN A SHOT MENTION, and the first run of this
# planner is the proof. Against a real 1893 script it chose:
#
#     february, workers, outside, railroad, receivers, jobs, coins, paper,
#     worthless, runs
#
# Four of those ten cannot be drawn as a single object. _content_words was
# tuned to answer "did any shot mention this noun", where a false keep costs
# nothing; here a false keep spends a GPU render and puts a meaningless picture
# on screen at the exact moment the viewer is listening. So the classes it
# lets through get named, by the shape that gave them away.

# Adjectives. "worthless" is a quality, and a quality has no silhouette.
_ADJECTIVE_TAILS = ("less", "ful", "ous", "ive", "able", "ible", "ish",
                    "ical", "ary", "ant", "ent")

# Positions and directions. "outside" is a relationship between things, not a
# thing — drawing it means drawing whatever it was outside OF.
_POSITIONAL = {
    "outside", "inside", "above", "below", "across", "beyond", "within",
    "without", "around", "behind", "beneath", "toward", "towards", "upward",
    "downward", "along", "amid", "among", "between", "beside", "near", "past",
    "through", "under", "upon", "ahead", "apart", "away", "back", "forward",
}

# Dates. A month is a label on time; it has no picture, and it was the very
# first thing this planner chose, because good scripts open on a date.
_CALENDAR = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "morning",
    "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
    "century", "decade", "season", "summer", "winter", "spring", "autumn",
}

# Verbs that survive every suffix rule because their plural looks like a noun.
# "runs" is the one that shipped — a bank RUN is a real thing, but the model
# draws jogging.
_VERB_PLURALS = {
    "runs", "holds", "makes", "takes", "gives", "comes", "goes", "means",
    "needs", "wants", "shows", "keeps", "seems", "turns", "moves", "finds",
    "leaves", "brings", "follows", "begins", "ends", "starts", "stops",
    "falls", "rises", "grows", "sells", "buys", "pays", "owes", "lends",
    "borrows", "spends", "saves", "loses", "wins", "sends", "asks", "tells",
}


def enabled() -> bool:
    return os.environ.get("RUFUS_INSERTS", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _cfg(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[inserts] {name}={raw!r} is not a number — using {default}")
        return default


def insert_words(script: str) -> list[str]:
    """Drawable nouns from the script, in the order they are spoken.

    Delegates the hard judgement to storyboard._content_words, which already
    knows that "vanished", "overnight" and "then" are not things and that
    "ship" and "city" are, and which was tuned against this channel's real
    scripts. Duplicating that vocabulary here would give two places to fix one
    mistake.
    """
    # CASE IS A SIGNAL AND LOWERCASING FIRST THROWS IT AWAY. The first version
    # of this function lowercased the script before extracting, which defeated
    # _content_words' own proper-noun rule and chose "philadelphia", "bangkok"
    # and "thai" as things to draw. A place name is not an object; a picture of
    # "Philadelphia" is a picture of whatever the model free-associates.
    allowed = storyboard._content_words(script or "")
    reads_as_noun = _noun_positions(script or "")

    seen: list[str] = []
    for raw in re.findall(r"\b[a-z]{4,}\b", (script or "").lower()):
        if raw in seen or raw not in allowed:
            continue
        if raw not in reads_as_noun or not _is_drawable(raw):
            continue
        seen.append(raw)
    return seen


# Words that put a NOUN next: articles, possessives, quantifiers, prepositions.
_NOUN_CUES = {
    "the", "a", "an", "this", "that", "these", "those", "their", "his", "her",
    "its", "our", "your", "my", "every", "each", "one", "two", "three", "some",
    "many", "few", "more", "most", "of", "in", "on", "at", "with", "for",
    "from", "into", "onto", "over", "under", "across", "by", "no", "another",
}

# Words that put a VERB next. "would float", "to abandon", "rushed to exchange"
# — every one of these was chosen as a thing to draw before this rule existed.
_VERB_CUES = {
    "to", "would", "will", "can", "could", "must", "should", "may", "might",
    "and", "or", "not", "never", "then", "also", "who", "which", "that",
}


def _noun_positions(script: str) -> set[str]:
    """Words that read as a NOUN somewhere in this script.

    No POS tagger here, and this repo does not need one for a heuristic — but
    English puts a reliable tell one word to the left. "the currency" is a
    noun; "would float" is a verb; "to abandon" is a verb. A word qualifies if
    ANY of its occurrences follows a noun cue, because one clear use is enough
    to know what the writer meant by it.
    """
    words = re.findall(r"[A-Za-z]+", script or "")
    seen: set[str] = set()
    verbal: set[str] = set()
    for i, w in enumerate(words):
        low = w.lower()
        seen.add(low)
        prev = words[i - 1].lower() if i else ""
        if prev in _VERB_CUES:
            verbal.add(low)
    # REJECT ON EVIDENCE, DO NOT ACCEPT ON IT. Requiring a noun cue to the LEFT
    # looked right and was far too strict: most nouns follow another noun or a
    # comma, not an article. On the 1893 script it threw away workers, coins,
    # paper and bank and kept two inserts out of a possible seven.
    #
    # The reliable signal is the negative one. "would float", "to abandon",
    # "rushed to exchange" — a verb cue immediately left is strong evidence the
    # word is a verb here, and those three were exactly what leaked. So keep
    # every word EXCEPT the ones something proved verbal.
    return seen - verbal


def _is_drawable(word: str) -> bool:
    """Whether one word names a single object a picture could BE.

    A heuristic with no POS tagger, same as everything else in this repo that
    judges words — but a stricter one than _content_words, because the cost of
    a wrong answer is different. There it was a noisy warning line; here it is
    a GPU render and a meaningless picture landing on the beat.
    """
    if word in _TOO_GENERIC or word in _POSITIONAL or word in _CALENDAR:
        return False
    if word in _PLAIN_ADJECTIVES:
        return False
    if word in _VERB_PLURALS:
        return False
    if word.endswith(_ADJECTIVE_TAILS) and word not in storyboard._FILMABLE_DESPITE_SUFFIX:
        return False
    # Everything _content_words already rejects — stopwords, the abstract
    # vocabulary, verb and concept endings — stays rejected. Delegating rather
    # than restating means one place to fix a shared mistake.
    return word in storyboard._content_words(word)


def _word_times(words: list[dict]) -> dict[str, list[float]]:
    """Every start time each spoken word occurs at, keyed by lowercase word.

    Whisper emits words with punctuation attached and in whatever case the
    renderer upper-cased them to, so both are stripped here rather than at the
    call site.
    """
    out: dict[str, list[float]] = {}
    for w in words or []:
        text = re.sub(r"[^a-z]", "", str(w.get("text", "")).lower())
        if not text:
            continue
        try:
            out.setdefault(text, []).append(float(w.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
    return out


def plan(script: str, words: list[dict], style: str = "") -> list[dict]:
    """The insert list: what to draw, and the second it lands.

    An insert is only planned for a noun the narration ACTUALLY SPEAKS. Whisper
    transcribes the finished audio, so a word the TTS swallowed or pronounced
    into something else simply has no timestamp and gets no picture — which is
    the correct outcome and not something a script-only planner could know.
    """
    if not enabled():
        return []
    limit = int(_cfg("RUFUS_INSERT_MAX", DEFAULT_MAX))
    gap = _cfg("RUFUS_INSERT_GAP", DEFAULT_GAP)
    hold = _cfg("RUFUS_INSERT_HOLD", DEFAULT_HOLD)

    spoken = _word_times(words)
    candidates: list[tuple[float, str]] = []
    for noun in insert_words(script):
        for at in spoken.get(noun, [])[:1]:   # first utterance only
            candidates.append((at, noun))
    candidates.sort()

    out: list[dict] = []
    last = -999.0
    for at, noun in candidates:
        # SPACING IS A CONTENT RULE, NOT A STYLE ONE. Two inserts inside half a
        # second read as one flicker and the viewer registers neither, so the
        # second is dropped rather than squeezed — losing a picture is cheaper
        # than losing both.
        # Compare the STORED value, not the raw one. Rounding after the check
        # can shave a hair off the gap, so the invariant would hold for numbers
        # nobody keeps and fail for the ones the renderer actually reads.
        at = round(at, 3)
        if at - last < gap:
            continue
        out.append({
            "word": noun,
            "at": at,
            "hold": hold,
            "prompt": insert_prompt(noun, style),
        })
        last = at
        if len(out) >= limit:
            break
    return out


def insert_prompt(noun: str, style: str = "") -> str:
    """What to draw for one insert.

    Deliberately spare. An insert is on screen for under a second at a fraction
    of frame size, so detail is wasted GPU: what has to read instantly is the
    SILHOUETTE. "A single X, centred, plain background" gives the model nothing
    to clutter with, and the channel's own style suffix does the rest so an
    insert belongs to the same world as the beat behind it.
    """
    base = (f"A single {noun}, centred in frame, one clear object, "
            f"plain flat background, bold readable silhouette, no text, "
            f"no words, no letters")
    return f"{base}. {style.strip()}" if style and style.strip() else base


def sfx_events(inserts: list[dict], gain: float) -> list[tuple[float, float]]:
    """(time, gain) per insert, in the shape audio_gen's mixer already takes.

    Its ffmpeg path delays each event into place from its own input, so a pop
    per insert needs no new mixing code — only the list.
    """
    return [(float(i["at"]), gain) for i in inserts if "at" in i]


def describe(inserts: list[dict]) -> str:
    """One human line — the log has to make a bad plan obvious at a glance."""
    if not inserts:
        return "[inserts] none planned"
    span = f"{inserts[0]['at']:.1f}s-{inserts[-1]['at']:.1f}s"
    names = ", ".join(i["word"] for i in inserts[:8])
    more = f" (+{len(inserts) - 8})" if len(inserts) > 8 else ""
    return f"[inserts] {len(inserts)} over {span}: {names}{more}"


if __name__ == "__main__":
    # Plan a script with SYNTHETIC timings — enough to see which nouns get
    # picked and in what order, without a voiceover or a GPU.
    text = " ".join(sys.argv[1:]) or (
        "February 20, 1893, Philadelphia — workers huddled outside the "
        "railroad office as receivers took over. Their jobs vanished "
        "overnight. People hoarded coins, fearing paper money was worthless. "
        "Bank runs followed and the economy crumbled.")
    fake = [{"text": w, "start": i * 0.32, "end": i * 0.32 + 0.3}
            for i, w in enumerate(re.findall(r"[A-Za-z]+", text))]
    got = plan(text, fake)
    print(describe(got))
    for ins in got:
        print(f"  {ins['at']:6.2f}s  {ins['word']:<14} {ins['prompt'][:60]}…")
