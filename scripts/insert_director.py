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
  RUFUS_INSERTS        0 (default) — 1 turns the whole layer on
  RUFUS_INSERT_MAX     most inserts in one video (per-format: 28 / 60)
  RUFUS_INSERT_GAP     0.45 minimum seconds between two inserts
  RUFUS_INSERT_HOLD    0.70 seconds an insert stays on screen
  RUFUS_INSERT_REPEAT  1    times one noun may be shown (raise for density)
  RUFUS_INSERT_MODE    nouns | phrases  (see plan_for)
  RUFUS_PHRASE_WORDS   4    spoken words per picture in phrase mode

HOW MANY PICTURES YOU ACTUALLY GET, and what limits it. Measured on a real
51-word script from this channel: seven drawable nouns. RUFUS_INSERT_MAX is
therefore NOT the cap in normal use — the vocabulary of the script is. Raising
the max alone changes nothing.

Density comes from three places, strongest first:
  1. RUFUS_FRAMES_PER_BEAT — several stills per beat, hard-cut. Ten beats at 4
     frames is forty pictures, and it needs nothing from this module.
  2. RUFUS_INSERT_REPEAT   — show a noun again when it is spoken again.
  3. RUFUS_INSERT_GAP      — how close together two inserts may land.
A script that names concrete things is worth more than all three.
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


def _limit() -> int:
    """The ceiling on inserts for this video. Per-format, because it is a
    DENSITY: 28 across forty seconds is a picture every second and a half, and
    the same 28 across nine minutes is a picture every nineteen — spent
    entirely in the first two minutes, because the candidates arrive in time
    order. See _thin for the other half of that.

    RUFUS_INSERT_MAX still overrides, and still means the same thing.
    """
    # Read directly rather than through _cfg: _cfg needs a numeric default and
    # the default here is "whatever the profile says", which is not a number
    # until the profile has been asked.
    raw = os.environ.get("RUFUS_INSERT_MAX", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            print(f"[inserts] RUFUS_INSERT_MAX={raw!r} is not a number — "
                  f"using the format's own ceiling")
    try:
        import video_format
        return int(video_format.get("insert_max", DEFAULT_MAX))
    except Exception:
        return DEFAULT_MAX


def _repeat() -> int:
    """How many times one noun may get its own picture.

    One is right for forty seconds: a Short says "coin" twice and the second
    picture would land four seconds after the first. Across nine minutes the
    same noun is said a dozen times, minutes apart, and showing it once means
    the through-line the storyboard builds in words has a picture only at its
    first mention.
    """
    try:
        import video_format
        return int(video_format.get("insert_repeat", 1))
    except Exception:
        return 1


def _thin(out: list, limit: int) -> list:
    """At most `limit` inserts, SPREAD rather than truncated.

    The candidates are in time order, so cutting the list at the limit keeps
    the earliest ones — which on a forty-second video is the whole video and on
    a nine-minute one is the first ninety seconds, with seven and a half
    minutes after it carrying none. Nobody would ship that on purpose and
    nothing downstream can see it: every insert in the list is correct.

    Same reasoning as vision_review's frame sample. A Short is under the limit
    in practice, so this changes nothing there.
    """
    if limit <= 0 or len(out) <= limit:
        return out
    step = len(out) / float(limit)
    return [out[min(len(out) - 1, int(i * step))] for i in range(limit)]


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
    # Agent nouns that name a ROLE rather than a look. "receivers" chose itself
    # on the 1893 script — in that script they are bankruptcy receivers, and
    # the model draws a telephone handset. Note the ones that DO have a look
    # (workers, traders, farmers, soldiers, miners) are deliberately absent.
    "receivers", "holders", "owners", "buyers", "sellers", "makers", "users",
    "lenders", "borrowers", "investors", "creditors", "debtors", "members",
    "leaders", "founders", "partners", "clients", "customers",
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
    # "-al" adjectives the tail rules deliberately miss, because bare "al" is
    # unsafe (metal, signal, canal, medal). Listing them one by one is the
    # price of not throwing those away.
    "universal", "general", "central", "personal", "natural", "normal",
    "formal", "legal", "royal", "equal", "annual", "usual", "actual",
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
# "ial"/"ional" catch commercial, financial, universal-adjacent forms that the
# shorter tails miss. Bare "al" is NOT here on purpose: metal, signal, canal and
# medal are all things a picture can be.
_ADJECTIVE_TAILS = ("less", "ful", "ous", "ive", "able", "ible", "ish",
                    "ical", "ary", "ant", "ent", "ial", "ional")

# Number words. A script that opens on "seventy percent" is doing its job; a
# picture of "seventy" is not a picture of anything.
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "percent", "half", "third", "quarter", "double",
    "dozen", "score", "many", "several", "hundreds", "thousands", "millions",
    "billions",
}

# Bare verbs whose base form is spelled like a noun and survives every suffix
# rule. "keep" was chosen from "brokers earned their keep".
_BARE_VERBS = {
    "keep", "earn", "earned", "make", "made", "take", "give", "come", "know",
    "think", "want", "need", "seem", "become", "remain", "stay", "began",
    "merge", "merged", "blur", "blurred", "repeal", "repealed", "change",
    "changed", "raise", "raised", "lower", "reach", "reached", "allow",
    "allowed", "cause", "caused", "force", "forced", "hold", "carry",
}

# Positions and directions. "outside" is a relationship between things, not a
# thing — drawing it means drawing whatever it was outside OF.
_POSITIONAL = {
    "outside", "inside", "above", "below", "across", "beyond", "within",
    "without", "around", "behind", "beneath", "toward", "towards", "upward",
    "downward", "along", "amid", "among", "between", "beside", "near", "past",
    "through", "under", "upon", "ahead", "apart", "away", "back", "forward",
}

# Degree and stance adverbs. They survive every suffix rule here — "quite" and
# "grey" both reached a live report of things no shot showed — because they end
# in nothing distinctive and are not verbs. There is no picture of "quite".
_ADVERBS = {
    "quite", "very", "rather", "almost", "nearly", "barely", "hardly",
    "simply", "merely", "truly", "really", "surely", "clearly", "exactly",
    "already", "still", "always", "never", "often", "sometimes", "usually",
    "again", "instead", "however", "therefore", "meanwhile", "suddenly",
    "finally", "eventually", "overnight", "today", "tomorrow", "yesterday",
    "enough", "even", "just", "only", "much", "quickly", "slowly",
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
    """Whether the insert layer runs at all. OFF unless asked for.

    IT USED TO DEFAULT ON, and that is how a run came back with "images on
    top of images": twenty-eight full-frame beat pictures with fourteen more
    overlaid on them. The owner had been switching it off by hand with
    `$env:RUFUS_INSERTS="0"` in every terminal, so the default was never
    exercised until they stopped typing it — which is exactly the moment a
    wrong default surfaces.

    An insert is a SECOND set of pictures painted over a video that is already
    finished. That is a deliberate style, not a sensible default, and the beat
    count now derives from the script so the base video is dense on its own.
    RUFUS_INSERTS=1 turns it on.
    """
    return os.environ.get("RUFUS_INSERTS", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def style_suffix() -> str:
    """The channel's own look, so an insert belongs to the beat behind it.

    Reusing comfy_client's detail suffix rather than writing a second style
    string is the same rule the world lock follows: two descriptions of one
    look drift, and the drift shows up as an insert that is visibly from a
    different video. Lives here rather than in a renderer because BOTH
    renderers need it and the second copy is where the drift starts.
    """
    try:
        import comfy_client
        # AN INSERT IS AN OBJECT BY DEFINITION — one thing, cut out, on a plain
        # backdrop, with nobody in it. Handing it the figure half of the style
        # block put a paragraph about arms, legs and where the head sits
        # directly after "a single sword, centred in frame", which is the same
        # mismatch that puts a stick figure next to a banana.
        return comfy_client._detail_for_shot("object")
    except Exception:
        return ""


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
    if word in _ADVERBS:
        return False
    if word in _PLAIN_ADJECTIVES or word in _NUMBER_WORDS or word in _BARE_VERBS:
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
    limit = _limit()
    gap = _cfg("RUFUS_INSERT_GAP", DEFAULT_GAP)
    hold = _cfg("RUFUS_INSERT_HOLD", DEFAULT_HOLD)

    spoken = _word_times(words)
    # REPEATS ARE A DENSITY KNOB. One picture per noun caps a 40-second video
    # at however many distinct drawable nouns the script happens to contain —
    # measured on a real script, ten out of fifty-one words. Showing the same
    # coin again when the narration says "coin" again is not a duplicate; it is
    # the through-line the storyboard already tries to build in words.
    per_word = max(1, int(_cfg("RUFUS_INSERT_REPEAT", _repeat())))
    candidates: list[tuple[float, str]] = []
    for noun in insert_words(script):
        # SPREAD, not the first N. Taking occurrences in order is right when
        # the whole video is forty seconds and every occurrence is "early". Over
        # nine minutes a common noun is said a dozen times, and the first three
        # are all in the opening minute — so the noun's pictures would cluster
        # there and the second half would carry none of it. Same thinning as
        # the ceiling below, one noun at a time.
        for at in _thin(spoken.get(noun, []), per_word):
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
    return _thin(out, limit)


def _with_style(base: str, style: str) -> str:
    """Base prompt + the channel style, with the insert's own framing LAST.

    An insert composites OVER a beat clip that already draws the place, so it
    has to be a cutout — a second background behind a 460px picture in the
    corner just makes two competing scenes in one frame. The channel's style
    block now asks for a background (it must: the beats looked like figures
    floating in white without one), so the override has to come after it.
    Prompts are read most-recent-wins by every model this pipeline drives.
    """
    if not (style and style.strip()):
        return base
    return (f"{base}. {style.strip()} For THIS picture only: no background "
            f"scenery, no room, no horizon — the subject alone on a plain "
            f"flat backdrop, cut out and ready to place over another image.")


def insert_prompt(noun: str, style: str = "") -> str:
    """What to draw for one insert.

    Deliberately spare. An insert is on screen for under a second at a fraction
    of frame size, so detail is wasted GPU: what has to read instantly is the
    SILHOUETTE. "A single X, centred, plain background" gives the model nothing
    to clutter with, and the channel's own style suffix does the rest so an
    insert belongs to the same world as the beat behind it.
    """
    base = (f"A single {_singular(noun)}, centred in frame, one clear object, "
            f"plain flat background, bold readable silhouette, no text, "
            f"no words, no letters")
    return _with_style(base, style)


# Words whose singular is not the word minus an "s", or which are not plural at
# all despite ending in one. Stripping blindly gave "pres" and "gla".
_NOT_PLURAL = {
    "press", "glass", "class", "grass", "cross", "dress", "brass", "mass",
    "paper", "canvas", "compass", "harness", "witness", "business", "access",
    "news", "series", "species", "gas", "bus", "lens", "axis", "basis",
    "crisis", "ships",  # "ships" is plural but a fleet reads better than one
}
_IRREGULAR = {
    "men": "man", "women": "woman", "children": "child", "feet": "foot",
    "teeth": "tooth", "geese": "goose", "mice": "mouse", "people": "person",
    "leaves": "leaf", "knives": "knife", "wolves": "wolf", "shelves": "shelf",
    "loaves": "loaf", "thieves": "thief", "wives": "wife", "lives": "life",
}


def _singular(noun: str) -> str:
    """One of the thing, because an insert should show ONE of the thing.

    "A single coins" is what the first version produced, and it is wrong twice
    over: ungrammatical to the text encoder, and visually wrong at 0.42 of
    frame width, where one clear object reads and a pile does not.

    Deliberately conservative. A word this cannot confidently singularise is
    returned unchanged — a plural picture is a small loss, and "pres" from
    "press" is a wrong one.
    """
    low = (noun or "").strip().lower()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    if low in _NOT_PLURAL or len(low) < 4 or not low.endswith("s"):
        return low
    if low.endswith(("ss", "us", "is")):
        return low
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"          # cities -> city
    if low.endswith(("ches", "shes", "xes", "zes", "sses")):
        return low[:-2]                # torches -> torch
    return low[:-1]


def plan_for(script: str, words: list[dict], style: str = "") -> list[dict]:
    """The plan for whichever mode is selected. One entry point for callers.

    nouns   (default) — a picture per drawable NOUN. Sparse, specific, and
                        capped by the script's vocabulary: seven on a real
                        51-word script here.
    phrases           — a picture per CLAUSE, tiling the whole narration. Capped
                        only by the length of the script, so the same 51 words
                        give eleven, each holding until the next one starts.

    Phrase mode is what "an image per word" actually wants: noun inserts cannot
    get denser than the nouns present, and RUFUS_FRAMES_PER_BEAT gets denser
    without getting more SPECIFIC — its extra frames are variations of one beat
    prompt cut at even intervals, matched to no word at all.
    """
    mode = os.environ.get("RUFUS_INSERT_MODE", "nouns").strip().lower()
    if mode in ("phrase", "phrases"):
        return plan_phrases(script, words, style)
    if mode not in ("noun", "nouns", ""):
        print(f"[inserts] RUFUS_INSERT_MODE={mode!r} unknown — using 'nouns'. "
              f"Known: nouns, phrases")
    return plan(script, words, style)


def phrase_prompt(text: str, style: str = "") -> str:
    """What to draw for one PHRASE of narration.

    Different job from insert_prompt. That one draws a named object on a plain
    background; this draws the little SCENE the phrase describes, because a
    phrase is the unit the viewer is hearing — "brokers earned their keep" is
    not a broker, it is people at work.

    The narration is handed over almost verbatim on purpose. Rewriting it into
    a visual description is a second interpretation of something the storyboard
    already interprets, and every paraphrase is a chance to drift away from the
    words being spoken at that exact second — which is the one thing this whole
    layer exists to nail.
    """
    clean = " ".join((text or "").split()).strip(" ,.;:!?-—")
    base = (f"A simple scene showing: {clean}. One clear subject, "
            f"uncluttered, bold readable silhouette, no text, no words, "
            f"no letters")
    return _with_style(base, style)


def plan_phrases(script: str, words: list[dict], style: str = "") -> list[dict]:
    """One picture per PHRASE, each landing on the second that phrase begins.

    WHY THIS EXISTS ALONGSIDE plan(). Noun inserts are limited by the script's
    vocabulary — measured on a real 51-word script from this channel, seven
    drawable nouns, and no setting raises that. Phrases are limited only by the
    length of the narration, so a hundred-word script yields twenty-five
    pictures at four words each, every one of them tied to a real timestamp.

    And it fixes the thing RUFUS_FRAMES_PER_BEAT does not: extra beat frames
    are generic variations of one prompt cut at even intervals, matched to no
    word at all. These are matched to the words by construction.

    Env:
      RUFUS_PHRASE_WORDS   4  spoken words per picture
    """
    if not enabled():
        return []
    per = max(1, int(_cfg("RUFUS_PHRASE_WORDS", 4)))
    limit = _limit()
    hold = _cfg("RUFUS_INSERT_HOLD", DEFAULT_HOLD)

    timed = []
    for w in words or []:
        text = str(w.get("text", "")).strip()
        try:
            start = float(w.get("start", 0.0))
        except (TypeError, ValueError):
            continue
        if text:
            timed.append((start, text))
    if not timed:
        return []
    timed.sort()

    # BREAK ON CLAUSES, NOT ON A COUNTER. Fixed-size chunks cut straight
    # through a thought — "In a bustling Wall" / "Street office Merrill Lynch"
    # — and the picture for half a phrase is a picture of nothing. Whisper
    # keeps the punctuation, so a comma or a full stop is a free clause
    # boundary; `per` becomes the maximum length rather than the only rule.
    groups: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    for start, text in timed:
        current.append((start, text))
        ends_clause = text.rstrip().endswith((",", ".", "!", "?", ";", ":", "—"))
        if len(current) >= per or (ends_clause and len(current) >= 2):
            groups.append(current)
            current = []
    if current:
        # A trailing fragment shorter than two words reads as a flash; fold it
        # into the phrase before it rather than giving it its own picture.
        if len(current) == 1 and groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    # THINNED BEFORE THE HOLDS ARE COMPUTED, not after. Each picture holds
    # until the next one starts, which is what makes them TILE the narration —
    # thinning afterwards would leave every surviving picture with the hold of
    # a neighbour that is no longer there, and the tiling would become a
    # flicker with gaps.
    groups = _thin(groups, limit)

    out: list[dict] = []
    for gi, chunk in enumerate(groups):
        at = round(chunk[0][0], 3)
        phrase = " ".join(t for _, t in chunk)
        # Hold until the next phrase starts, so the pictures TILE the narration
        # instead of flashing and leaving gaps. The last one keeps the default.
        nxt = groups[gi + 1][0][0] if gi + 1 < len(groups) else None
        span = round(nxt - at, 3) if nxt is not None else hold
        anchor = next((t.lower() for _, t in chunk
                       if _is_drawable(re.sub(r"[^a-z]", "", t.lower()))),
                      chunk[0][1].lower())
        out.append({
            "word": re.sub(r"[^a-z]", "", anchor) or "phrase",
            "at": at,
            "hold": max(0.25, span),
            "prompt": phrase_prompt(phrase, style),
            "phrase": phrase,
        })
    return out


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
