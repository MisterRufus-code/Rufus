#!/usr/bin/env python3
"""
run_review.py — measure a finished run, so the next one can be better.

WHY THIS EXISTS. Every improvement to this pipeline in recent memory came from
the owner watching a video, noticing something, and pasting a log. That works,
and it does not scale: it needs a person to watch, to remember what the last
six runs looked like, and to be right about which of the twenty things on
screen is the one that matters. The evidence was always there — the debug
folder holds the script, every beat prompt and every keyframe — but nobody was
reading it.

Three real defects found that way, all of which this module now measures
directly:

  "why all the images with coin"      one object named in 9 of 10 prompts
  "restated the location in 13 shots" one clause on half the sequence
  "QC ⚠ 5 stretches over 5s"          a picture held for six seconds

Every one is a number this can compute from files already on disk, without a
model, without a GPU and without anyone watching. That is the whole design:
DETERMINISTIC FIRST. A measurement that is free and exact runs on every run; a
vision model's opinion is a separate, optional pass (see RUFUS_REVIEW), because
GPU time on this box is the scarcest thing there is and a run already takes
twenty-five minutes.

CONTRACT: read-only and non-fatal. This module reads a finished run and writes
one review.json beside it. It never touches the video, never re-renders, and
every failure returns a partial result rather than raising — a broken reviewer
must not turn a good run into an error.

    python scripts/run_review.py                # the most recent run
    python scripts/run_review.py <run_id>       # one specific run
    python scripts/run_review.py --all          # every run, then the patterns
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402

REVIEW_FILE = "review.json"

# The clauses this pipeline appends to prompts, and what too many of each means.
# Both were added for a real reason and both went wrong the same way — by
# appearing so often that they stopped being a note and became the subject.
THREAD_MARK = "Continuing from the previous shot:"
SETTING_MARK = "Same place as the rest of the sequence:"
BLANK_MARK = "blank and unmarked"

# A held picture is the one defect a viewer feels without being able to name —
# but only when nothing chose it.
#
# 3.5s was right when every shot was the same length and a long one meant the
# planner had run out of pauses. It is wrong now: the cut planner weights each
# beat's share by its tone, so a revelation deliberately runs to about 4.4s on
# a 38-second video and a resolution to 3.4s. Flagging those would be the
# measurement contradicting the feature, which is the same mistake as counting
# a beat's own sub-frames as duplicates.
#
# 5s is the line QC already draws, and it is the point past which a hold stops
# reading as emphasis and starts reading as a stall — for a Short. Both modules
# held the number 5.0 with a comment each saying it matched the other, which is
# a hand-copy that happened to be true and would not have survived a format
# whose ordinary shot is 3.6 seconds long. One source now.
import video_format as _vf
LONG_HOLD_S = float(_vf.get("max_hold_s", 5.0))

# Below this many pictures there is no pattern to find — one prompt is
# trivially 100% of one prompt.
MIN_PROMPTS_FOR_DOMINANCE = 5

# Above this share of the prompts, one object is no longer a through-line — it
# is what the video is about. Measured from the run the owner complained about:
# a stack of coins in 9 of 10.
DOMINANT_SHARE = 0.55

# Words that appear in most prompts because of the STYLE block or the
# storyboard's own vocabulary, not because the video is about them. Counting
# these would make every run look like it was about "frame" and "shot".
_PROMPT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "with",
    "from", "for", "is", "are", "as", "by", "his", "her", "its", "their",
    "shot", "close", "wide", "mid", "frame", "scene", "view", "background",
    "foreground", "left", "right", "same", "previous", "continuing", "place",
    "rest", "sequence", "one", "two", "three", "another", "over", "under",
    "into", "out", "up", "down", "now", "then", "while", "as", "this", "that",
    "figure", "figures", "person", "people", "man", "woman", "hand", "hands",
    "face", "faces", "line", "art", "black", "white", "flat", "colour",
    "color", "thin", "bold", "simple", "small", "large", "empty", "front",
}


def _run_dir(run_id: str) -> Path:
    return paths.debug_root() / run_id


def latest_run_id() -> str:
    """The most recently modified run folder, or "" if there are none."""
    try:
        runs = [d for d in paths.debug_root().iterdir() if d.is_dir()]
    except OSError:
        return ""
    if not runs:
        return ""
    return max(runs, key=lambda d: d.stat().st_mtime).name


def all_run_ids(limit: int | None = 60) -> list[str]:
    """Run folders, newest first. `limit=None` means every one of them.

    The cap exists so the dashboard's pages stay quick on a machine with
    hundreds of runs. It does NOT belong on `--all`, which measured sixty of
    the owner's eighty-six and said nothing about the other twenty-six — a
    silent cap, which is the one thing this repo treats as always a bug.
    """
    try:
        runs = [d for d in paths.debug_root().iterdir() if d.is_dir()]
    except OSError:
        return []
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in (runs if limit is None else runs[:limit])]


def _read_prompts(d: Path) -> list[str]:
    """The beat prompts, from run_report.md.

    Parsed out of the report rather than re-derived, because the report is what
    was actually SENT — after the thread clause, the setting clause, the
    readable-text defusal and the style suffix were all appended. Measuring the
    storyboard's own output instead would miss every defect this module exists
    to catch, since all of them are things appended later.
    """
    report = d / "run_report.md"
    try:
        text = report.read_text(encoding="utf-8")
    except OSError:
        return []
    # Beat blocks are fenced: **Beat N**\n\n```\n<prompt>\n```
    return [m.group(1).strip() for m in
            re.finditer(r"\*\*Beat \d+\*\*\s*\n+```\n(.*?)\n```", text, re.S)]


def _read_script(d: Path) -> str:
    for name in ("script.txt",):
        try:
            return (d / name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def _keyframes(d: Path) -> list[Path]:
    """ONE frame per beat, not every frame on disk.

    In cut and i2i modes a beat is saved as 07.png, 07a.png, 07b.png — the same
    scene a moment apart, rendered on one seed, and near-identical BY DESIGN.
    Counting them as duplicates reported 73 near-identical pairs on a run whose
    whole point was that the shot advances inside the beat, which is the
    measurement contradicting the feature. Comparing the base frame of each
    beat asks the question that was meant: do two different BEATS look the
    same.
    """
    base = [p for p in d.glob("*.png")
            if p.is_file() and re.fullmatch(r"\d+", p.stem)]
    if base:
        return sorted(base, key=lambda p: int(p.stem))
    return sorted(p for p in d.glob("*.png") if p.is_file())


# The framing phrases storyboard._FRAMINGS puts at the FRONT of every prompt.
# Matched here rather than imported so this module keeps its contract of
# reading a finished run with nothing else loaded — and asserted equal in
# tests, because a copy that drifts is the failure this repo keeps meeting.
_FRAMING_MARKS = {
    "wide": "Wide shot:",
    "mid": "Medium shot:",
    "close": "Close shot:",
    "detail": "Close detail:",
}


def _framing_counts(prompts: list[str]) -> dict:
    """Which distances the sequence used, and whether it ever moved.

    A sequence at one distance for its whole length is invisible in the text —
    every prompt differs, every subject differs — and unmistakable on screen.
    That is the same shape as the face defect, and the same reason it needed
    counting rather than noticing.
    """
    counts = {k: sum(1 for p in prompts if p.lstrip().startswith(v))
              for k, v in _FRAMING_MARKS.items()}
    given = sum(counts.values())
    return {"counts": counts, "given": given,
            "distinct": sum(1 for v in counts.values() if v),
            "share": round(given / len(prompts), 3) if prompts else 0.0}


def _clause_counts(prompts: list[str]) -> dict:
    n = len(prompts) or 1
    thread = sum(1 for p in prompts if THREAD_MARK in p)
    setting = sum(1 for p in prompts if SETTING_MARK in p)
    blank = sum(1 for p in prompts if BLANK_MARK in p)
    return {
        "thread_clause": thread,
        "thread_share": round(thread / n, 3),
        "setting_clause": setting,
        "setting_share": round(setting / n, 3),
        "blank_surface_clause": blank,
        "blank_surface_share": round(blank / n, 3),
    }


def _common_suffix(prompts: list[str]) -> str:
    """The longest tail every prompt in the run shares.

    THIS IS THE STYLE BLOCK, exactly, and finding it this way rather than by
    matching its opening words is what makes the measurement survive a style
    nobody has written yet. The first version split on the three style names
    this repo ships, and the moment it met an older run using a niche's own
    `style_suffix` the whole block leaked into the word counts — which is why a
    real pass over fifty runs reported that nine of nine prompts were about
    "lens" (from "no lens blur"), ten of ten about "hairstyles", and nine of
    nine about "modern". Three confident findings, all of them a description of
    the style suffix.

    Every prompt in a run ends with the identical appended block, so the
    longest common suffix is that block and nothing else.
    """
    if len(prompts) < 2:
        return ""
    shortest = min(len(p) for p in prompts)
    i = 0
    while i < shortest and len({p[-(i + 1)] for p in prompts}) == 1:
        i += 1
    tail = prompts[0][len(prompts[0]) - i:] if i else ""
    # Only trust it if it is long enough to be a style block rather than a
    # coincidence of shared punctuation.
    return tail if len(tail) > 80 else ""


def _subject_words(prompt: str, common: str = "") -> set[str]:
    """The content words of the BEAT half of a prompt."""
    head = prompt[:len(prompt) - len(common)] if common and prompt.endswith(common) else prompt
    head = re.split(r"Minimalist stick-figure|Flat 2D vector|Monochrome ink",
                    head, maxsplit=1)[0]
    # Drop the appended clauses WHOLE, not just their markers. Stripping only
    # the marker left "market, cobblestone, bright" behind on thirteen prompts
    # and this metric duly reported that the video was about a market — which
    # it partly was, but the count came from a clause the pipeline wrote, not
    # from a picture the storyboard chose. The clause counts above measure the
    # appended text; this measures what the storyboard itself asked for.
    for mark in (THREAD_MARK, SETTING_MARK):
        head = re.sub(re.escape(mark) + r"[^.]*\.?", " ", head)
    head = head.split(BLANK_MARK)[0]
    words = re.findall(r"[a-z]{3,}", head.lower())
    return {w for w in words if w not in _PROMPT_STOPWORDS}


def _dominant_subject(prompts: list[str]) -> dict:
    """The object that appears in the most prompts, and in how many.

    THE COIN METRIC. "why all the images with coin" was a real report about a
    real run, and this is that complaint as a number: one word present in 9 of
    10 prompts. Counted per PROMPT rather than per occurrence, because a word
    repeated three times in one shot is one picture of it.
    """
    if not prompts:
        return {"word": "", "prompts": 0, "share": 0.0}
    common = _common_suffix(prompts)
    counts: Counter = Counter()
    for p in prompts:
        counts.update(_subject_words(p, common))
    if not counts:
        return {"word": "", "prompts": 0, "share": 0.0}
    word, hits = counts.most_common(1)[0]
    return {"word": word, "prompts": hits, "share": round(hits / len(prompts), 3)}


def _cut_metrics(video_path: Path | None) -> dict:
    """Cut spacing, from the QC sidecar the renderer already writes."""
    out = {"cuts": 0, "longest_hold_s": None, "long_holds": 0, "duration_s": None}
    if video_path is None:
        return out
    qc_file = Path(str(video_path) + ".qc.json")
    try:
        qc = json.loads(qc_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return out
    out["duration_s"] = qc.get("duration")
    cuts = qc.get("cuts") or []
    out["cuts"] = len(cuts)
    marks = [0.0] + [float(c) for c in cuts]
    if out["duration_s"]:
        marks.append(float(out["duration_s"]))
    gaps = [b - a for a, b in zip(marks, marks[1:]) if b > a]
    if gaps:
        out["longest_hold_s"] = round(max(gaps), 2)
        out["long_holds"] = sum(1 for g in gaps if g > LONG_HOLD_S)
        out["mean_hold_s"] = round(sum(gaps) / len(gaps), 2)
    return out


def _near_duplicates(frames: list[Path]) -> dict:
    """How many keyframes are near-copies of another.

    An average-hash rather than a byte hash: the pipeline's own freshness gate
    already rejects identical bytes, so the images that survive to here and
    still look the same are the ones a difference in a single pixel let
    through. Pillow only — no new dependency for a diagnostic.
    """
    out = {"frames": len(frames), "near_duplicate_pairs": 0,
           "duplicated_frames": 0, "duplicate_share": 0.0}
    if len(frames) < 2:
        return out
    try:
        from PIL import Image
    except ImportError:
        return out

    hashes = []
    for f in frames:
        try:
            with Image.open(f) as im:
                small = im.convert("L").resize((8, 8))
                px = list(small.getdata())
        except Exception:
            continue
        avg = sum(px) / len(px)
        hashes.append((f.name, sum(1 << i for i, v in enumerate(px) if v > avg)))

    # HOW MANY FRAMES have a twin, not how many PAIRS exist. Pairs grow with
    # the square of the sequence, so "73 pairs" says nothing a reader can
    # picture — five identical frames among forty is 10 pairs and five frames
    # among fifteen is also 10, and those are very different videos.
    pairs = 0
    twinned = set()
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i][1] ^ hashes[j][1]).count("1") <= 5:
                pairs += 1
                twinned.add(i)
                twinned.add(j)
    out["near_duplicate_pairs"] = pairs
    out["duplicated_frames"] = len(twinned)
    out["duplicate_share"] = round(len(twinned) / len(hashes), 3) if hashes else 0.0
    return out


# Verbs that describe a body at rest. A shot built on one of these and nothing
# else is a portrait: the figure exists, and nothing is happening to it.
_STATIC_VERBS = {
    "stand", "stands", "standing", "sit", "sits", "sitting", "look", "looks",
    "looking", "watch", "watches", "watching", "face", "faces", "facing",
    "gaze", "gazes", "gazing", "stare", "stares", "staring", "wait", "waits",
    "waiting", "hold", "holds", "holding", "wear", "wears", "wearing",
    "appear", "appears", "seem", "seems", "pose", "poses", "posing",
    "gather", "gathers", "gathering", "surround", "surrounds", "surrounding",
}

# Physical things that can be caught mid-happening. Deliberately a stem list —
# matched with an optional s/es/ed/ing tail — and deliberately incomplete: the
# finding below is about a SHARE of the sequence, so a verb this misses costs
# a little sensitivity and never a false alarm on its own.
_ACTION_STEMS = (
    "scatter", "pour", "spill", "slam", "kick", "drag", "tip", "topple",
    "overturn", "throw", "hurl", "smash", "shatter", "break", "snap", "tear",
    "rip", "cut", "chop", "burn", "flood", "crash", "collapse", "crack",
    "snatch", "grab", "shove", "push", "pull", "swing", "strike", "hammer",
    "dig", "lift", "haul", "carry", "drop", "fall", "run", "flee", "chase",
    "climb", "leap", "jump", "stumble", "reach", "hand", "pass", "toss",
    "pile", "stack", "sweep", "spray", "splash", "pound", "knock", "slice",
    "count", "weigh", "stamp", "seal", "pry", "wrench", "yank", "bolt",
    "scramble", "clutch", "shield", "duck", "recoil", "flinch", "point",
)
_ACTION_RE = re.compile(
    r"\b(?:" + "|".join(_ACTION_STEMS) + r")(?:s|es|ed|ing)?\b", re.I)

# Below this share of shots carrying an action, the sequence is a gallery of
# portraits. High on purpose: a sequence has every right to hold on a still
# object, and a check that fires on a good run is the noise this repo has
# twice had to walk back. A real run measured 3 of 16.
_ACTION_SHARE_MIN = 0.4


def _action_share(prompts: list[str]) -> float:
    """Share of shots with something physically happening in them."""
    if not prompts:
        return 1.0
    acting = sum(1 for p in prompts if _ACTION_RE.search(p or ""))
    return round(acting / len(prompts), 3)


def _findings(m: dict) -> list[dict]:
    """The measurements that are out of range, phrased as what to do.

    A number nobody acts on is the same as no number. Each finding names the
    lever, because "setting_share 0.54" is a fact and "the setting clause is on
    half the shots — SETTING_SHARE caps it" is a decision.
    """
    out = []
    c = m.get("clauses", {})
    if c.get("setting_share", 0) > 0.35:
        out.append({
            "id": "setting_clause_everywhere",
            "severity": "high",
            "text": (f"The location was restated in {c['setting_clause']} of "
                     f"{m['beats']} shots. Past a third it stops being a "
                     f"reminder and becomes a second description competing "
                     f"with each shot's own — storyboard.SETTING_SHARE caps it."),
        })
    if c.get("thread_share", 0) > 0.45:
        out.append({
            "id": "thread_everywhere",
            "severity": "high",
            "text": (f"The through-line was restated in {c['thread_clause']} of "
                     f"{m['beats']} shots. A thread named everywhere stops "
                     f"connecting the shots and starts repeating them."),
        })
    d = m.get("dominant_subject", {})
    # A short sequence cannot evidence a pattern: one prompt is trivially 100%
    # of one prompt, and a live pass reported "numbers appears in 1 of 1
    # prompts" as a dominance failure on a run that produced one picture.
    if (d.get("share", 0) >= DOMINANT_SHARE and d.get("word")
            and m.get("beats", 0) >= MIN_PROMPTS_FOR_DOMINANCE):
        out.append({
            "id": "one_object_dominates",
            "severity": "high",
            "text": (f'"{d["word"]}" appears in {d["prompts"]} of {m["beats"]} '
                     f"prompts. One object in most of the frames is the "
                     f'"why is everything coins" failure.'),
        })
    cuts = m.get("cuts", {})
    if cuts.get("long_holds"):
        out.append({
            "id": "pictures_held_too_long",
            "severity": "medium",
            "text": (f"{cuts['long_holds']} picture(s) held longer than "
                     f"{LONG_HOLD_S}s, the longest {cuts.get('longest_hold_s')}s. "
                     f"More beats (SD_CLIPS) is the lever; extra frames per "
                     f"beat only re-render the same description."),
        })
    dup = m.get("frames", {})
    if dup.get("duplicate_share", 0) >= 0.3 and dup.get("frames", 0) >= 5:
        out.append({
            "id": "repeated_images",
            "severity": "medium",
            "text": (f"{dup['duplicated_frames']} of {dup['frames']} beat "
                     f"images have a near-identical twin in another beat. The "
                     f"freshness gate only rejects identical bytes, so these "
                     f"got through by a pixel."),
        })
    if c.get("blank_surface_share", 0) > 0.4:
        out.append({
            "id": "text_props_everywhere",
            "severity": "low",
            "text": (f"{c['blank_surface_clause']} of {m['beats']} prompts "
                     f"needed the blank-surfaces defusal, which means the "
                     f"storyboard keeps reaching for signs, screens and "
                     f"documents. They are the hardest thing for an image "
                     f"model to draw without garbling."),
        })
    # MEASURED AGAINST WHAT THE SCRIPT DESERVES, not against a fixed number.
    # The absolute version fired on 48 of 50 real runs — every one of them
    # predates the adaptive beat count, so "only 10 pictures" was one fact
    # about the past repeated forty-eight times. By this module's own standard
    # that is not a finding, it is noise, and noise is what people learn to
    # scroll past.
    fr = m.get("framing") or {}
    if fr.get("given", 0) >= 4 and fr.get("distinct") == 1:
        only = next(k for k, v in fr["counts"].items() if v)
        out.append({
            "id": "one_distance_everywhere",
            "severity": "medium",
            "text": (f"Every one of {fr['given']} shots is framed '{only}'. "
                     f"A sequence that never changes distance reads as a "
                     f"slideshow however good the drawings are — and it is "
                     f"invisible in the prompts, because every subject "
                     f"differs."),
        })

    # THE COMPLAINT THAT NOTHING COULD COUNT. "the pictures are not
    # interesting" was the owner opening a folder of sixteen and seeing
    # thirteen shots of figures standing upright facing the viewer. Every one
    # of them was a correct drawing of its line, so nothing measured from the
    # text could object — the prompts differ, the subjects differ, the framing
    # varies. What they share is that nothing is happening in any of them.
    share = m.get("action_share")
    if share is not None and m.get("beats", 0) >= 6 and share < _ACTION_SHARE_MIN:
        acting = round(share * m["beats"])
        out.append({
            "id": "nothing_is_happening",
            "severity": "high",
            "text": (f"Only {acting} of {m['beats']} shots have anything "
                     f"physically happening in them — the rest are people and "
                     f"objects at rest. A sequence of portraits is the "
                     f"difference between a video people watch and one they "
                     f"scroll past, and it is invisible in every other "
                     f"measurement here because each shot is a correct drawing "
                     f"of its own line. The storyboard brief asks for the VERB "
                     f"first; this is what it looks like when that is not "
                     f"landing."),
        })

    want = _deserved_beats(m.get("script_words", 0))
    if m.get("beats") and want and m["beats"] < want * 0.7:
        out.append({
            "id": "few_pictures",
            "severity": "low",
            "text": (f"{m['beats']} pictures for a {m['script_words']}-word "
                     f"script. The current beat rule would give about {want} "
                     f"— roughly one per {_vf.get('words_per_picture', 5)} "
                     f"spoken words. SD_CLIPS overrides it."),
        })
    return out


def _deserved_beats(words: int) -> int:
    """What main._target_beats would choose for a script this long.

    NO LONGER A COPY. It was one, deliberately — main.py pulls in the whole
    pipeline and this module reads finished runs with nothing else loaded — and
    the copy cost exactly what a copy costs: main moved from four words to five
    the same day the cut rhythm was fixed, this did not, and for a while the
    analyzer reported runs as short of a target the pipeline had stopped aiming
    at. The measurement contradicting the feature, which is the failure this
    module exists to catch.

    A second format would have done it again, in the quieter direction: the
    Shorts constants (floor 10, ceiling 30, one per five words) applied to a
    1,350-word script cap the answer at 30, so a nine-minute run that rendered
    twenty-four pictures — the SD_CLIPS default, one picture every twenty-two
    seconds — would have been measured as generous. The analyzer would have
    been blind precisely where the defect is likeliest.

    video_format is the rule itself and imports nothing but os, so reading it
    costs none of what importing main would.
    """
    if not words:
        return 0
    return _vf.target_beats(words)


def review(run_id: str, video_path: Path | None = None) -> dict:
    """Measure one run. Always returns a dict; never raises."""
    d = _run_dir(run_id)
    prompts = _read_prompts(d)
    script = _read_script(d)
    frames = _keyframes(d)

    m = {
        "run_id": run_id,
        "beats": len(prompts),
        "script_words": len(script.split()) if script else 0,
        "clauses": _clause_counts(prompts),
        "dominant_subject": _dominant_subject(prompts),
        "framing": _framing_counts(prompts),
        "action_share": _action_share(prompts),
        "cuts": _cut_metrics(video_path),
        "frames": _near_duplicates(frames),
    }
    m["findings"] = _findings(m)

    # AND THEN LOOK AT THEM. Everything above is measured from text — the
    # prompts, the script, the QC sidecar — and every image defect this
    # channel has actually suffered was found by the owner opening the
    # gallery instead. Off by default (it costs seconds a frame); when it is
    # on, its findings join the rest so one page answers "what is wrong with
    # this run" rather than two.
    try:
        import vision_review
        if vision_review.enabled():
            seen = vision_review.review_frames(frames, prompts)
            if seen.get("looked_at"):
                m["vision"] = seen
                m["findings"].extend(seen.get("findings", []))
    except Exception as e:                      # never fatal, always audible
        print(f"[review] picture review skipped (non-fatal): {e}")
    return m


def save(run_id: str, data: dict) -> Path | None:
    """Write review.json beside the run's other artefacts. None on failure."""
    try:
        d = _run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        out = d / REVIEW_FILE
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out
    except OSError as e:
        print(f"[review] could not write review for {run_id}: {e}")
        return None


def load(run_id: str) -> dict | None:
    try:
        return json.loads((_run_dir(run_id) / REVIEW_FILE)
                          .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def review_and_save(run_id: str | None = None,
                    video_path: Path | None = None) -> dict:
    """The entry point a run calls. Fail-open by contract."""
    try:
        rid = run_id or os.environ.get("RUFUS_DEBUG_RUN_ID") or latest_run_id()
        if not rid:
            return {}
        data = review(rid, video_path)
        save(rid, data)
        n = len(data.get("findings", []))
        if n:
            print(f"[review] {n} finding(s) — see the Insights page")
            for f in data["findings"][:3]:
                print(f"[review]   {f['text']}")
        else:
            print("[review] nothing out of range")
        return data
    except Exception as e:                      # never fatal
        print(f"[review] skipped (non-fatal): {e}")
        return {}


def patterns(limit: int = 20) -> dict:
    """What keeps happening, across runs.

    THE POINT OF KEEPING THESE. One run's numbers say a video was weak. Twenty
    runs' numbers say which defect is systematic, and a defect that shows up in
    four of the last six runs is a code change rather than a bad seed.
    """
    rows = []
    for rid in all_run_ids(limit):
        data = load(rid)
        if data:
            rows.append(data)
    counts: Counter = Counter()
    for r in rows:
        for f in r.get("findings", []):
            counts[f["id"]] += 1
    return {
        "runs_reviewed": len(rows),
        "recurring": [{"id": k, "runs": v, "share": round(v / len(rows), 2)}
                      for k, v in counts.most_common()] if rows else [],
        "rows": rows,
    }


def _gpu_is_busy() -> str:
    """The channel whose render currently holds the GPU, or "".

    ONLY THE COMMAND LINE ASKS. Inside a run, main.py itself holds this lock,
    so checking there would skip the picture review on every single run — the
    guard would prevent exactly the thing it exists to enable. What this
    protects against is the other case: somebody running `run_review.py --all`
    from a prompt while a render is going, loading a 7B vision model onto a
    3090 that ComfyUI is already using and taking both down with an
    out-of-memory error.
    """
    try:
        import channel_config
        from filelock import FileLock, Timeout
        root = Path(__file__).parent.parent
        for cid in channel_config.list_channels():
            # The exact name main.py's _acquire_lock() uses — same lock,
            # checked non-blockingly rather than held.
            lock = FileLock(str(root / f"rufus.{cid}.lock.lock"))
            try:
                lock.acquire(timeout=0)
            except Timeout:
                return cid
            except Exception:
                continue
            lock.release()
    except Exception:
        pass
    return ""


def main() -> None:
    # The text half is free and runs whatever else is happening. The picture
    # half wants the GPU, and a render is already using it.
    import vision_review
    if vision_review.enabled():
        busy = _gpu_is_busy()
        if busy:
            print(f"[review] a run is rendering ({busy}) — measuring the text "
                  f"only. The picture review needs the GPU that run is using; "
                  f"re-run this when it finishes.")
            os.environ["RUFUS_VISION"] = "0"

    args = [a for a in sys.argv[1:] if a != "--all"]
    if "--all" in sys.argv[1:]:
        ids = all_run_ids(limit=None)
        for rid in ids:
            review_and_save(rid)
        # Aggregate over what was just measured, not over a smaller default —
        # the first pass reviewed fifty-five runs and then reported patterns
        # across twenty, which reads as most of the work having been discarded.
        p = patterns(limit=len(ids))
        print(f"\n{p['runs_reviewed']} run(s) reviewed")
        for r in p["recurring"]:
            print(f"  {r['runs']:>3} runs ({int(r['share']*100)}%)  {r['id']}")
        return
    rid = args[0] if args else latest_run_id()
    if not rid:
        print("No runs found under media_library/debug/")
        return
    data = review_and_save(rid)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
