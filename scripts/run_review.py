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

# A held picture is the one defect a viewer feels without being able to name.
# QC already warns past 5s; 3.5s is where a fast-cut channel starts to drag.
LONG_HOLD_S = 3.5

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


def all_run_ids(limit: int = 60) -> list[str]:
    """Run folders, newest first."""
    try:
        runs = [d for d in paths.debug_root().iterdir() if d.is_dir()]
    except OSError:
        return []
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs[:limit]]


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
    return sorted(p for p in d.glob("*.png") if p.is_file())


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


def _subject_words(prompt: str) -> set[str]:
    """The content words of the BEAT half of a prompt.

    Everything from the style block onward is identical in every prompt of a
    run, so including it would make every word in it look universal. The style
    always begins at the same sentence, and splitting there is exact where a
    word-frequency threshold would be a guess.
    """
    head = re.split(r"Minimalist stick-figure|Flat 2D vector|Monochrome ink",
                    prompt, maxsplit=1)[0]
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
    counts: Counter = Counter()
    for p in prompts:
        counts.update(_subject_words(p))
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
    out = {"frames": len(frames), "near_duplicate_pairs": 0}
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

    pairs = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i][1] ^ hashes[j][1]).count("1") <= 5:
                pairs += 1
    out["near_duplicate_pairs"] = pairs
    return out


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
    if d.get("share", 0) >= DOMINANT_SHARE and d.get("word"):
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
    if dup.get("near_duplicate_pairs", 0) > 2:
        out.append({
            "id": "repeated_images",
            "severity": "medium",
            "text": (f"{dup['near_duplicate_pairs']} pairs of near-identical "
                     f"keyframes. The freshness gate only rejects identical "
                     f"bytes, so these got through by a pixel."),
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
    if m.get("beats") and m["beats"] < 12:
        out.append({
            "id": "few_pictures",
            "severity": "low",
            "text": (f"Only {m['beats']} pictures. At ~40 seconds that is "
                     f"three seconds a frame; the beat count is computed from "
                     f"script length unless SD_CLIPS overrides it."),
        })
    return out


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
        "cuts": _cut_metrics(video_path),
        "frames": _near_duplicates(frames),
    }
    m["findings"] = _findings(m)
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


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--all"]
    if "--all" in sys.argv[1:]:
        ids = all_run_ids()
        for rid in ids:
            review_and_save(rid)
        p = patterns()
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
