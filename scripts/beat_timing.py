#!/usr/bin/env python3
"""
beat_timing.py — how long each picture is on screen, known before it is drawn.

THE ORDER THIS EXISTS TO MAKE POSSIBLE. The owner's call, and it is the right
one: record the voice FIRST, measure it, and only show it to be chosen at the
end. Timings come from Whisper reading the actual audio, so until a take exists
there is nothing to measure — every shot length before this was a guess from
the script's word count, and a guess is what produced a gallery whose pictures
nobody could tell were going to flash past.

WHY ONE MEASUREMENT SERVES ALL THREE TAKES. The takes differ in pace: Kokoro's
speed runs 0.92 to 1.03 across the tones, so a forty-five second script lands
between about forty-four and forty-nine seconds. What does NOT change is how
many pictures the video wants — that is the script's beat count, and
_max_shots (audio_dur // 1.6s) only starts binding below about twenty-six
seconds of audio. A sixteen-shot script is nowhere near it at any of the three
speeds. So: one image set, measured once against a reference take, and whichever
take is chosen at the end supplies the exact cut points at render time.

The numbers this hands the gallery stage are worth having on their own. "Shot 3,
4.2s" next to a picture is the difference between choosing images and choosing
images you know will be readable — and a shot that comes back at the 1.6s floor
is a beat the narration cannot carry, which is worth seeing before forty
minutes of rendering rather than after.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def measure(mp3, script: str, n: int,
            tones: list[str] | None = None) -> list[dict]:
    """Per-shot spans for `n` pictures over this audio.

    Returns [{index, start, end, seconds}], or [] with a printed reason. Never
    raises: this is a nicety on a page, and a Whisper that will not load must
    cost the numbers rather than the stage.

    It reuses audio_gen's own transcription and cut planner rather than
    measuring a second way. Two functions that both decide where a cut goes are
    two functions that will disagree, and the one that disagrees silently is
    the one on this page — showing durations the render then does not use.
    """
    mp3 = Path(mp3)
    if not mp3.exists():
        print(f"[timing] {mp3} is not there")
        return []
    if n <= 0:
        return []
    try:
        import audio_gen
    except Exception as e:
        print(f"[timing] audio_gen unavailable ({e}) — no shot lengths")
        return []
    try:
        segments, info = audio_gen._transcribe(mp3)
        segments = list(segments)
        audio_dur = float(getattr(info, "duration", 0) or 0)
        if audio_dur <= 0:
            audio_dur = max((float(getattr(s, "end", 0) or 0) for s in segments),
                            default=0.0)
        if audio_dur <= 0:
            print(f"[timing] {mp3.name} transcribed to nothing")
            return []
        ends = audio_gen._sentence_ends(segments)
        cuts = audio_gen._plan_cuts(ends, audio_dur, n, tones)
    except Exception as e:
        print(f"[timing] could not measure {mp3.name} ({e})")
        return []

    bounds = [0.0] + list(cuts) + [audio_dur]
    spans = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        spans.append({"index": i, "start": round(start, 2),
                      "end": round(end, 2),
                      "seconds": round(max(0.0, end - start), 2)})
    return spans


def spoken_shots(mp3, n: int, tones: list[str] | None = None) -> list[dict]:
    """Per-shot spans WITH the words actually spoken over each one.

    THE GAP THIS CLOSES, and it is the one the owner has been describing since
    the first gallery. Image prompts were planned from _split_beats — a split
    of the SCRIPT TEXT. The renderer cuts on sentence boundaries found in the
    AUDIO. Nothing ever made those two agree, so shot 7's picture was drawn for
    the seventh chunk of the text while shot 7 on screen covered whatever
    happened to be said between the sixth and seventh cut. _build_sd_prompts'
    own docstring promised "the on-screen image tracks the voice-over"; it was
    true only by luck.

    Now the voice exists before the pictures, so there is no need to guess: the
    words under each shot are a fact, and they are what the prompt for that shot
    should be written from. Say "cucumber" at 12.4s and the picture covering
    12.4s is drawn from a sentence containing cucumber.

    Words are bucketed by their MIDPOINT. A word straddling a cut belongs to
    the shot it is mostly in — assigning by start would put a word that is
    almost entirely under the next picture with the previous one.
    """
    spans = measure(mp3, "", n, tones)
    if not spans:
        return []
    try:
        import audio_gen
        segments, _info = audio_gen._transcribe(Path(mp3))
        words = [w for seg in segments for w in getattr(seg, "words", []) or []]
    except Exception as e:
        print(f"[timing] no word timings ({e}) — spans without their words")
        return spans
    if not words:
        return spans

    buckets: list[list[str]] = [[] for _ in spans]
    for w in words:
        try:
            mid = (float(w.start) + float(w.end)) / 2.0
        except Exception:
            continue
        for i, sp in enumerate(spans):
            if sp["start"] <= mid < sp["end"] or (i == len(spans) - 1
                                                  and mid >= sp["start"]):
                buckets[i].append(str(w.word).strip())
                break
    out = []
    for sp, said in zip(spans, buckets):
        row = dict(sp)
        row["text"] = " ".join(said).strip()
        out.append(row)
    silent = [r["index"] + 1 for r in out if not r["text"]]
    if silent:
        print(f"[timing] shot(s) {silent} have no words under them — they sit "
              f"in a pause, and their picture has nothing to depict")
    return out


# Words that carry no subject. A shot about "the bank" and a shot about "the
# war" share "the" and nothing else, and matching on those would merge the
# whole video into one picture.
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with as is are was were be been being it its it's he she they them his her
their you your we our i me my not no so such very just also too then now than
what which who whom whose when where why how all any both each few more most
other some only own same s t can will don should could would may might must
""".split())


def _subject_words(text: str) -> set:
    import re as _re
    return {w for w in _re.findall(r"[a-z']+", (text or "").lower())
            if w not in _STOP and len(w) > 2}


def _same_subject(a: str, b: str, threshold: float = 0.34) -> bool:
    """Whether two shots are about the same thing, by shared content words.

    Deliberately lexical rather than a model call: this runs once per adjacent
    pair on every gallery, the answer has to be the same every time for the
    same script, and "do these two sentences share their nouns" is a question
    a set intersection answers honestly. A model would be slower, cost money,
    and disagree with itself between runs.

    Jaccard over the SMALLER set, not the union: a one-clause shot next to a
    long one is about the same thing if everything it names appears in the
    other, and dividing by the union would punish it for being short.
    """
    wa, wb = _subject_words(a), _subject_words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= threshold


def merge_same_subject(shots: list[dict], max_hold: float = 0.0,
                       threshold: float = 0.34) -> list[dict]:
    """Hold one picture across adjacent shots that are about the same thing.

    THE COMPLAINT THIS ANSWERS, in the owner's words: "many images repeat one
    after another — not the same image, but a new one very similar to the
    previous". Of course they did. Two consecutive sentences about the same
    bank were two prompts about the same bank, drawn from noise twice, and the
    result was a pair of near-identical pictures with a cut between them that
    the narration never asked for.

    So they become one shot with one picture, held across both. Three things
    fall out of that: the near-duplicates stop, the remaining pictures get more
    time to be read, and the GPU draws fewer of them.

    max_hold is QC's own max_hold_s — the point at which it starts reporting
    "stretches over 5s without a cut". Merging past the number QC complains
    about would trade one defect for another it already knows how to name.
    """
    if not shots:
        return []
    if max_hold <= 0:
        try:
            import video_format as _vf
            max_hold = float(_vf.get("max_hold_s", 5.0))
        except Exception:
            max_hold = 5.0

    out = [dict(shots[0])]
    for nxt in shots[1:]:
        cur = out[-1]
        joined = float(nxt.get("end", 0)) - float(cur.get("start", 0))
        if (joined <= max_hold
                and _same_subject(cur.get("text", ""), nxt.get("text", ""),
                                  threshold)):
            cur["end"] = nxt.get("end", cur.get("end"))
            cur["seconds"] = round(joined, 2)
            cur["text"] = f'{cur.get("text", "")} {nxt.get("text", "")}'.strip()
            cur["held"] = int(cur.get("held", 1)) + 1
            continue
        out.append(dict(nxt))
    for i, r in enumerate(out):
        r["index"] = i
        r.setdefault("held", 1)
    merged = len(shots) - len(out)
    if merged:
        print(f"[timing] {merged} shot(s) held rather than cut — adjacent "
              f"sentences about the same thing share one picture")
    return out


def too_short(spans: list[dict], floor: float = 0.0) -> list[int]:
    """Shots at or under the minimum segment — the beats the narration cannot
    carry. Asking for more pictures than the audio can hold is what produced
    the machine-gun run; this is that warning, one stage earlier."""
    if not spans:
        return []
    if floor <= 0:
        try:
            import audio_gen
            floor = audio_gen.MIN_SEG
        except Exception:
            floor = 1.6
    return [s["index"] for s in spans if s["seconds"] <= floor + 0.01]


def describe(spans: list[dict]) -> str:
    if not spans:
        return "no shot lengths measured"
    total = spans[-1]["end"]
    shortest = min(s["seconds"] for s in spans)
    return (f"{len(spans)} shot(s) over {total:.0f}s, "
            f"shortest {shortest:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mp3")
    ap.add_argument("script_file")
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    text = Path(a.script_file).read_text(encoding="utf-8")
    spans = measure(a.mp3, text, a.n)
    print(describe(spans))
    for s in spans:
        print(f"  {s['index']+1:2}  {s['start']:6.2f}s → {s['end']:6.2f}s  "
              f"({s['seconds']:.2f}s)")
