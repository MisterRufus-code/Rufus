#!/usr/bin/env python3
"""
main.py – Runs the full Rufus pipeline end to end.

Steps:
    1. Research a real seed (Reddit story → wisdom quote fallback)
    2. Fetch 5 candidate videos (parallel)
    3. GPT-4o Vision describes all → GPT picks the best
    4. Write script from seed + scene (35-50s, value-focused)
    5. Render: TTS + Whisper + FFmpeg → 1080x1920 mp4 (all clips cut together)
    6. Save to local SQLite DB (incl. full script + seed)
    7. Upload to YouTube (private, with thumbnail)

stdout is also tee'd to logs/rufus_YYYYMMDD.log so cron runs leave an audit trail.

Usage:
    python main.py                  # full run
    python main.py --skip-upload    # render only, no upload (for testing)
    python main.py --niche finance  # override active niche for this run
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import paths
import run_progress

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
OUTPUT_DIR  = paths.output_dir()
LOG_DIR     = paths.log_dir()

# Absolute quality floor: no per-channel config or env var may push the
# upload threshold below this, in either the auto-upload gate below or the
# dashboard's Approve action (see dashboard.py's HARD_MIN_UPLOAD_SCORE).
HARD_MIN_UPLOAD_SCORE = 7


# ── Single-instance lock (cron overlap protection) ───────────────────────────────
# Cross-platform (Windows + Linux + macOS) via filelock — POSIX os.kill(pid, 0)
# raises on Windows. An OS advisory lock held for the life of the process means a
# second overlapping run simply can't acquire it.

from filelock import FileLock, Timeout

_INSTANCE_LOCK = None   # created per-run by _acquire_lock (per-CHANNEL lock file)


def _acquire_lock(channel_id: str = "main_en", wait_seconds: float = 0) -> None:
    """Refuse to start if another run of the SAME channel is alive.

    Per-channel, not global: the corruption risk a lock protects against —
    clashing temp files and double DB writes for one channel's video — only
    exists within a channel. Different channels are safe to run concurrently
    (ComfyUI queues their GPU jobs, used_seeds.json has its own lock, SQLite
    serializes writers), and multi-channel scaling requires it.

    wait_seconds > 0 (used by --scheduled): a full-motion video can run
    1.5-2h, so 5 daily triggers spaced 3-4h apart WILL sometimes overlap.
    With the old timeout=0 the later trigger died instantly and that slot
    was silently lost (output quietly dropped from 5/day to 2-3/day, no
    alert). Waiting for the predecessor instead keeps the slot — the queue
    just serializes."""
    global _INSTANCE_LOCK
    _INSTANCE_LOCK = FileLock(str(ROOT / f"rufus.{channel_id}.lock") + ".lock")
    if wait_seconds > 0:
        print(f"[lock] channel '{channel_id}' busy — waiting up to "
              f"{wait_seconds/3600:.1f}h for the current run to finish...")
    try:
        _INSTANCE_LOCK.acquire(timeout=wait_seconds)
    except Timeout:
        print(f"ERROR: another Rufus run for channel '{channel_id}' is in progress "
              f"(lock held: rufus.{channel_id}.lock.lock). "
              f"{'Waited ' + str(int(wait_seconds/60)) + ' min, giving up. ' if wait_seconds else ''}"
              f"Wait for it, or delete the .lock file if it crashed.")
        sys.exit(1)


def _release_lock() -> None:
    try:
        if _INSTANCE_LOCK is not None and _INSTANCE_LOCK.is_locked:
            _INSTANCE_LOCK.release()
    except Exception:
        pass


def _sweep_run_temp() -> None:
    """Delete THIS run's clip temp files on exit (success or failure).

    Clip generators stamp temp names with our pid (comfy/sd _client), so the
    glob only ever matches our own files — safe under concurrent per-channel
    runs. After a successful render the clips are already muxed into the
    final mp4; after a failure they're useless — either way they'd otherwise
    sit until _housekeeping's 14-day cutoff."""
    pid = os.getpid()
    removed = 0
    for sub in ("comfy", "sd"):
        d = paths.media_root() / "temp" / sub
        if not d.exists():
            continue
        for f in d.glob(f"*_{pid}_*"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"[cleanup] removed {removed} temp clip file(s) for pid {pid}")


def _ensure_media_root() -> None:
    """Guard against media_library existing as a stray FILE instead of a
    folder (seen in the wild on Windows — AV quarantine restore, an
    interrupted download, a manual slip, a broken/orphaned reparse point
    from OneDrive placeholder corruption). Every downstream
    `.mkdir(parents=True, exist_ok=True)` call recurses up to this path, and
    `exist_ok` only suppresses FileExistsError when the existing entry
    `is_dir()` — so anything unhealthy here hard-crashes EVERY clip-fetch
    and render path with WinError 183.

    Deliberately does NOT gate on `media_root.exists()` first: a broken
    reparse point can make `os.stat()` (which `.exists()`/`.is_dir()` rely
    on) report "nothing here" while the name is still very much occupied in
    the parent directory's index — exactly the case that slips past an
    exists()-gated check and still blows up the raw CreateDirectory call
    later. Renaming the raw path only touches the directory-entry name, not
    the (possibly broken) target, so it works regardless of why the entry
    is unhealthy. A truly empty spot just raises FileNotFoundError, which is
    the expected no-op case."""
    media_root = paths.media_root()
    if media_root.is_dir():
        return   # already healthy — nothing to do
    backup = media_root.with_name(f"media_library.bak-{int(time.time())}")
    try:
        os.rename(str(media_root), str(backup))
        print(f"[maint] media_library was not a healthy folder — moved aside to {backup.name}")
    except FileNotFoundError:
        pass   # nothing was there — fine, a later mkdir(parents=True) creates it fresh
    except OSError as e:
        print(f"[maint] WARNING: media_library is unhealthy and could not be moved aside "
              f"({type(e).__name__}: {e}) — if this recurs, delete it manually: {media_root}")


# ── Housekeeping (disk + logs never grow unbounded) ──────────────────────────────

def _housekeeping(max_log_days: int = 90, max_cache_days: int = 14,
                  max_output_days: int = 14) -> None:
    """Delete old logs and stale cache/temp media. Cheap, runs every start.

    media_library/debug/ (every run's script/voiceover/keyframes) is
    deliberately EXEMPT from this sweep — it's the permanent quality-review
    record now, not a rolling cache, so it's kept forever and never
    auto-deleted here. Disk usage is the tradeoff; prune it by hand if it
    grows too large."""
    cutoff_logs  = time.time() - max_log_days * 86400
    cutoff_cache = time.time() - max_cache_days * 86400
    removed = 0
    for d, cutoff in (
        (LOG_DIR, cutoff_logs),
        (LOG_DIR / "scripts", cutoff_logs),
        (paths.media_root() / "cache", cutoff_cache),
        (paths.media_root() / "temp", cutoff_cache),
        (paths.media_root() / "music", cutoff_cache),
    ):
        if not d.exists():
            continue
        for f in d.rglob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
    removed += _housekeep_output(max_output_days)

    if removed:
        print(f"[maint] cleaned {removed} stale file(s)")

    _housekeep_debug()
    _report_debug_usage()


def _debug_usage() -> tuple[int, int]:
    """(bytes, run-folder count) under the debug root. (0, 0) if absent."""
    root = paths.debug_root()
    if not root.exists():
        return 0, 0
    total = 0
    runs = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        runs += 1
        for f in d.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    return total, runs


def _report_debug_usage() -> None:
    """Say what the permanent review record costs, and how much room is left.

    The debug tree is exempt from the sweep BY DESIGN — it is the quality
    record, not a cache. That decision is right, and it was made without a
    number attached, which is the problem: "prune it by hand if it grows too
    large" is only actionable if you can see that it has. Three things make it
    urgent on this machine specifically — the system drive has single-digit GB
    free, RUFUS_BEAT_MOTION=cut writes three stills per beat instead of one,
    and a full disk fails a render at the very end, after the GPU time is
    already spent.

    Reporting only. Set RUFUS_DEBUG_MAX_GB to actually bound it.
    """
    total, runs = _debug_usage()
    if not runs:
        return
    gb = total / 1024 ** 3
    free_gb = 0.0
    try:
        free_gb = shutil.disk_usage(paths.debug_root()).free / 1024 ** 3
    except OSError:
        pass

    line = f"[maint] debug record: {gb:.1f} GB across {runs} run(s)"
    if free_gb:
        line += f" — {free_gb:.1f} GB free on that drive"
    print(line)

    if free_gb and free_gb < 15.0:
        print(f"[maint] ⚠ only {free_gb:.1f} GB free. A render that fills the "
              f"disk fails AFTER the GPU time is spent. Prune "
              f"{paths.debug_root()}, or set RUFUS_DEBUG_MAX_GB to cap it.")


def _housekeep_debug() -> int:
    """Prune oldest debug runs down to RUFUS_DEBUG_MAX_GB. Off unless set.

    Opt-in because the tree is deliberately a permanent record; this exists so
    that decision can be bounded on a small disk without being reversed. A run
    belonging to a video still awaiting review is never pruned, for the same
    reason _housekeep_output protects its mp4 — the reviewer needs the
    keyframes and the report to judge it.
    """
    raw = os.environ.get("RUFUS_DEBUG_MAX_GB", "").strip()
    if not raw:
        return 0
    try:
        cap_bytes = float(raw) * 1024 ** 3
    except ValueError:
        print(f"[maint] RUFUS_DEBUG_MAX_GB={raw!r} is not a number — ignoring")
        return 0

    root = paths.debug_root()
    if not root.exists():
        return 0

    try:
        from db_manager import _conn
        with _conn() as c:
            protected = {r[0] for r in c.execute(
                "SELECT run_id FROM videos WHERE upload_status='pending' "
                "AND run_id IS NOT NULL").fetchall()}
    except Exception as e:
        print(f"[maint] debug prune skipped (DB unavailable: {e})")
        return 0

    dirs = []
    for d in root.iterdir():
        if not d.is_dir() or d.name in protected:
            continue
        size = sum(f.stat().st_size for f in d.rglob("*")
                   if f.is_file()) if d.exists() else 0
        dirs.append((d.stat().st_mtime, size, d))

    total = _debug_usage()[0]
    removed = 0
    for _mtime, size, d in sorted(dirs):
        if total <= cap_bytes:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        removed += 1
    if removed:
        print(f"[maint] pruned {removed} debug run(s) to stay under "
              f"{raw} GB (runs awaiting review were kept)")
    return removed


def _housekeep_output(max_output_days: int) -> int:
    """Delete rendered videos + their thumbnail/QC sidecars older than the
    window — output/ was NOT covered before, so at 5 videos/day it grew
    without bound (mp4s + .thumb.jpg + .mp4.qc.json forever).

    CRITICAL protection: a video still `pending` review is NEVER deleted,
    regardless of age — the reviewer (possibly a different person, days
    later) needs the file to approve it. Only approved (already on YouTube,
    so the local copy is a disposable backup) and rejected (never shipping)
    videos are swept. Fail-safe: any DB error skips output cleanup entirely
    rather than risk deleting a file whose status we can't confirm."""
    out_dir = OUTPUT_DIR
    if not out_dir.exists():
        return 0
    cutoff = time.time() - max_output_days * 86400
    try:
        from db_manager import _conn
        with _conn() as c:
            protected = {r[0] for r in c.execute(
                "SELECT video_file FROM videos WHERE upload_status='pending' "
                "AND video_file IS NOT NULL").fetchall()}
    except Exception as e:
        print(f"[maint] output cleanup skipped (DB unavailable: {e})")
        return 0

    removed = 0
    for mp4 in out_dir.rglob("*.mp4"):
        try:
            if str(mp4) in protected:
                continue                       # awaiting review — never delete
            if mp4.stat().st_mtime >= cutoff:
                continue                       # still within the keep window
            for sidecar in (mp4, mp4.with_suffix(".thumb.jpg"),
                           Path(str(mp4) + ".qc.json")):
                if sidecar.exists():
                    sidecar.unlink()
                    removed += 1
        except OSError:
            continue
    return removed


# ── Tee stdout/stderr to a daily log file ───────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def _enable_file_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"rufus_{time.strftime('%Y%m%d')}.log"
    log_fp   = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    return log_path


sys.path.insert(0, str(Path(__file__).parent))
from research        import get_seed
from media_fetcher   import fetch_candidates
from llava_tagger    import pick_best_video
from script_writer   import (write_script, write_script_until_good, preanalyze,
                             check_blacklist, add_to_blacklist)
from audio_gen       import render
from db_manager      import init_db, save_video, update_youtube_id


def _parse_video_queries(analysis: str) -> list[str]:
    """Extract VIDEO QUERIES line from pre-analysis output (item 7)."""
    for line in (analysis or "").split("\n"):
        if "VIDEO QUERIES" in line.upper() or "VIDEO QUERY" in line.upper():
            parts = line.split(":", 1)
            if len(parts) > 1:
                queries = [q.strip().strip("'\"") for q in parts[1].split(",")]
                return [q for q in queries if len(q) > 2][:3]
    return []


# Per-beat shot types — four FRAMINGS for the literal subject the narrator names.
# They rotate so a Short reads like real coverage (intimate detail → person in the
# situation → the place, wide → the object itself) instead of four identical
# framings. Each framing is just a lens on the REAL thing the line is about — the
# emotion/light is treatment, never a replacement for showing the actual subject.
_SD_ANCHORS = [
    {
        # Intensity — push in tight on the literal thing so its detail fills frame.
        "camera": "extreme close-up, 100mm macro, f/2.8, razor-thin focus, slight handheld imperfection",
        "subject_hint": "the LITERAL thing the line names, shot in tight macro — real banknotes and their texture, the actual numbers on a screen, hands doing the specific action — detail fills the frame",
        "light": "single hard light raking across at 45°, deep inky shadow, one bright specular catch",
    },
    {
        # Human — a real person literally doing/feeling what the line describes.
        "camera": "medium portrait, 85mm, f/1.8, eyes tack-sharp, shot at eye level",
        "subject_hint": "a real, ordinary person literally IN the situation the line describes — at the desk paying bills, checking the account, mid-decision — candid, unposed, a real feeling on the face, never smiling at the camera",
        "light": "moody window light from one side, soft warm rim from behind, natural skin, real pores",
    },
    {
        # Scale — the actual place/scene the line is set in, shown wide.
        "camera": "wide establishing, 24mm, deep focus",
        "subject_hint": "the actual place or scene the line describes, shown wide — the office, the trading floor, the home, the city — a person small in it only if the line implies one",
        "light": "cold blue pre-dawn or hard golden-hour, long shadows, atmospheric haze, real depth",
    },
    {
        # Object — the exact object the line names, shot cleanly as the hero.
        "camera": "tight overhead or 50mm still-life, f/4, deliberate composition",
        "subject_hint": "the exact object the line names, shot as the hero of the frame — the paycheck, the bank statement, the stack of bills, the contract, the worn tool — real and specific, NOT an abstract symbol",
        "light": "soft directional light revealing every texture, gentle shadow, tactile and real",
    },
]

# Which _SD_ANCHORS index each beat gets, in order (repeats once exhausted).
# NOT a plain beat_index % 4 — that put "wide establishing" (index 2, the one
# framing that pulls the viewer OUT to a spectator's view of the scene) into
# every 4th beat mechanically, ~25% of a video, regardless of what the line
# actually needed. Live feedback: the viewer doesn't feel "inside" the video.
# A Short holds attention by staying physically close — hands, faces, the
# object itself — not by cutting away to establish geography every four
# beats like a documentary. Wide still exists (useful for a genuine scene-
# setting moment, e.g. "picture a 19th-century marketplace") but now only
# once every 6 beats (~17%) instead of every 4th (25%), with close-up and
# portrait — the two intimate framings — carrying most of the video.
_ANCHOR_SEQUENCE = [0, 1, 3, 1, 0, 2]   # close-up, portrait, object, portrait, close-up, wide


def _anchor_for_beat(i: int) -> dict:
    """The camera/subject/light anchor for beat index `i` (0-based)."""
    return _SD_ANCHORS[_ANCHOR_SEQUENCE[i % len(_ANCHOR_SEQUENCE)]]


def _split_beats(script: str, max_scenes: int = 10, min_words: int = 3) -> list[str]:
    """Split a script into ordered visual beats (one per spoken sentence).

    Merges fragments shorter than ``min_words`` into the previous beat, then
    collapses the shortest adjacent beats until at most ``max_scenes`` remain —
    so each beat is a meaningful chunk of speech that earns its own image.
    Order is preserved, which is what lets clip[i] line up with beat[i] at
    render time (audio_gen cuts on sentence boundaries in list order).
    """
    import re
    # Same abbreviation guard as audio_gen._sentence_ends: a naive split on
    # [.!?] chopped "...saw the U.S. government issue..." into two broken beats
    # at "U.S." (seen live), giving the prompt-writer garbage like "The
    # government issue United States Notes" as a "sentence".
    abbrev = re.compile(
        r'(mr|mrs|ms|dr|st|vs|etc|inc|co|jr|sr|prof|gen|col|sgt|no'
        r'|[a-z](\.[a-z])+)\.$', re.IGNORECASE)
    parts = re.split(r"(?<=[.!?…])\s+", script.strip())
    raw: list[str] = []
    for s in (p.strip() for p in parts):
        if not s:
            continue
        # If the previous piece ended on an abbreviation (not a real sentence
        # end), this piece is its continuation — glue them back together.
        if raw and abbrev.search(raw[-1].rstrip('"\')]')):
            raw[-1] = f"{raw[-1]} {s}"
        else:
            raw.append(s)
    if not raw:
        return []

    # Merge sub-min_words fragments into the previous beat (or the next if first).
    beats: list[str] = []
    for s in raw:
        if len(s.split()) < min_words and beats:
            beats[-1] = f"{beats[-1]} {s}".strip()
        else:
            beats.append(s)
    if len(beats) > 1 and len(beats[0].split()) < min_words:
        beats[1] = f"{beats[0]} {beats[1]}".strip()
        beats = beats[1:]

    # Collapse the shortest adjacent pair until we fit max_scenes.
    while len(beats) > max_scenes:
        widths = [len(b.split()) for b in beats]
        # index of the adjacent pair with the smallest combined width
        j = min(range(len(beats) - 1), key=lambda k: widths[k] + widths[k + 1])
        beats[j] = f"{beats[j]} {beats[j + 1]}".strip()
        del beats[j + 1]
    return beats


# Text-bearing props that diffusion models render as instantly-recognizable
# AI gibberish (garbled headlines, fake UI, nonsense digits). The prompt
# instruction bans making them readable, but GPT drifts — seen live:
# "calendar page turning to December 31, 2022", "newspaper headlines about
# the crisis", "'Follow' button with Bitcoin graphics". This deterministic
# net catches every prompt that mentions one and appends a defusing clause,
# regardless of whether the instruction was obeyed.
#
# TWO KINDS OF TRIGGER, and the second was missing. An OBJECT that bears text
# (newspaper, ledger, screen) is easy to name. A SCENE that implies text is
# not: nobody writes "sign" when they write "a protest of unemployed workers",
# but a protest is placards, and placards are lettering. Live proof from the
# Great Depression run — shot 7 was "A group of well-dressed individuals
# ignoring a nearby protest of unemployed workers", it matched nothing here, so
# the blank-surfaces clause never fired, and the rendered image came back with
# signs reading "ISSUES" in garbled type. Same gap for a storefront, a trading
# floor, a classroom, a memorial.
_TEXT_PROP_RE = re.compile(
    r"(?i)\b(newspaper|headline|calendar|screen|display|smartphone|phone|laptop|"
    r"monitor|button|sign|signage|label|poster|banner|placard|document|ledger|"
    r"letter|scroll|parchment|statement|contract|certificate|chart|graph|"
    r"ticker|keyboard|billboard|menu|book|page|note|"
    # scenes that are made of lettering even when no object is named
    r"protest|protesters|demonstration|rally|march|picket|strike|"
    r"storefront|shopfront|shop front|store front|marquee|"
    r"stock exchange|trading floor|newsstand|classroom|blackboard|whiteboard|"
    r"memorial|gravestone|headstone|plaque|map)\b")

# PHRASED AFFIRMATIVELY, ON PURPOSE. The previous version of this clause read
# "…absolutely no readable text, numbers, or interface elements anywhere in the
# image" — a negation inside the POSITIVE prompt, which is the one place it
# cannot work: CLIP has no "not" operator, so the encoder saw the tokens text,
# numbers, readable, lettering and the sampler painted them. A live batch of 40
# money_history stills came back with invented lettering on a coin, a
# newspaper, a ledger, a bank facade and two documents — every one of those
# prompts carried the old clause. The suppression now lives in the negative
# conditioning (comfy_client.DEFAULT_STILLS_NEGATIVE, substituted by
# comfy_template into the sampler's own negative wire); what stays here is a
# POSITIVE description of the surface we want — blank paper — which the
# sampler can actually render.
_DETEXT_SENTINEL = "blank and unmarked"
_DETEXT_CLAUSE = (
    " Every page, sign, coin face, and screen in the frame is blank and "
    "unmarked — plain smooth empty surfaces, angled away or seen at a "
    "distance, described by shape, color and wear alone.")


def _defuse_readable_text(prompt: str) -> str:
    """Append the blank-surfaces clause to prompts that mention a text-bearing
    prop. Only when triggered — a clean prompt stays untouched (keeps token
    budgets tight and avoids diluting every prompt)."""
    if _TEXT_PROP_RE.search(prompt) and _DETEXT_SENTINEL not in prompt.lower():
        return prompt.rstrip() + _DETEXT_CLAUSE
    return prompt


# ── Per-beat era ─────────────────────────────────────────────────────────────
# A money_history script is NOT set in one period. It opens in history and lands
# in the viewer's present — that pivot is the whole format. A period rule applied
# to the WHOLE run therefore contradicts the beats that were written to be
# modern, and the contradiction was live three times over:
#
#   run #58 beat 08  "a group of economists and policymakers in a CONTEMPORARY
#                     setting" + "Period setting: 1791 ... no business suits"
#   run #58 beat 10  the Chronicler in a "timeless green field" + the same 1791
#                     clause, which is why he came back in modern dress
#   run #53 beat 10  "a CONTEMPORARY classroom" + "Period setting: 1791" — on a
#                     script about the 1923 Rentenmark, so the year was wrong
#                     for the whole video, not just that beat
#
# The era is therefore derived per beat, from THIS run's script, and stated in
# the instruction the prompt-writer reads — rather than pasted onto every prompt
# after the fact, where it can only ever fight the beats it doesn't fit.

# Words that mean "the viewer's present", unambiguously.
_PRESENT_DAY_RE = re.compile(
    r"(?i)(\btoday\b|\btoday's\b|\bnowadays\b|\bthese days\b|\bright now\b|"
    r"\bmodern\b|\bmodern-day\b|\bcontemporary\b|\bcurrent(ly)?\b|"
    r"\b21st century\b|\bstill (happens|going|true|works|do|does)\b)")

# Second person is a WEAK signal, and the reason is a collision between two
# rules this pipeline sets itself. The SOUND section requires every script to
# address the viewer at least once ("A script with no 'you' in it is a
# lecture"), so "you" appears in historical beats as a rhetorical device:
# "You could swap cheaper silver for premium gold at the fixed rate" is 1873,
# not now. Treating "you" as present-day on its own tagged that beat modern and
# produced "a wide establishing shot of a MODERN BANK ... sleek architecture
# and digital displays" inside an 1865 story. So second person only means the
# present when nothing marks the sentence as past.
_SECOND_PERSON_RE = re.compile(r"(?i)\b(you|your|you're|you've|yours)\b")

# Past-tense markers that override a rhetorical "you". Modals first: "could",
# "would" and "had" are what carry the hypothetical-historical framing.
_PAST_MARKER_RE = re.compile(
    r"(?i)\b(was|were|had|did|could|would|used to|"
    r"\w+ed)\b")

# A 3- or 4-digit year, optionally BC/BCE/AD. Bare 3-digit numbers are too
# easily a quantity ("under five percent"), so they need the era marker.
_YEAR_RE = re.compile(
    r"(?i)\b(\d{4}|\d{3}\s?(?:BC|BCE|AD|CE))\b")


def _script_period(script: str) -> str:
    """The historical period this script is set in, as a short phrase, or "".

    Taken from the EARLIEST year the script names, because the format opens in
    the past and moves toward the present: the opening year is the setting, a
    later one is usually the payoff. Derived per run from the run's own script,
    which is what stops a previous video's period leaking into this one."""
    years = _YEAR_RE.findall(script)
    if not years:
        return ""
    def _numeric(y: str) -> int:
        n = int(re.sub(r"\D", "", y))
        return -n if re.search(r"(?i)bc", y) else n
    return min((y[0] if isinstance(y, tuple) else y for y in years), key=_numeric).strip()


def _beat_is_present_day(beat: str) -> bool:
    """True when this beat speaks about the viewer's present rather than the
    past. These beats must NOT get a period rule — that is the contradiction
    that put 18th-century dress instructions on a modern classroom.

    An explicit marker ("today", "modern", "these days") decides on its own.
    Second person decides only when nothing marks the sentence as past, because
    this pipeline's own SOUND rule puts "you" into historical beats on purpose
    — see _SECOND_PERSON_RE."""
    if _PRESENT_DAY_RE.search(beat):
        return True
    return bool(_SECOND_PERSON_RE.search(beat)) and not _PAST_MARKER_RE.search(beat)


def _beat_era_tag(beat: str, period: str) -> str:
    """The era label shown next to a beat in the prompt-writer's instruction."""
    if _beat_is_present_day(beat) or not period:
        return "present day"
    return period


def _strip_beat_echo(line: str, beat: str) -> str:
    """Remove the beat's narration text if the prompt-writer echoed it.

    Seen live: despite instructions, GPT prefixed each image prompt with its
    beat's spoken sentence ("During the Civil War, 1862 saw the U.S. A medium
    portrait of...") — narration text inside a FLUX prompt dilutes the visual
    description and risks the model painting words. Deterministic guard: if
    the line starts with the beat's opening words, cut them and keep the rest
    (only when what remains is still a usable prompt)."""
    b = beat.strip().rstrip(".!?…").strip()
    if len(b) < 15:
        return line
    # Longest common prefix between the line and the beat (case-insensitive):
    # cut exactly what was echoed — a full echo, or a partial one.
    ll, bl = line.lower(), b.lower()
    lcp = 0
    while lcp < min(len(ll), len(bl)) and ll[lcp] == bl[lcp]:
        lcp += 1
    if lcp >= 15:
        rest = line[lcp:].lstrip(" .!?…—-")
        if len(rest) > 20:
            return rest[0].upper() + rest[1:]
    return line


# ── Cross-run image freshness ────────────────────────────────────────────────
# The script side already blocks repeats (blacklist + embedding gate), but
# nothing stopped every money_history video from opening on the same "vintage
# banknote close-up". A rolling log of recent runs' image prompts is fed back
# into the prompt-writer as a DO-NOT-REPEAT list, so each run must find new
# visual angles. Channel-scoped: two channels never censor each other's ideas.

RECENT_PROMPTS_FILE = CONFIG_DIR / "recent_image_prompts.json"


def _recent_image_prompts(limit_runs: int = 8, max_lines: int = 25) -> list[str]:
    """Last few runs' image prompts for the active channel, oldest first."""
    try:
        data = json.loads(RECENT_PROMPTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ch = os.environ.get("RUFUS_CHANNEL", "main_en")
    runs = [r for r in data.get("runs", []) if r.get("channel", "main_en") == ch]
    prompts: list[str] = []
    for r in runs[-limit_runs:]:
        prompts.extend(r.get("prompts", []))
    return prompts[-max_lines:]


def _remember_image_prompts(prompts: list[str], cap_runs: int = 24) -> None:
    """Append this run's accepted image prompts to the rolling log (capped)."""
    if not prompts:
        return
    data: dict = {}
    if RECENT_PROMPTS_FILE.exists():
        try:
            data = json.loads(RECENT_PROMPTS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    runs = data.get("runs", [])
    runs.append({
        "ts": int(time.time()),
        "channel": os.environ.get("RUFUS_CHANNEL", "main_en"),
        "prompts": [p[:160] for p in prompts],
    })
    data["runs"] = runs[-cap_runs:]
    try:
        RECENT_PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_PROMPTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[fresh] couldn't save image-prompt history: {e}")


def _freshness_block() -> str:
    """DO-NOT-REPEAT block for the prompt-writer, or '' on a channel's first runs."""
    recent = _recent_image_prompts()
    if not recent:
        return ""
    lines = "\n".join(f"  - {p[:90]}" for p in recent)
    return (
        "\nFRESHNESS — DO NOT REPEAT RECENT VIDEOS:\n"
        "The visual ideas below already appeared in this channel's recent videos. "
        "Do NOT reuse or closely echo any of these subjects/compositions. If a beat "
        "naturally suggests one of them, pick a DIFFERENT literal subject from that "
        "beat, or a sharply different moment, composition, era-detail, or setting:\n"
        f"{lines}\n"
    )


def _split_merged_prompts(blob: str, n: int) -> list[str]:
    """Recover `n` image prompts from a reply that arrived as one paragraph.

    The prompt builder asks for one prompt per line; when the model ignores
    that, the newline split yields a single enormous "prompt" and the run
    renders ONE image for the whole video. Each prompt is 2-4 sentences, so
    the sentences are regrouped into n even chunks — imperfect where a chunk
    boundary lands mid-scene, but far better than one image per video.

    Returns [] when the blob plainly isn't a merged batch (too few sentences
    to split), so the caller keeps whatever it already had."""
    import re as _re
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", blob.strip()) if s.strip()]
    # Need at least one sentence per prompt to chunk at all. A genuinely
    # single prompt (2-4 sentences) against n=10 fails this and is left alone,
    # which is the protection against mangling a reply that wasn't merged.
    if n <= 1 or len(sentences) < n:
        return []
    per = len(sentences) / n
    out = []
    for i in range(n):
        chunk = " ".join(sentences[int(round(i * per)):int(round((i + 1) * per))]).strip()
        if len(chunk) <= 20:
            return []
        out.append(chunk)
    return out if len(out) == n else []


def _build_sd_prompts(script: str, niche: str, max_scenes: int = 10) -> list[str]:
    """One ultra-detailed SD prompt per spoken beat, in narration order.

    Each prompt's SUBJECT depicts what the narrator says during that beat (a
    photo of stocks while he talks about stocks), so when the renderer cuts on
    sentence boundaries the on-screen image tracks the voice-over. Prompts use
    pro Realistic-Vision token language with a rotating camera anchor for visual
    variety and the niche's color grade. Returns one prompt per beat (≤max_scenes).
    """
    import re

    beats = _split_beats(script, max_scenes=max_scenes)
    if not beats:
        beats = [f"{niche} concept"]
    n = len(beats)

    try:
        niche_data  = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
        niche_style = niche_data["niches"].get(niche, {}).get("style_suffix", "")
    except Exception:
        niche_style = ""
    color_grade = niche_style or "cinematic color grade, muted tones, film grain"

    def _fallback_prompt(i: int, beat: str) -> str:
        a   = _anchor_for_beat(i)
        cue = beat[:80].rstrip(".,;:! ")
        return (
            f"RAW photo, ({cue}:1.35), {a['subject_hint']}, {a['light']}, "
            f"{a['camera']}, {color_grade}, photorealistic, hyperrealistic, "
            "sharp focus, professional editorial photography, film grain"
        )

    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        key = ""
        if keys_file.exists():
            key = json.loads(keys_file.read_text(encoding="utf-8")).get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            raise RuntimeError(
                "OpenAI key missing — SD prompt generation requires GPT-4o-mini.\n"
                "Add 'openai' key to config/keys.json or use RUFUS_VIDEO_SOURCE=pexels."
            )
        period = _script_period(script)

        try:
            import character_engine
            # Not is_flux-gated — character_engine.py is generic per-niche
            # (money_history's timeless Chronicler is comfy/FLUX, but the SD
            # niches — finance/motivation/mindset/business/personal_development
            # — each ship their own starter character too, disabled by
            # default). character_clause() itself returns "" for any niche
            # without an enabled character block, so this is a no-op today
            # for every SD niche until the owner opts one in.
            # n_beats lets "anchor" mode name the exact beat numbers the
            # character appears in; "all" mode ignores it.
            char_clause = character_engine.character_clause(niche, len(beats))
        except Exception as e:
            # Fail-open like every other optional step, but SAY SO. This used to
            # swallow the error silently, which meant a broken character config
            # was indistinguishable from a working one that the model ignored —
            # exactly the ambiguity that made the live "character never appears"
            # report expensive to diagnose.
            print(f"           ⚠ character clause skipped (non-fatal): {e}")
            char_clause = ""

        # STORYBOARD FIRST. The per-beat writer below has never seen the story
        # — it gets ten sentences and illustrates each alone, which is how a
        # line about the denarius's silver content became "a family gathered
        # around a modest dinner table". One pass over the WHOLE script plans
        # the shots as a sequence instead, so they can carry something forward
        # from each other. Falls through to the per-beat path on any failure.
        try:
            import storyboard
            shots = storyboard.plan(
                script, beats,
                era_tags=[_beat_era_tag(b, period) for b in beats],
                character_clause=char_clause, niche=niche)
            if shots:
                shots = [_defuse_readable_text(s) for s in shots]
                for i, s in enumerate(shots):
                    print(f"             {i+1}. {s}")
                return shots[:n]
        except Exception as e:
            print(f"           ⚠ storyboard skipped (non-fatal): {e}")

        beat_lines = "\n".join(
            f"  Beat {i+1} [ERA={_beat_era_tag(b, period)}] "
            f"(CAMERA={_anchor_for_beat(i)['camera'].split(',')[0]}): "
            f"\"{b}\""
            for i, b in enumerate(beats)
        )
        anchor_lines = "\n".join(
            f"  Beat {i+1}: framing={_anchor_for_beat(i)['subject_hint']}; "
            f"lighting={_anchor_for_beat(i)['light']}; "
            f"lens={_anchor_for_beat(i)['camera']}"
            for i in range(n)
        )
        # FLUX (ComfyUI) reads full natural-language sentences and renders
        # period-accurate scenes far better than SD1.5 tag-soup. Detect the
        # target engine so FLUX niches get sentence prompts that lock each
        # image to its beat's literal subject.
        _vs = os.environ.get("RUFUS_VIDEO_SOURCE", "").strip().lower()
        is_flux = _vs == "comfy"
        if not _vs:
            try:
                _nd = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
                is_flux = _nd["niches"].get(niche, {}).get("video_source") == "comfy"
            except Exception:
                is_flux = False

        fresh_block = _freshness_block()

        _FLUX_INSTRUCTION = (
            "You write prompts for FLUX.1-dev, which understands full "
            "natural-language sentences (NOT comma tag-soup).\n"
            f"Write EXACTLY {n} image prompts for a '{niche}' YouTube Short — "
            "ONE per spoken beat, in narration order. Together they are a "
            "PHOTO-ESSAY telling one story, not a collection of stock shots.\n\n"
            "CRITICAL — THE SCENE MUST MATCH THE SCRIPT: prompt N depicts the EXACT "
            "literal thing the narrator says in beat N. If a beat names a historical "
            "period, place, currency, person, or object, the image MUST show THAT, "
            "and be period-accurate.\n\n"
            "SPOKEN BEATS (prompt N must show beat N):\n"
            f"{beat_lines}\n\n"
            # Character clause sits HERE, immediately after the CRITICAL block
            # and before the 15-rule list, not buried inside it. It was
            # previously rule 8 of 15 (~41% into an 8KB instruction) with 3.5KB
            # of further rules after it and no restatement — and live runs
            # produced 10/10 prompts with no character at all. Head position +
            # the closer restatement below put it in both spots the model
            # actually weights.
            f"{char_clause}"
            + ("\n" if char_clause else "")
            + "RULES:\n"
            "- 2 to 4 vivid natural-language sentences per prompt.\n"
            "- DESCRIBE ONLY WHAT THE CAMERA SEES. Never quote, repeat, or paraphrase "
            "the beat's narration text inside the prompt — the narration is the "
            "voice-over, not the image. A prompt that opens by restating the beat "
            "('During the Civil War, 1862 saw...') is a FAILURE; open with the shot "
            "itself ('A medium portrait of...').\n"
            "- LOCK TO THE BEAT'S ANCHOR: find the single most specific noun, proper "
            "noun, number, date, or named object in beat N — the exact thing a viewer "
            "hearing that sentence would picture — and make THAT the subject of prompt "
            "N. Never drift to a generic or merely thematically-related scene instead.\n"
            "- Show the LITERAL subject. Examples: 'the first coins of Lydia' -> a macro "
            "shot of ancient electrum Lydian stater coins on a worn stone counter; "
            "'Weimar hyperinflation' -> 1923 Germany, a wheelbarrow overflowing with "
            "near-worthless Reichsmark banknotes on a cobbled street; 'Bretton Woods' -> "
            "a 1944 conference hall, men in 1940s suits around a long table.\n"
            "- ERA — OBEY THE [ERA=...] TAG ON EACH BEAT ABOVE, PER BEAT. This script "
            "deliberately moves from the past to the viewer's present; the beats are "
            "NOT all in one period, and treating them as if they were is the single "
            "worst thing you can do here.\n"
            "  • [ERA=<a year>] → every visible detail belongs to that time and place: "
            "clothing, hairstyles, footwear, tools, vehicles, architecture, money. No "
            "business suits, t-shirts, jeans, trainers, backpacks, cars, tarmac, street "
            "lighting, power lines, screens, printed banknotes or credit cards.\n"
            "  • [ERA=present day] → an ordinary scene of TODAY. Do NOT give it "
            "historical dress, props, or setting. A period costume on a present-day "
            "beat is a hard failure, exactly as bad as a phone in a 1791 scene.\n"
            "- Vary framing across consecutive beats (macro object, wide establishing "
            "shot of a place, a person's hands handling the item, overhead of documents). "
            "Never repeat the same framing back-to-back.\n"
            "- PEOPLE & FACES — this model renders a tight, front-facing, emotional "
            "close-up of a face worst (distorted, uncanny results). So NEVER write an "
            "extreme close-up of a single face. Keep any person MID-DISTANCE or smaller "
            "in the frame, shown from a three-quarter or profile angle, looking at what "
            "they are doing rather than at the camera, or with the focus on their hands / "
            "the object / the wider scene. For a named real person, evoke them through "
            "the setting, period, and action rather than a tight portrait. When a face "
            "is visible, describe it as natural, calm, and anatomically normal.\n"
            "- FLAT 2D ILLUSTRATION, NOT A PHOTOGRAPH: this must read as clean vector-"
            "style illustration — never a photograph, 3D render, or photorealistic "
            "CGI. Simplified geometric shapes, confident bold outlines of consistent "
            "weight, flat unshaded color fills — no film grain, no lens blur, no "
            "photographic depth of field, no skin pores or fabric-weave texture. "
            "Figures and objects are graphic and stylized rather than anatomically "
            "photographic: bold silhouettes, minimal internal linework, expressive "
            "poses read through shape and posture rather than photographic detail. "
            "Backgrounds simplify into clean shapes and negative space, never "
            "photographic clutter. Avoid gradients, soft shadows, or any rendering "
            "technique that reads as a photograph trying to look like art.\n"
            "- Apply this exact color grade to EVERY prompt: "
            f"{color_grade}.\n"
            "- MAKE THE FRAME ALIVE — every image must contain a story, not a catalog "
            "shot: something mid-action or freshly happened (hands mid-motion, a crowd "
            "caught mid-rush, smoke rising in a simplified graphic curl, ink still wet, "
            "a chair just pushed back), atmosphere through bold graphic shape rather "
            "than photographic realism (a hard-edged light-shape cutting across the "
            "panel, a strong silhouette against a color-block sky, dramatic scale "
            "contrast between figure and setting), and confident flat lighting with "
            "real graphic contrast (a bold dark shape against a bright color field, a "
            "single warm light-shape in an otherwise dark panel) — never a boring, "
            "evenly-lit centered layout. A static object centered on a plain background "
            "is a FAILURE; give the subject context, scale, "
            "and consequence.\n"
            "- PEOPLE DOING THINGS, NOT OBJECTS ON TABLES: most frames must show real "
            "people mid-action inside the beat's world — counting, arguing, queueing, "
            "signing, hauling, fleeing, celebrating — at mid-distance per the face rules "
            "above. At most 2 of the prompts may be object-only shots, and even those "
            "must show ACTIVE consequence (banknotes burning in a stove, coins spilling "
            "from a dropped purse, a ledger left open in the rain) — never a display "
            "shot of the item on a surface.\n"
            "- TELL ONE STORY ACROSS THE SEQUENCE: while every prompt stays locked to "
            "its own beat's literal subject, the images together must read as one "
            "unfolding photo-essay — establish the world and its people, show the "
            "pressure building through their actions, land the turning point, show the "
            "aftermath. Carry a visible recurring thread (the same kind of place, "
            "people, or object) through the sequence and let its CONDITION evolve with "
            "the story: pristine → strained → transformed. The last frame should feel "
            "like the consequence of the first, not an unrelated image.\n"
            "- NEVER NAME WORDS THAT WOULD BE PRINTED IN THE FRAME. The image model "
            "garbles written words, and the gibberish instantly exposes the image as "
            "AI. The moment your prompt says WHAT a headline/coin/sign/ledger/document "
            "reads, the model tries to paint those letters and fails. So: never quote "
            "or describe wording, never name the newspaper, bank, or company, never "
            "give a date or figure that would appear ON an object. Write the object as "
            "a blank physical thing instead — 'a folded newspaper, its page blank "
            "newsprint' not 'a newspaper headlined BANK PANIC'; 'a worn gold coin, its "
            "face a smooth featureless disc' not 'a coin stamped 1907'. Better still, "
            "pick a subject that carries no writing at all: hands, faces, a queue, a "
            "locked door, an empty vault, spilled coins. (Rufus overlays its own "
            "captions — the image never needs to say anything.)\n"
            "- All prompts must be visually distinct.\n"
            f"{fresh_block}\n"
            # Restated last, in the model's other high-attention position. The
            # freshness block above is an explicit "do NOT repeat anything from
            # recent videos" list that can run ~2.7KB — without this line right
            # after it, a recurring character reads as exactly the thing it's
            # telling the model to stop doing.
            + (f"REMINDER — the recurring character above is REQUIRED and is the one "
               f"element that SHOULD repeat across prompts and across videos; the "
               f"freshness list never applies to it.\n\n" if char_clause else "")
            + f"Output EXACTLY {n} prompts, one per line. No numbering, no labels, no blank lines."
        )

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": _FLUX_INSTRUCTION if is_flux else (
                    "You are a cinematographer turned Stable Diffusion prompt writer "
                    "(Realistic Vision v5.1). You don't make stock photos — you make "
                    "frames that feel SHOT by a human for one specific line of narration.\n"
                    f"Write EXACTLY {n} image prompts for a {niche} YouTube Short — one per beat.\n\n"
                    "SPOKEN BEATS — prompt N must SHOW exactly what the narrator says in beat N:\n"
                    f"{beat_lines}\n\n"
                    "PER-BEAT FRAMING — use this framing to show the beat's literal subject:\n"
                    f"{anchor_lines}\n\n"
                    "TOKEN FORMAT — comma-separated SD tokens only, NO sentences:\n"
                    "RAW photo, (SUBJECT:1.35), ACTION/POSE, SETTING+TEXTURE, LIGHTING, OPTICS, COLOR_GRADE\n\n"
                    "MANDATORY SECTIONS — every prompt must contain ALL 7, in this exact order:\n"
                    "1. ACTIVATOR: always 'RAW photo,' — never skip this\n"
                    "2. SUBJECT (20-25 words): ultra-specific — age, build, ethnicity, exact clothing texture, "
                    "facial expression, one dominant emotion. Boost weight: (description:1.35). "
                    "Example: (weathered South Asian man 48yo, deep-set eyes with 3am shadows, "
                    "loosened silk tie half-undone, jaw tight, quiet dread on face:1.35)\n"
                    "3. ACTION/POSE (8-10 words): exactly what the subject is doing or the object's "
                    "precise orientation — 'both hands gripping phone screen, knuckles white'\n"
                    "4. SETTING+TEXTURE (15-20 words): specific location with material detail — "
                    "'home office at 3am, cold monitor glow, scattered bank statements, coffee ring "
                    "on oak desk, fingerprints on glass, empty takeout container'\n"
                    "5. LIGHTING (10-12 words): one named source with color temperature and shadow quality — "
                    "'single 3200K tungsten desk lamp at 45 degrees left, deep inky shadow right side, "
                    "specular catch light on forehead'\n"
                    "6. OPTICS (8-10 words): camera + lens + aperture + focus point — "
                    "'Nikon Z7II, 85mm f/1.4, focus on eyes, subject fills 70 percent of frame'\n"
                    "7. COLOR_GRADE (5-8 words): niche-specific — "
                    f"'{color_grade}'\n\n"
                    "CONTENT RULES — in priority order:\n"
                    "• ACCURACY FIRST: subject of prompt N = the LITERAL concrete thing the narrator "
                    "names in beat N. 'paycheck' → real paycheck with printed numbers and routing "
                    "numbers visible; 'savings' → physical cash going into account or jar; "
                    "'the market' → actual stock ticker board with red/green numbers; "
                    "'debt' → real credit-card statement with balance highlighted. SHOW THE THING.\n"
                    "• ZERO ABSTRACT SYMBOLISM: never 'envelope signifying decisions', 'road "
                    "representing the journey'. If the beat is abstract, find the most concrete "
                    "object a real person would actually see in that situation.\n"
                    f"{char_clause}"
                    "• EMOTION SECOND: once the literal subject is locked, add real emotion through "
                    "posture, expression, lighting — never a neutral or smiling pose, never looking at camera.\n"
                    "• BEAT 1 = the scroll-stopper: highest contrast, most arresting framing of the "
                    "literal subject, most emotionally charged — the frame that makes someone stop scrolling.\n"
                    "• SETTING must be lived-in and specific: scattered papers, cold monitor glow, "
                    "coffee rings, worn upholstery, fingerprints on glass. Never clean and generic.\n"
                    "• MID-ACTION, NEVER A DISPLAY SHOT: every frame must catch an action ALREADY "
                    "UNDERWAY at its most unstable instant — coins mid-spill and still falling, a hand "
                    "closing on a ledger, a door swinging, paper lifting in a draught, a figure "
                    "turning away. Frozen arrangements ('a coin ON a table', 'documents laid out') are "
                    "the single biggest cause of dead, static-looking video: the animator continues the "
                    "motion it can SEE, so a still with no implied movement can only be slowly zoomed. "
                    "Give it a vector to continue.\n"
                    "• MATCH THE EMOTIONAL REGISTER of that beat's narration, in the physics of the "
                    "shot. Collapse/loss/panic → things falling, scattering, tipping past balance, dust "
                    "lifting, a downward or destabilised camera. Growth/triumph → rising, opening, "
                    "light breaking through, an upward or widening camera. Secrecy/tension → a hand "
                    "withdrawing into shadow, a door closing to a slit, something half-hidden. "
                    "Revelation → something being pulled open, uncovered, lifted into light. The "
                    "viewer should feel the beat before they parse the object.\n"
                    "• MICRO-DETAIL: name the small physical facts a macro lens would catch — "
                    "worn edges, dust, fingerprints, hairline scratches, tarnish, condensation, "
                    "loose threads, paper fibre, the grain of the material. These specifics are "
                    "what make a frame read as a real photograph instead of a render.\n"
                    "• NO quality tokens (8k, masterpiece, best quality) — the image model's text "
                    "encoder is an LLM that reads description, not tag spam, and the photographic "
                    "direction (lens, aperture, lighting falloff, grain) is appended separately.\n"
                    f"• All {n} prompts must be completely distinct: different subjects, settings, framings.\n"
                    "• BAN: posed, smiling at camera, corporate handshake, generic silhouette, "
                    "stock photo, perfect unblemished skin, catalog pose.\n"
                    "• STORY THREAD: the frames together should read as ONE unfolding story "
                    "(setup → pressure → turning point → aftermath) with people mid-action "
                    "driving it — object-only shots max 2 of the batch, and only with active "
                    "consequence, never a display shot.\n"
                    f"{fresh_block}\n"
                    "90–110 words per prompt, written as vivid natural-language description "
                    "(the stills model's text encoder is an LLM — it reads prose far better "
                    "than a comma-separated tag dump).\n\n"
                    f"Output EXACTLY {n} lines. No numbering, no labels, no blank lines. Beat order."
                ),
            }],
            max_tokens=1800,
            temperature=0.85,
            timeout=90,
        )
        raw_lines = resp.choices[0].message.content.strip().split("\n")
        lines = [re.sub(r"^[\d\.\-\)\s]+", "", l).strip()
                 for l in raw_lines if l.strip()]
        lines = [l for l in lines if len(l) > 20]

        # The whole batch arriving as ONE paragraph is a real, observed failure:
        # the instruction says "one per line", the model ignored it, and the
        # split above yielded a single line. main.py then calls generate_clips
        # with n=1, so a 40-second video was built from ONE image — with only a
        # log warning to show for it. Recover the individual prompts rather than
        # shipping that.
        if len(lines) < n and lines:
            recovered = _split_merged_prompts(max(lines, key=len), n)
            if recovered:
                print(f"[sd] GPT ignored 'one per line' — recovered {len(recovered)} "
                      f"prompts from the merged reply")
                lines = recovered
        lines = [_strip_beat_echo(l, beats[i]) if i < len(beats) else l
                 for i, l in enumerate(lines)]
        lines = [_defuse_readable_text(l) for l in lines]

        if not lines:
            raise RuntimeError("GPT returned no valid prompts for SD generation")
        if len(lines) < n:
            print(f"[sd] GPT returned {len(lines)}/{n} prompts — using partial batch")
        return lines[:n]
    except RuntimeError:
        raise   # re-raise clean errors (missing key, empty response)
    except Exception as e:
        raise RuntimeError(f"SD prompt generation failed: {e}") from e


def load_niche_cfg(override: str = None):
    data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    if override:
        if override not in data["niches"]:
            print(f"Unknown niche '{override}'. Available: {list(data['niches'])}")
            sys.exit(1)
        # Set env var so all sub-modules pick it up without touching the file on disk.
        os.environ["RUFUS_NICHE_OVERRIDE"] = override
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _todays_niche() -> str:
    """Pick today's niche from config schedule. Day-of-year mod schedule length."""
    from datetime import datetime
    data     = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    schedule = data.get("schedule") or [data.get("active", "finance")]
    doy      = datetime.now().timetuple().tm_yday   # 1-366
    return schedule[(doy - 1) % len(schedule)]


def _all_scheduled_niches() -> list[str]:
    """Return unique niches present in schedule, preserving order."""
    data     = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    schedule = data.get("schedule") or [data.get("active", "finance")]
    seen     = []
    for n in schedule:
        if n not in seen:
            seen.append(n)
    return seen


def run(skip_upload: bool = False, niche_override: str = None, output_dir: Path = None,
        channel_id: str = None, topic: str = None, lock_wait: float = 0):
    # Channel resolution FIRST (read-only) so the instance lock can be
    # per-channel — see _acquire_lock. Legacy installs without channels.json
    # get a synthesized "main_en" channel — behavior unchanged.
    from channel_config import load_channel
    channel = load_channel(channel_id)

    _acquire_lock(channel.id, wait_seconds=lock_wait)
    import atexit
    atexit.register(_release_lock)   # release on any exit path (idempotent)
    atexit.register(_sweep_run_temp) # this run's clip temps never orphan
    os.environ["RUFUS_CHANNEL"] = channel.id          # sub-modules inherit it
    if channel.voice:
        os.environ.setdefault("RUFUS_EDGE_VOICE", channel.voice)

    run_progress.begin(channel.id, niche=niche_override or "", topic=topic or "")

    log_path = _enable_file_logging()
    _ensure_media_root()
    _housekeeping()
    start    = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)
    niche_cfg = {**niche_cfg, **channel.niche_overrides.get(active, {})}
    out_dir  = output_dir or channel.output_dir

    print(f"\n{'='*52}")
    print(f"  RUFUS  |  channel: {channel.id}  |  niche: {active}  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  log:   {log_path}")
    print(f"{'='*52}\n")

    init_db()

    # ── Step 1: Research seed + pre-analyse ────────────────────────────────────
    run_progress.update(1, f'researching "{topic}"' if topic else "researching source material")
    if topic:
        print(f"[ 1 / 7 ]  Researching your topic: \"{topic}\"...")
    else:
        print("[ 1 / 7 ]  Researching real source material...")
    try:
        seed = get_seed(active, topic=topic)
        if seed.get("type") == "reddit":
            print(f"           → Reddit: {seed.get('title', '')[:80]}")
        elif seed.get("type") == "hackernews":
            print(f"           → HN:     {seed.get('title', '')[:80]}")
        elif seed.get("type") == "stackexchange":
            print(f"           → SE:     {seed.get('title', '')[:80]}")
        elif seed.get("type") == "rss":
            print(f"           → RSS:    {seed.get('title', '')[:80]}  [{seed.get('source', '')}]")
        else:
            print(f"           → Quote:  \"{seed.get('content', '')[:80]}\" — {seed.get('source')}\n")
    except Exception as e:
        print(f"           ✗ Step 1 failed: {e}")
        sys.exit(1)

    # Supervisor: reject a thin/generic/off-topic seed before spending a script
    # + render on it. One retry only (fail-open, opt out with RUFUS_SUPERVISOR=0).
    try:
        from supervisor import judge_seed
        ok, reason = judge_seed(seed, active)
        if not ok:
            print(f"           ⚠ supervisor rejected seed ({reason}) — trying one more...")
            # Manual --topic runs must never silently swap to a random topic
            # on a supervisor rejection — re-resolve the SAME requested topic.
            seed = get_seed(active, topic=topic)
            ok2, reason2 = judge_seed(seed, active)
            print(f"           → retry seed {'accepted' if ok2 else 'used anyway'} ({reason2})")
    except Exception as e:
        print(f"           ⚠ seed supervisor skipped (non-fatal): {e}")

    # Pre-analysis runs here so the hook angle is available for video selection
    seed_analysis = ""
    script_run_id = None
    try:
        seed_analysis, script_run_id, _ = preanalyze(seed)
    except Exception as e:
        print(f"           ⚠ Pre-analysis failed (non-fatal): {e}")

    # One human-readable folder per run (media_library/debug/<run_id>/) shared
    # by every stage — comfy_client's images/prompts, and the raw script +
    # pre-mix voiceover from audio_gen — always, not just RUFUS_DEBUG=1 runs
    # (the quality-review workflow needs every run's images/scripts logged,
    # not an opt-in subset). Env var, not a function param, so it reaches
    # every sub-module the same way RUFUS_CHANNEL already does.
    if script_run_id:
        os.environ["RUFUS_DEBUG_RUN_ID"] = script_run_id

    # Source resolution: explicit env > per-niche config > default "sd".
    # RUFUS_VIDEO_SOURCE=sd      → Stable Diffusion stills + Ken Burns (GPU), one
    #                              content-matched image per spoken beat (default).
    # RUFUS_VIDEO_SOURCE=pexels  → Pexels stock footage.
    video_source  = (os.environ.get("RUFUS_VIDEO_SOURCE")
                     or niche_cfg.get("video_source") or "sd").strip().lower()
    video_queries = _parse_video_queries(seed_analysis)

    # SD and diffusers GENERATE clips from the script, so they defer to step 2.5
    # (after the script exists). Only stock sources fetch up front.
    DEFERRED_SOURCES = ("sd", "diffusers", "comfy")

    # ── Step 2: Get candidate clips — generated (SD) or stock (Pexels) ──────────
    candidates = []
    scene = ""
    if video_source in DEFERRED_SOURCES:
        # Clips are generated AFTER the script is written (step 2.5) so they can
        # be tailored to the actual content. Placeholder scene gives write_script
        # context; candidates are filled in later.
        scene = niche_cfg.get("llava_context", f"{active} scene")
        run_progress.update(2, f"{video_source} mode — clips deferred until the script exists")
        print(f"[ 2 / 7 ]  {video_source} mode — clip generation deferred until after scripting\n")

    if video_source not in DEFERRED_SOURCES:
        run_progress.update(2, "fetching stock footage")
        print("[ 2 / 7 ]  Fetching candidate videos (parallel)...")
        try:
            if video_queries:
                print(f"           → script queries: {video_queries}")
            candidates = fetch_candidates(n=7, extra_keywords=video_queries or None)
            print(f"           → {len(candidates)} candidates downloaded\n")
        except Exception as e:
            print(f"           ✗ Step 2 failed: {e}")
            sys.exit(1)

    # ── Step 3: AI picks best video (stock only — generated clips are on-topic) ─
    if video_source in DEFERRED_SOURCES:
        print(f"[ 3 / 7 ]  {video_source} mode — skipping vision pick\n")
    elif scene:
        print("[ 3 / 7 ]  Generated clips are purpose-built — skipping vision pick\n")
    else:
        run_progress.update(3, "picking the best clip")
        print("[ 3 / 7 ]  AI selecting best video...")
        try:
            video_path, scene = pick_best_video(
                candidates, niche_cfg["llava_context"],
                seed=seed, analysis=seed_analysis or None,
            )
            print(f"           → selected: {video_path.name}")
            short = scene[:120] + "..." if len(scene) > 120 else scene
            print(f"           → {short}\n")
        except Exception as e:
            print(f"           ✗ Step 3 failed: {e}")
            sys.exit(1)

    # ── Step 4: Write script (reuses pre-analysis, no duplicate API call) ──────
    run_progress.update(4, "writing the script")
    print("[ 4 / 7 ]  Writing script with GPT...")
    try:
        # Escalating loop, not a single pass: when a cycle fails the fact gate
        # or misses the score bar, retry with a WHOLE new hook/angle (the
        # failure is fed forward) instead of redrafting the body under a hook
        # that's already lost. Bounded by RUFUS_SCRIPT_CYCLES/MAX_COST.
        result = write_script_until_good(scene, seed=seed,
                                         precomputed_analysis=seed_analysis or None,
                                         run_id=script_run_id)
        script = result["script"]

        if check_blacklist(script):
            print("           ⚠ Similar script already used – regenerating...")
            try:
                result = write_script(scene + " (make it different from previous versions)",
                                      seed=seed,
                                      precomputed_analysis=seed_analysis or None,
                                      run_id=script_run_id)
                script = result["script"]
            except Exception as _regen_err:
                print(f"           ⚠ Blacklist regen failed ({_regen_err}) — using original script")

        # Semantic near-duplicate gate: catches paraphrase-level repeats the
        # exact-match blacklist can't (different seeds, same underlying facts,
        # differently-worded script — still the same video to a viewer). One
        # regeneration with the objection fed back, same pattern as fact-gate.
        from script_writer import check_similarity, add_embedding
        is_dup, sim, script_vec = check_similarity(script, channel.id)
        if is_dup:
            print(f"           ⚠ Script is {sim:.0%} similar to a recent video — regenerating...")
            try:
                result = write_script(
                    scene + " (a recent video already covered this angle — take a "
                            "DIFFERENT angle: different hook, different examples, "
                            "different framing of the same facts)",
                    seed=seed, precomputed_analysis=seed_analysis or None,
                    run_id=script_run_id)
                script = result["script"]
                _, sim2, script_vec = check_similarity(script, channel.id)
                print(f"           → regenerated ({sim2:.0%} similar now)")
            except Exception as _sim_err:
                print(f"           ⚠ similarity regen failed ({_sim_err}) — using original script")

        # Topic-clustering gate: catches a script that's WORDED distinctly
        # (passes the similarity gate above) but keeps landing on the same
        # underlying topic within the last two weeks — e.g. three separately
        # written videos all about "compound interest". Time-windowed, not
        # count-windowed: the same topic is fair game again once it's stale.
        try:
            from script_writer import (extract_core_topic, check_topic_similarity,
                                       add_topic_embedding, TOPIC_WINDOW_DAYS)
            core_topic = extract_core_topic(seed_analysis)
            is_dup_topic, topic_sim, topic_vec = check_topic_similarity(core_topic, channel.id)
            if is_dup_topic:
                print(f"           ⚠ Topic \"{core_topic[:60]}\" covered "
                      f"{topic_sim:.0%} similarly in the last {TOPIC_WINDOW_DAYS} days — regenerating...")
                try:
                    result = write_script(
                        scene + f" (a recent video already covered this same core "
                                f"topic — \"{core_topic}\" — pick a genuinely DIFFERENT "
                                f"topic from the source, not just different wording)",
                        seed=seed, precomputed_analysis=seed_analysis or None,
                        run_id=script_run_id)
                    script = result["script"]
                    # Measure the REGENERATED script, not the frozen
                    # pre-analysis — seed_analysis doesn't change on regen, so
                    # re-extracting CORE from it would always re-measure the
                    # old topic and the gate could never verify the rewrite
                    # actually moved. The new script's hook line is the best
                    # cheap proxy for what topic it now covers.
                    new_hook = script.strip().split("\n")[0]
                    _, topic_sim2, topic_vec = check_topic_similarity(new_hook, channel.id)
                    print(f"           → regenerated (topic {topic_sim2:.0%} similar now)")
                    # The regenerated script also needs a fresh full-script
                    # embedding recorded — the one from the ORIGINAL draft
                    # (script_vec) no longer corresponds to what's shipping.
                    _, _, script_vec = check_similarity(script, channel.id)
                except Exception as _topic_err:
                    print(f"           ⚠ topic regen failed ({_topic_err}) — using original script")
            add_topic_embedding(topic_vec, channel.id)
        except Exception as e:
            print(f"           ⚠ topic-clustering gate skipped (non-fatal): {e}")

        # Full script, not a truncated preview — so the terminal itself is a
        # complete log of what's shipping, not just a teaser you have to open
        # a file to finish reading.
        print(f"           → {script}")
        print(f"           → score {result['score']}/10  attempts={result['attempts_used']}  "
              f"cost=${result['cost_usd']:.4f}\n")
        # One reviewable file per run (script now, image prompts at Step 2.5).
        if script_run_id:
            paths.write_run_report(
                script_run_id, script=script,
                meta={"niche": active, "channel": channel.id,
                      "score": f"{result.get('score', 0)}/10",
                      "fact_gate": "pass" if result.get("fact_ok", True) else
                                   f"FAIL — {result.get('fact_reason', '')}",
                      "attempts": result.get("attempts_used"),
                      "cost_usd": f"{result.get('cost_usd', 0):.4f}"})
    except Exception as e:
        print(f"           ✗ Step 4 failed: {e}")
        sys.exit(1)

    # Supervisor: factual-integrity gate — verify the script didn't contradict
    # or fabricate beyond its source (the prompt forbids it; this checks GPT
    # complied). One rewrite with the objection fed back; if the rewrite is
    # still flagged, render anyway but HOLD the upload for human review —
    # wrong facts must never publish themselves. RUFUS_SUPERVISOR=0 disables.
    facts_hold = None
    try:
        from supervisor import judge_script_facts
        ok_f, why_f = judge_script_facts(script, seed, niche_name=active, run_id=script_run_id)
        if not ok_f:
            print(f"           ⚠ fact-check flagged: {why_f} — rewriting once...")
            try:
                result = write_script(
                    scene + f" (FACTUAL CORRECTION REQUIRED — previous draft was "
                            f"rejected for: {why_f}. Stick strictly to the source.)",
                    seed=seed, precomputed_analysis=seed_analysis or None,
                    run_id=script_run_id)
                script = result["script"]
                # Re-embed the corrected script — the original draft's vector
                # no longer matches what will actually air. Recording happens
                # once, below, after this gate.
                _, _, script_vec = check_similarity(script, channel.id)
            except Exception as _fc_err:
                print(f"           ⚠ fact-fix rewrite failed ({_fc_err}) — keeping original")
            ok2, why2 = judge_script_facts(script, seed, niche_name=active, run_id=script_run_id)
            if not ok2:
                facts_hold = why2
                print(f"           ⚠ still flagged ({why2}) — upload will be HELD for review")
            else:
                print(f"           → rewrite passed fact-check ({why2})")
    except Exception as e:
        print(f"           ⚠ fact-check supervisor skipped (non-fatal): {e}")

    # Record dedup memory ONCE, after the LAST gate that can change the script
    # (the fact-check rewrite above). Recording before it stored blacklist
    # entries + embeddings for drafts that never aired — future scripts got
    # rejected as "similar to a recent video" against text no viewer ever saw.
    try:
        add_to_blacklist(script)
        add_embedding(script_vec, channel.id)
    except Exception as e:
        print(f"           ⚠ dedup-memory save failed (non-fatal): {e}")

    # ── Step 2.5: Generate one content-matched SD image per spoken beat ─────────
    # Each prompt depicts what the narrator says during that beat, in order, so
    # the renderer's sentence-boundary cuts keep the image tracking the voice-over.
    # Fallback chain so a render never dies:  comfy → sd → diffusers → pexels.
    if video_source in DEFERRED_SOURCES:
        run_progress.update(2, f"generating images for each beat ({video_source})")
        print(f"[ 2.5/7 ]  Generating clips from script content ({video_source})...")
        try:
            # One image per beat; SD_CLIPS (if set) caps the scene count.
            # 10 scenes over ~45s ≈ 3-4.5s per image — Shorts-pace cutting.
            # Sentence count naturally caps beats; SD_CLIPS env is the dial.
            max_scenes = int(os.environ.get("SD_CLIPS", "10"))
            prompts = _build_sd_prompts(script, active, max_scenes=max_scenes)
            print(f"           → {len(prompts)} beat-matched prompts:")
            for i, p in enumerate(prompts):
                print(f"             {i+1}. {p}")
            if script_run_id:
                rp = paths.write_run_report(script_run_id, prompts=prompts)
                if rp:
                    print(f"           → run report: {rp}")

            # Supervisor: catch prompt-builder drift (near-duplicates, off-topic
            # imagery) BEFORE burning FLUX/SD generation time on doomed images.
            # One retry only (fail-open, opt out with RUFUS_SUPERVISOR=0).
            try:
                from supervisor import judge_footage_prompts
                hook = script.strip().split("\n")[0]
                ok, reason = judge_footage_prompts(prompts, active, hook, run_id=script_run_id)
                if not ok:
                    print(f"           ⚠ supervisor rejected prompts ({reason}) — rewriting once...")
                    retry_prompts = _build_sd_prompts(script, active, max_scenes=max_scenes)
                    ok2, reason2 = judge_footage_prompts(retry_prompts, active, hook, run_id=script_run_id)
                    prompts = retry_prompts
                    print(f"           → retry prompts {'accepted' if ok2 else 'used anyway'} ({reason2})")
            except Exception as e:
                print(f"           ⚠ footage supervisor skipped (non-fatal): {e}")

            # Remember what this run is about to render so the NEXT run's
            # prompt-writer is told not to repeat these visual ideas.
            try:
                _remember_image_prompts(prompts)
            except Exception as e:
                print(f"           ⚠ image-prompt history save failed (non-fatal): {e}")

            if video_source == "comfy":
                # ComfyUI stills (best quality, needs ~24GB VRAM / RTX 3090) —
                # model is whatever's exported to config/stills_api.json
                # (Z-Image-Turbo recommended, Apache 2.0/commercial-safe).
                from comfy_client import generate_clips as comfy_generate
                candidates = comfy_generate(prompts, n=len(prompts), niche=active)
                if not candidates:
                    print("           ⚠ ComfyUI offline — trying A1111 SD...")
                    from sd_client import generate_clips as sd_generate
                    candidates = sd_generate(prompts, n=len(prompts), prebuilt=True)
                    if not candidates:
                        print("           ⚠ A1111 offline — trying diffusers in-process...")
                        try:
                            from diffusers_client import generate_clips as diffusers_generate
                            candidates = diffusers_generate(prompts)
                        except Exception as _diff_err:
                            print(f"           ⚠ diffusers also failed ({_diff_err})")
            elif video_source == "diffusers":
                from diffusers_client import generate_clips as diffusers_generate
                candidates = diffusers_generate(prompts)
            else:
                from sd_client import generate_clips as sd_generate
                candidates = sd_generate(prompts, n=len(prompts), prebuilt=True)

                # A1111 offline or returned nothing → try diffusers in-process
                if not candidates:
                    print("           ⚠ A1111 offline — trying diffusers in-process...")
                    try:
                        from diffusers_client import generate_clips as diffusers_generate
                        candidates = diffusers_generate(prompts)
                        if candidates:
                            print(f"           → diffusers fallback: {len(candidates)} clips ready\n")
                    except Exception as _diff_err:
                        print(f"           ⚠ diffusers also failed ({_diff_err})")

            if candidates:
                scene = f"{video_source}-generated: " + "; ".join(prompts[:2])
                print(f"           → {len(candidates)} clips ready\n")
            else:
                print("           ⚠ all SD sources failed — falling back to Pexels")
                video_source = "pexels"

            if not candidates:
                if video_queries:
                    print(f"           → using script queries: {video_queries}")
                candidates = fetch_candidates(n=5, extra_keywords=video_queries or None)
                print(f"           → Pexels fallback: {len(candidates)} clips\n")
        except Exception as e:
            print(f"           ✗ Clip generation failed: {e}")
            sys.exit(1)

    # ── Step 5: Render (all clips cut together) ─────────────────────────────────
    # RUFUS_RENDERER=remotion uses the React engine (spring-pop captions, smooth
    # crossfades, progress bar); anything else uses the FFmpeg engine. Remotion
    # failures fall back to FFmpeg so a render always completes.
    renderer = os.environ.get("RUFUS_RENDERER", "ffmpeg").strip().lower()
    run_progress.update(5, f"rendering the video ({renderer})")
    print(f"[ 5 / 7 ]  Rendering Short ({renderer})...")
    try:
        if renderer == "remotion":
            try:
                from remotion_renderer import render as remotion_render
                output_path = remotion_render(script, candidates, out_dir)
            except Exception as e:
                print(f"           ⚠ Remotion failed ({e}) — falling back to FFmpeg")
                output_path = render(script, candidates, out_dir)
        else:
            output_path = render(script, candidates, out_dir)
        print(f"           → {output_path}")
    except Exception as e:
        print(f"           ✗ Step 5 failed: {e}")
        sys.exit(1)

    # Automated output QC — is this file actually a publishable Short?
    # Criticals (no audio, wrong resolution, truncated encode) hold the upload;
    # warnings are printed so quality trends stay visible. Never fatal.
    qc = None
    try:
        from qc_check import run_qc, print_report
        try:
            import audio_gen as _ag
            _cuts = list(getattr(_ag, "LAST_CUTS", []) or [])
        except Exception:
            _cuts = []
        qc = run_qc(output_path, cuts=_cuts)
        print_report(qc)
        try:
            Path(str(output_path) + ".qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        except OSError:
            pass
    except Exception as e:
        print(f"           ⚠ QC skipped (non-fatal): {e}")
    print()

    # WHY the OLD auto-gate would (or wouldn't) have cleared this video — still
    # computed and shown to the human reviewer as a signal, even though NOTHING
    # auto-uploads anymore (see Step 7): a friend approving/rejecting in the
    # dashboard benefits from knowing "the gate would have held this for X".
    _hold_min_score = max(HARD_MIN_UPLOAD_SCORE,
                         int(os.environ.get("RUFUS_MIN_UPLOAD_SCORE",
                             str(channel.upload.get("min_score", 8)))))
    _hold_score = result.get("score", 0)
    if qc is not None and not qc.get("ok", True):
        hold_reason = f"QC failed: {'; '.join(qc['critical'])}"
    elif facts_hold:
        hold_reason = f"factual integrity: {facts_hold}"
    elif _hold_score < _hold_min_score:
        hold_reason = f"score {_hold_score}/10 < {_hold_min_score}/10 threshold"
    else:
        hold_reason = None

    # Metadata (title/description) + thumbnail — generated ALWAYS now, right
    # after render, so the approval queue has something to show/edit before
    # any upload decision exists (previously only built if the old auto-gate
    # already passed, i.e. never persisted for a held video at all).
    thumb_path = None
    meta = None
    if not skip_upload:
        try:
            from thumbnail_gen import make_thumbnail
            thumb_path = make_thumbnail(output_path, script, niche=active)
            print(f"           thumbnail: {thumb_path.name}")
        except Exception as e:
            print(f"           ⚠ thumbnail generation skipped: {e}")
        try:
            from youtube_uploader import build_metadata
            meta = build_metadata(script, active, niche_cfg)
        except Exception as e:
            print(f"           ⚠ metadata generation skipped: {e}")

    # ── Step 6: Save to DB ──────────────────────────────────────────────────────
    run_progress.update(6, "saving to the database")
    print("[ 6 / 7 ]  Saving to database...")
    db_id = None
    try:
        hook  = script.strip().split("\n")[0][:100]
        db_id = save_video(
            niche=active,
            script_hook=hook,
            scene_desc=scene[:500],
            video_file=str(output_path),
            script_full=script,
            seed_type=seed.get("type"),
            seed_source=seed.get("source"),
            seed_content=(seed.get("content", "") or "")[:1000],
            seed_url=seed.get("url") or None,
            run_id=result.get("run_id"),
            score=result.get("score", 0),
            criterion_scores=result.get("criterion_scores"),
            attempts_used=result.get("attempts_used"),
            final_temperature=result.get("final_temperature"),
            score_reasoning=(result.get("reasoning") or "")[:2000],
            channel=channel.id,
            hold_reason=hold_reason,
            title=(meta or {}).get("title"),
            description=(meta or {}).get("description"),
            upload_status="pending",
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 7: Review queue — NOTHING uploads without a human approving it in
    # the dashboard. RUFUS_AUTO_UPLOAD=1 is an explicit opt-out escape hatch
    # back to the old fully-automatic behavior (gated exactly as before), for
    # anyone who decides later they don't want a manual step after all.
    yt_url = None
    yt_id  = None
    auto_upload = os.environ.get("RUFUS_AUTO_UPLOAD", "0").strip().lower() in \
        ("1", "true", "yes", "on")
    min_score = _hold_min_score
    final_score = result.get("score", 0)

    if skip_upload:
        print("[ 7 / 7 ]  Upload skipped (--skip-upload)\n")
    elif not auto_upload:
        print(f"[ 7 / 7 ]  Queued for review (id={db_id}) — approve in the "
              f"dashboard to upload.")
        if hold_reason:
            print(f"           note for reviewer: the auto-gate would also "
                  f"have held this — {hold_reason}")
        print(f"           Video: {output_path}\n")
        # Ping the phone. An unattended 06:30 render is useless if nobody
        # knows it finished — the queue only works when you're told.
        try:
            import notify
            notify.notify_pending_review(
                title=(meta or {}).get("title") or script.strip().split("\n")[0][:80],
                score=result.get("score", 0), niche=active,
                video_id=db_id, hold_reason=hold_reason,
                video_path=output_path)
        except Exception as e:
            print(f"           ⚠ notification skipped (non-fatal): {e}")
    elif qc is not None and not qc.get("ok", True):
        print(f"[ 7 / 7 ]  Upload held — output failed QC: {'; '.join(qc['critical'])}")
        print(f"           Video saved for review: {output_path}\n")
    elif facts_hold:
        print(f"[ 7 / 7 ]  Upload held — factual integrity flag: {facts_hold}")
        print(f"           Verify against the source, then upload manually if it's fine.")
        print(f"           Video saved for review: {output_path}\n")
    elif final_score < min_score:
        print(f"[ 7 / 7 ]  Upload held — score {final_score}/10 < {min_score}/10 threshold.")
        print(f"           Video saved for review: {output_path}\n")
    else:
        print(f"[ 7 / 7 ]  Score {final_score}/10 ≥ {min_score} — "
              f"RUFUS_AUTO_UPLOAD=1, uploading...")
        try:
            from youtube_uploader import upload

            if db_id and meta and meta.get("title"):
                from db_manager import update_title
                update_title(db_id, meta["title"])

            yt_url, yt_id = upload(output_path, script, thumbnail_path=thumb_path,
                                   metadata=meta, source_url=seed.get("url") or None,
                                   seed_source=seed.get("source"))
            print(f"           → {yt_url}\n")

            if db_id and yt_id:
                try:
                    update_youtube_id(db_id, yt_id)
                    from db_manager import set_upload_status
                    set_upload_status(db_id, "approved")
                except Exception as e:
                    print(f"           ⚠ DB youtube_id update failed (video IS uploaded): {e}")
        except Exception as e:
            yt_url = None
            print(f"           ✗ Upload failed: {e}")
            print(f"           Video saved locally: {output_path} — check YouTube "
                  f"Studio before re-uploading (may have partially gone through)\n")
            if db_id:
                try:
                    from db_manager import mark_upload_failed
                    mark_upload_failed(db_id, str(e))
                except Exception as db_err:
                    # Losing the failure record silently means report.py
                    # undercounts failed uploads — say so, don't hide it.
                    print(f"           ⚠ could not record upload failure in DB: {db_err}")

    # ── Done ────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"{'='*52}")
    print(f"  Done in {elapsed:.0f}s  |  {active}")
    if yt_url:
        print(f"  YouTube: {yt_url}")
    print(f"  File:    {output_path}")
    print(f"{'='*52}\n")

    # Release the per-channel lock on NORMAL completion — critical for
    # --rotate, which calls run() again in the same process: filelock's
    # FileLock is not reentrant across instances, so without this the second
    # iteration sees its own predecessor's lock and dies with "another Rufus
    # run in progress" — --rotate could only ever produce its FIRST video.
    # Abnormal exits (sys.exit mid-run) terminate the whole process, where
    # the atexit-registered _release_lock covers it.
    _release_lock()
    run_progress.update(7, "queued for review")
    run_progress.finish("done")

    return {"video": str(output_path), "youtube_url": yt_url, "script": script, "seed": seed}


def _run_or_notify(niche: str | None, **kwargs) -> None:
    """Crash alert for every entry point, not just --scheduled
    (run_scheduled.bat already curls a raw ntfy alert on a non-zero exit
    code, but ONLY for that one entry point and ONLY via ntfy — a manual
    run gets nothing, and a Pushover/Telegram-only setup gets nothing even
    scheduled). Re-raises so the process still exits non-zero —
    run_scheduled.bat's own fallback stays as deliberate redundancy for the
    case Python itself never starts (broken venv, syntax error)."""
    try:
        run(niche_override=niche, **kwargs)
    except SystemExit as e:
        # A step called sys.exit(1). Without this the progress file stays
        # "running" forever and the dashboard shows a phantom active run.
        if getattr(e, "code", 0):
            run_progress.finish("failed", "a pipeline step exited early — see the run log")
        raise   # sys.exit(1) paths already print their own reason
    except KeyboardInterrupt:
        run_progress.finish("cancelled", "stopped by hand")
        raise
    except Exception as e:
        import traceback
        run_progress.finish("failed", str(e))
        try:
            import notify
            notify.notify_run_failed(f"{e}\n{traceback.format_exc()}",
                                     niche=niche, channel=kwargs.get("channel_id"))
        except Exception:
            pass   # never let a notification failure mask the real crash
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rufus pipeline runner")
    parser.add_argument("--skip-upload", action="store_true", help="Render only, skip YouTube upload")
    parser.add_argument("--niche",       type=str,            help="Override niche (e.g. finance, motivation, mindset)")
    parser.add_argument("--scheduled",   action="store_true", help="Use today's niche from the channel/config schedule (for cron)")
    parser.add_argument("--rotate",      action="store_true", help="Run one video per unique niche in the schedule")
    parser.add_argument("--output-dir",  type=str,            help="Directory to write rendered mp4 files (overrides RUFUS_OUTPUT_DIR env var)")
    parser.add_argument("--channel",     type=str,            help="Channel id from config/channels.json (default: default_channel / legacy)")
    parser.add_argument("--topic",       type=str,            help="Make a video about THIS topic instead of an auto-picked one (resolved to a real Wikipedia article, e.g. --topic \"Bretton Woods\")")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.niche, args.scheduled, args.rotate)) > 1:
        print("Use only one of --niche, --scheduled, --rotate")
        sys.exit(1)
    if args.topic and args.rotate:
        print("--topic makes one specific video — it can't be combined with --rotate")
        sys.exit(1)

    out_dir_arg = Path(args.output_dir) if args.output_dir else None

    # Channel schedule (if defined) takes precedence over niches.json schedule.
    def _channel_schedule() -> list[str]:
        try:
            from channel_config import load_channel
            ch = load_channel(args.channel)
            if ch.schedule:
                return ch.schedule
        except Exception:
            pass
        data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
        return data.get("schedule") or [data.get("active", "finance")]

    if args.rotate:
        seen: list[str] = []
        for n in _channel_schedule():
            if n not in seen:
                seen.append(n)
        print(f"\n[rotate] producing {len(seen)} video(s): {seen}\n")
        for n in seen:
            # Clear any prior env override so each iteration starts clean
            os.environ.pop("RUFUS_NICHE_OVERRIDE", None)
            _run_or_notify(n, skip_upload=args.skip_upload,
                          output_dir=out_dir_arg, channel_id=args.channel)
    elif args.scheduled:
        from datetime import datetime
        schedule = _channel_schedule()
        doy      = datetime.now().timetuple().tm_yday
        n        = schedule[(doy - 1) % len(schedule)]
        print(f"\n[scheduled] today's niche: {n}\n")
        # Scheduled triggers can overlap when a video takes 1.5-2h (full
        # motion) and slots are 3-4h apart — wait for the predecessor instead
        # of dying and silently dropping the slot. RUFUS_SCHED_LOCK_WAIT
        # (seconds) tunes the cap; default 3h.
        _sched_wait = float(os.environ.get("RUFUS_SCHED_LOCK_WAIT", str(3 * 3600)))
        _run_or_notify(n, skip_upload=args.skip_upload,
                      output_dir=out_dir_arg, channel_id=args.channel, topic=args.topic,
                      lock_wait=_sched_wait)
    else:
        _run_or_notify(args.niche, skip_upload=args.skip_upload,
                      output_dir=out_dir_arg, channel_id=args.channel, topic=args.topic)
