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
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
OUTPUT_DIR  = Path(os.environ.get("RUFUS_OUTPUT_DIR", ROOT / "media_library" / "output"))
LOG_DIR     = ROOT / "logs"


# ── Single-instance lock (cron overlap protection) ───────────────────────────────
# Cross-platform (Windows + Linux + macOS) via filelock — POSIX os.kill(pid, 0)
# raises on Windows. An OS advisory lock held for the life of the process means a
# second overlapping run simply can't acquire it.

from filelock import FileLock, Timeout

_INSTANCE_LOCK = None   # created per-run by _acquire_lock (per-CHANNEL lock file)


def _acquire_lock(channel_id: str = "main_en") -> None:
    """Refuse to start if another run of the SAME channel is alive.

    Per-channel, not global: the corruption risk a lock protects against —
    clashing temp files and double DB writes for one channel's video — only
    exists within a channel. Different channels are safe to run concurrently
    (ComfyUI queues their GPU jobs, used_seeds.json has its own lock, SQLite
    serializes writers), and multi-channel scaling requires it."""
    global _INSTANCE_LOCK
    _INSTANCE_LOCK = FileLock(str(ROOT / f"rufus.{channel_id}.lock") + ".lock")
    try:
        _INSTANCE_LOCK.acquire(timeout=0)   # non-blocking
    except Timeout:
        print(f"ERROR: another Rufus run for channel '{channel_id}' is in progress "
              f"(lock held: rufus.{channel_id}.lock.lock). "
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
        d = ROOT / "media_library" / "temp" / sub
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
    media_root = ROOT / "media_library"
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
                  max_debug_days: int = 30) -> None:
    """Delete old logs and stale cache/temp/debug media. Cheap, runs every start.

    Debug gets its own, longer window (~a month, not 14 days) — RUFUS_DEBUG
    output (script/voiceover/keyframes per run) is for reviewing what a run
    actually produced, so it's worth keeping around longer than throwaway
    cache/temp, but still needs a cap or it grows forever."""
    cutoff_logs  = time.time() - max_log_days * 86400
    cutoff_cache = time.time() - max_cache_days * 86400
    cutoff_debug = time.time() - max_debug_days * 86400
    debug_dir    = ROOT / "media_library" / "debug"
    removed = 0
    for d, cutoff in (
        (LOG_DIR, cutoff_logs),
        (LOG_DIR / "scripts", cutoff_logs),
        (ROOT / "media_library" / "cache", cutoff_cache),
        (ROOT / "media_library" / "temp", cutoff_cache),
        (ROOT / "media_library" / "music", cutoff_cache),
        (debug_dir, cutoff_debug),
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
    # Debug files live in per-run subfolders (media_library/debug/<run_id>/) —
    # clean up any now-empty ones the file pass above leaves behind.
    if debug_dir.exists():
        for sub in debug_dir.iterdir():
            try:
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            except OSError:
                continue
    if removed:
        print(f"[maint] cleaned {removed} stale file(s)")


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
from script_writer   import write_script, preanalyze, check_blacklist, add_to_blacklist
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
        data = json.loads(RECENT_PROMPTS_FILE.read_text())
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
            data = json.loads(RECENT_PROMPTS_FILE.read_text())
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
        RECENT_PROMPTS_FILE.write_text(json.dumps(data, indent=2))
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
        niche_data  = json.loads(NICHES_FILE.read_text())
        niche_style = niche_data["niches"].get(niche, {}).get("style_suffix", "")
    except Exception:
        niche_style = ""
    color_grade = niche_style or "cinematic color grade, muted tones, film grain"

    def _fallback_prompt(i: int, beat: str) -> str:
        a   = _SD_ANCHORS[i % len(_SD_ANCHORS)]
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
            key = json.loads(keys_file.read_text()).get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            raise RuntimeError(
                "OpenAI key missing — SD prompt generation requires GPT-4o-mini.\n"
                "Add 'openai' key to config/keys.json or use RUFUS_VIDEO_SOURCE=pexels."
            )
        beat_lines = "\n".join(
            f"  Beat {i+1} (CAMERA={_SD_ANCHORS[i % len(_SD_ANCHORS)]['camera'].split(',')[0]}): "
            f"\"{b}\""
            for i, b in enumerate(beats)
        )
        anchor_lines = "\n".join(
            f"  Beat {i+1}: framing={_SD_ANCHORS[i % len(_SD_ANCHORS)]['subject_hint']}; "
            f"lighting={_SD_ANCHORS[i % len(_SD_ANCHORS)]['light']}; "
            f"lens={_SD_ANCHORS[i % len(_SD_ANCHORS)]['camera']}"
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
                _nd = json.loads(NICHES_FILE.read_text())
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
            "RULES:\n"
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
            "- PERIOD ACCURACY: zero anachronisms — no modern objects, clothing, logos, "
            "screens, or writing in a historical scene. Show present-day items only if "
            "the beat is explicitly about today.\n"
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
            "- PHOTOREALISM, NOT ILLUSTRATION: this must read as an actual photograph — "
            "never a painting, digital art, 3D render, or CGI. Name a specific real "
            "camera + lens for the shot (e.g. 'shot on a Leica M6, 35mm f/2', "
            "'85mm f/1.4 portrait lens, shallow depth of field, soft background bokeh') "
            "so the optics feel real, not synthetic. Include natural imperfection: "
            "visible film grain, natural skin texture (pores, not airbrushed or waxy), "
            "authentic period-accurate wear and grime on objects/clothing, realistic "
            "uneven lighting rather than a flat studio look. Avoid oversaturated colors, "
            "plastic-looking skin, or an overly clean/staged composition.\n"
            "- Apply this exact color grade to EVERY prompt: "
            f"{color_grade}.\n"
            "- MAKE THE FRAME ALIVE — every image must contain a story, not a catalog "
            "shot: something mid-action or freshly happened (hands mid-motion, a crowd "
            "blurred in a rush, smoke still rising, ink still wet, a chair just pushed "
            "back), atmosphere the viewer can feel (dust motes in a shaft of light, "
            "rain-slick cobblestones, steam off a cup, harsh low winter sun), and "
            "cinematic light with real contrast (chiaroscuro, a single lamp in the dark, "
            "golden hour through a window) — never flat, even lighting. A static object "
            "centered on a plain surface is a FAILURE; give the subject context, scale, "
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
            "- NO on-screen text, captions, watermarks, or written numbers in the image "
            "(Rufus overlays its own captions).\n"
            "- All prompts must be visually distinct.\n"
            f"{fresh_block}\n"
            f"Output EXACTLY {n} prompts, one per line. No numbering, no labels, no blank lines."
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
                    "• EMOTION SECOND: once the literal subject is locked, add real emotion through "
                    "posture, expression, lighting — never a neutral or smiling pose, never looking at camera.\n"
                    "• BEAT 1 = the scroll-stopper: highest contrast, most arresting framing of the "
                    "literal subject, most emotionally charged — the frame that makes someone stop scrolling.\n"
                    "• SETTING must be lived-in and specific: scattered papers, cold monitor glow, "
                    "coffee rings, worn upholstery, fingerprints on glass. Never clean and generic.\n"
                    "• NO quality tokens (8k, masterpiece, best quality) — those are appended separately.\n"
                    f"• All {n} prompts must be completely distinct: different subjects, settings, framings.\n"
                    "• BAN: posed, smiling at camera, corporate handshake, generic silhouette, "
                    "stock photo, perfect unblemished skin, catalog pose.\n"
                    "• STORY THREAD: the frames together should read as ONE unfolding story "
                    "(setup → pressure → turning point → aftermath) with people mid-action "
                    "driving it — object-only shots max 2 of the batch, and only with active "
                    "consequence, never a display shot.\n"
                    f"{fresh_block}\n"
                    "90–110 words per prompt. Dense comma-separated tokens only — zero full sentences.\n\n"
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
        lines = [_strip_beat_echo(l, beats[i]) if i < len(beats) else l
                 for i, l in enumerate(lines)]

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
    data = json.loads(NICHES_FILE.read_text())
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
    data     = json.loads(NICHES_FILE.read_text())
    schedule = data.get("schedule") or [data.get("active", "finance")]
    doy      = datetime.now().timetuple().tm_yday   # 1-366
    return schedule[(doy - 1) % len(schedule)]


def _all_scheduled_niches() -> list[str]:
    """Return unique niches present in schedule, preserving order."""
    data     = json.loads(NICHES_FILE.read_text())
    schedule = data.get("schedule") or [data.get("active", "finance")]
    seen     = []
    for n in schedule:
        if n not in seen:
            seen.append(n)
    return seen


def run(skip_upload: bool = False, niche_override: str = None, output_dir: Path = None,
        channel_id: str = None, topic: str = None):
    # Channel resolution FIRST (read-only) so the instance lock can be
    # per-channel — see _acquire_lock. Legacy installs without channels.json
    # get a synthesized "main_en" channel — behavior unchanged.
    from channel_config import load_channel
    channel = load_channel(channel_id)

    _acquire_lock(channel.id)
    import atexit
    atexit.register(_release_lock)   # release on any exit path (idempotent)
    atexit.register(_sweep_run_temp) # this run's clip temps never orphan
    os.environ["RUFUS_CHANNEL"] = channel.id          # sub-modules inherit it
    if channel.voice:
        os.environ.setdefault("RUFUS_EDGE_VOICE", channel.voice)

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

    # Debug mode: one human-readable folder per run (media_library/debug/<run_id>/)
    # shared by every stage — comfy_client's images/prompts, and now the raw
    # script + pre-mix voiceover from audio_gen — instead of each stage picking
    # its own timestamp. Env var, not a function param, so it reaches every
    # sub-module the same way RUFUS_CHANNEL already does.
    if os.environ.get("RUFUS_DEBUG") and script_run_id:
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
        print(f"[ 2 / 7 ]  {video_source} mode — clip generation deferred until after scripting\n")

    if video_source not in DEFERRED_SOURCES:
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
    print("[ 4 / 7 ]  Writing script with GPT...")
    try:
        result = write_script(scene, seed=seed,
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

        preview = script[:100] + "..." if len(script) > 100 else script
        print(f"           → {preview}")
        print(f"           → score {result['score']}/10  attempts={result['attempts_used']}  "
              f"cost=${result['cost_usd']:.4f}\n")
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
        print(f"[ 2.5/7 ]  Generating clips from script content ({video_source})...")
        try:
            # One image per beat; SD_CLIPS (if set) caps the scene count.
            # 10 scenes over ~45s ≈ 3-4.5s per image — Shorts-pace cutting.
            # Sentence count naturally caps beats; SD_CLIPS env is the dial.
            max_scenes = int(os.environ.get("SD_CLIPS", "10"))
            prompts = _build_sd_prompts(script, active, max_scenes=max_scenes)
            print(f"           → {len(prompts)} beat-matched prompts:")
            for i, p in enumerate(prompts):
                print(f"             {i+1}. {p[:90]}")

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
                # ComfyUI + FLUX.1-dev (best quality, needs ~24GB VRAM / RTX 3090).
                from comfy_client import generate_clips as comfy_generate
                candidates = comfy_generate(prompts, n=len(prompts), niche_cfg=niche_cfg)
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
        qc = run_qc(output_path)
        print_report(qc)
        try:
            Path(str(output_path) + ".qc.json").write_text(json.dumps(qc, indent=2))
        except OSError:
            pass
    except Exception as e:
        print(f"           ⚠ QC skipped (non-fatal): {e}")
    print()

    # WHY the OLD auto-gate would (or wouldn't) have cleared this video — still
    # computed and shown to the human reviewer as a signal, even though NOTHING
    # auto-uploads anymore (see Step 7): a friend approving/rejecting in the
    # dashboard benefits from knowing "the gate would have held this for X".
    _hold_min_score = int(os.environ.get("RUFUS_MIN_UPLOAD_SCORE",
                          str(channel.upload.get("min_score", 8))))
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
            thumb_path = make_thumbnail(output_path, script)
            print(f"           thumbnail: {thumb_path.name}")
        except Exception as e:
            print(f"           ⚠ thumbnail generation skipped: {e}")
        try:
            from youtube_uploader import build_metadata
            meta = build_metadata(script, active, niche_cfg)
        except Exception as e:
            print(f"           ⚠ metadata generation skipped: {e}")

    # ── Step 6: Save to DB ──────────────────────────────────────────────────────
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
                                   metadata=meta)
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

    return {"video": str(output_path), "youtube_url": yt_url, "script": script, "seed": seed}


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
        data = json.loads(NICHES_FILE.read_text())
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
            run(skip_upload=args.skip_upload, niche_override=n,
                output_dir=out_dir_arg, channel_id=args.channel)
    elif args.scheduled:
        from datetime import datetime
        schedule = _channel_schedule()
        doy      = datetime.now().timetuple().tm_yday
        n        = schedule[(doy - 1) % len(schedule)]
        print(f"\n[scheduled] today's niche: {n}\n")
        run(skip_upload=args.skip_upload, niche_override=n,
            output_dir=out_dir_arg, channel_id=args.channel, topic=args.topic)
    else:
        run(skip_upload=args.skip_upload, niche_override=args.niche,
            output_dir=out_dir_arg, channel_id=args.channel, topic=args.topic)
