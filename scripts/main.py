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
LOCK_FILE   = ROOT / "rufus.lock"


# ── Single-instance lock (cron overlap protection) ───────────────────────────────

def _acquire_lock() -> None:
    """Refuse to start if another Rufus run is alive (overlapping cron + manual
    runs corrupt temp files and double-write the DB). Stale locks self-clear."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
        except (ValueError, OSError):
            pid = -1
        if pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, 0)   # signal 0 = existence check only
                print(f"ERROR: another Rufus run (pid {pid}) is in progress. "
                      f"Wait for it, or delete {LOCK_FILE} if it crashed.")
                sys.exit(1)
            except ProcessLookupError:
                pass              # stale lock from a dead process — take over
            except PermissionError:
                print(f"ERROR: pid {pid} exists (no permission to signal) — assuming live run.")
                sys.exit(1)
    LOCK_FILE.write_text(str(os.getpid()))


def _release_lock() -> None:
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except OSError:
        pass


# ── Housekeeping (disk + logs never grow unbounded) ──────────────────────────────

def _housekeeping(max_log_days: int = 90, max_cache_days: int = 14) -> None:
    """Delete old logs and stale cache/temp media. Cheap, runs every start."""
    cutoff_logs  = time.time() - max_log_days * 86400
    cutoff_cache = time.time() - max_cache_days * 86400
    removed = 0
    for d, cutoff in (
        (LOG_DIR, cutoff_logs),
        (LOG_DIR / "scripts", cutoff_logs),
        (ROOT / "media_library" / "cache", cutoff_cache),
        (ROOT / "media_library" / "temp", cutoff_cache),
        (ROOT / "media_library" / "music", cutoff_cache),
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


# Per-beat camera anchors — rotate so consecutive scenes read like film coverage
# (macro → establishing → human → overhead) instead of four identical framings.
_SD_ANCHORS = [
    {
        "camera": "EXTREME CLOSE-UP macro, Canon 100mm f/2.8L Macro, f/2.8, razor-thin depth of field",
        "subject_hint": "tight detail on a face, hands, or object surface filling the frame",
        "light": "single hard tungsten light at 45°, deep chiaroscuro, inky shadows, specular rim highlight",
    },
    {
        "camera": "WIDE ESTABLISHING panorama, Sony FE 24mm f/1.4 GM, f/8, deep focus, circular polarizer",
        "subject_hint": "subject small against a vast environment, strong leading lines converging on subject",
        "light": "golden hour natural light, long warm directional shadows, atmospheric haze, amber sky",
    },
    {
        "camera": "MEDIUM SHOT portrait, NIKKOR Z 85mm f/1.4, f/1.8, selective focus on face",
        "subject_hint": "waist-up, subject off-center left, environmental context right, mid-motion or reaction",
        "light": "3-point lighting: soft-box key 45° camera-left, 2:1 fill right, warm rim light from behind",
    },
    {
        "camera": "AERIAL overhead flat-lay nadir, DJI Mavic 3 Pro 24mm, f/5.6, symmetrical composition",
        "subject_hint": "bird's-eye view, geometric pattern or arranged objects, minimalist negative space",
        "light": "diffused even overhead daylight, soft shadows revealing texture, no harsh highlights",
    },
]


def _split_beats(script: str, max_scenes: int = 6, min_words: int = 3) -> list[str]:
    """Split a script into ordered visual beats (one per spoken sentence).

    Merges fragments shorter than ``min_words`` into the previous beat, then
    collapses the shortest adjacent beats until at most ``max_scenes`` remain —
    so each beat is a meaningful chunk of speech that earns its own image.
    Order is preserved, which is what lets clip[i] line up with beat[i] at
    render time (audio_gen cuts on sentence boundaries in list order).
    """
    import re
    raw = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", script.strip()) if s.strip()]
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


def _build_sd_prompts(script: str, niche: str, max_scenes: int = 6) -> list[str]:
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
            "8k uhd, sharp focus, professional editorial photography, film grain"
        )

    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        if keys_file.exists():
            key = json.loads(keys_file.read_text()).get("openai", "")
            if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
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
                client = OpenAI(api_key=key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            "You are an elite Stable Diffusion prompt engineer specializing in "
                            "Realistic Vision v5.1 (ultra-photorealistic checkpoint). "
                            f"Write EXACTLY {n} image prompts for a {niche} YouTube Short — one per beat.\n\n"
                            "SPOKEN BEATS — prompt N MUST show what the narrator says during beat N:\n"
                            f"{beat_lines}\n\n"
                            "PER-BEAT CAMERA/FRAMING — use these exact specs for each slot:\n"
                            f"{anchor_lines}\n\n"
                            "TOKEN FORMAT (mandatory for RV5.1 — pure comma-separated tokens, NO sentences):\n"
                            "RAW photo, (SUBJECT:1.35), SETTING TEXTURE, COMPOSITION, LIGHTING, CAMERA+LENS, COLOR GRADE\n\n"
                            "RULES:\n"
                            "• Every prompt MUST start with 'RAW photo,' — it is the RV5.1 quality activator.\n"
                            "• SUBJECT = the literal thing the narrator mentions, ultra-specific: "
                            "'investor' → '(weathered 52yo man, salt-and-pepper stubble, 3am under-eye shadows, "
                            "rumpled charcoal suit, loosened tie:1.35)'. Clothes, age, expression, skin texture.\n"
                            "• SETTING = physical texture detail: 'office' → "
                            "'glass-walled 40th-floor corner office, city lights blurred below, "
                            "scattered papers, cold blue monitor glow on face'.\n"
                            "• LIGHTING = named setup only: 'single overhead tungsten key 45°, "
                            "deep shadow fill, specular rim on shoulder edge'. Never just 'dramatic'.\n"
                            f"• COLOR GRADE on every prompt: {color_grade}.\n"
                            "• All {n} subjects and locations must be completely distinct — no repeats.\n"
                            "• DO NOT add quality tags (8k, masterpiece, etc.) — those are appended separately.\n"
                            "• 55–70 words per prompt. Dense SD tokens only.\n\n"
                            f"Output EXACTLY {n} lines. No numbering, no labels, no blank lines. Beat order."
                        ),
                    }],
                    max_tokens=1100,
                    temperature=0.85,
                )
                raw_lines = resp.choices[0].message.content.strip().split("\n")
                lines = [re.sub(r"^[\d\.\-\)\s]+", "", l).strip()
                         for l in raw_lines if l.strip()]
                lines = [l for l in lines if len(l) > 20]

                # Pad/realign so we always return exactly one prompt per beat, in order.
                out = []
                for i in range(n):
                    out.append(lines[i] if i < len(lines) else _fallback_prompt(i, beats[i]))
                return out
    except Exception as e:
        print(f"[sd] GPT prompt generation skipped ({e}) — using beat fallback")

    # Key-free fallback: one anchored prompt per beat, content-matched and distinct.
    return [_fallback_prompt(i, b) for i, b in enumerate(beats)]


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
        channel_id: str = None):
    _acquire_lock()
    import atexit
    atexit.register(_release_lock)   # release on any exit path (idempotent)

    # Channel resolution (channel-in-a-box). Legacy installs without
    # channels.json get a synthesized "main_en" channel — behavior unchanged.
    from channel_config import load_channel
    channel = load_channel(channel_id)
    os.environ["RUFUS_CHANNEL"] = channel.id          # sub-modules inherit it
    if channel.voice:
        os.environ.setdefault("RUFUS_EDGE_VOICE", channel.voice)

    log_path = _enable_file_logging()
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
    print("[ 1 / 7 ]  Researching real source material...")
    try:
        seed = get_seed(active)
        if seed.get("type") == "reddit":
            print(f"           → Reddit: {seed.get('title', '')[:80]}")
        elif seed.get("type") == "hackernews":
            print(f"           → HN:     {seed.get('title', '')[:80]}")
        elif seed.get("type") == "stackexchange":
            print(f"           → SE:     {seed.get('title', '')[:80]}")
        else:
            print(f"           → Quote:  \"{seed.get('content', '')[:80]}\" — {seed.get('source')}\n")
    except Exception as e:
        print(f"           ✗ Step 1 failed: {e}")
        sys.exit(1)

    # Pre-analysis runs here so the hook angle is available for video selection
    seed_analysis = ""
    script_run_id = None
    try:
        seed_analysis, script_run_id, _ = preanalyze(seed)
    except Exception as e:
        print(f"           ⚠ Pre-analysis failed (non-fatal): {e}")

    # Source resolution: explicit env > per-niche config > default "sd".
    # RUFUS_VIDEO_SOURCE=sd      → Stable Diffusion stills + Ken Burns (GPU), one
    #                              content-matched image per spoken beat (default).
    # RUFUS_VIDEO_SOURCE=pexels  → Pexels stock footage.
    video_source  = (os.environ.get("RUFUS_VIDEO_SOURCE")
                     or niche_cfg.get("video_source") or "sd").strip().lower()
    video_queries = _parse_video_queries(seed_analysis)

    # SD GENERATES clips from the script, so it defers to step 2.5 (after the
    # script exists). Only stock sources fetch up front.
    DEFERRED_SOURCES = ("sd",)

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

        add_to_blacklist(script)
        preview = script[:100] + "..." if len(script) > 100 else script
        print(f"           → {preview}")
        print(f"           → score {result['score']}/10  attempts={result['attempts_used']}  "
              f"cost=${result['cost_usd']:.4f}\n")
    except Exception as e:
        print(f"           ✗ Step 4 failed: {e}")
        sys.exit(1)

    # ── Step 2.5: Generate one content-matched SD image per spoken beat ─────────
    # Each prompt depicts what the narrator says during that beat, in order, so
    # the renderer's sentence-boundary cuts keep the image tracking the voice-over.
    # Fallback chain so a render never dies:  sd → pexels.
    if video_source in DEFERRED_SOURCES:
        print(f"[ 2.5/7 ]  Generating SD clips from script content...")
        try:
            # One image per beat; SD_CLIPS (if set) caps the scene count.
            max_scenes = int(os.environ.get("SD_CLIPS", "6"))
            prompts = _build_sd_prompts(script, active, max_scenes=max_scenes)
            print(f"           → {len(prompts)} beat-matched prompts:")
            for i, p in enumerate(prompts):
                print(f"             {i+1}. {p[:90]}")

            from sd_client import generate_clips as sd_generate
            candidates = sd_generate(prompts, n=len(prompts), prebuilt=True)
            if candidates:
                scene = "SD-generated: " + "; ".join(prompts[:2])
                print(f"           → {len(candidates)} clips ready\n")
            else:
                print("           ⚠ SD failed — falling back to Pexels")
                video_source = "pexels"

            if not candidates:
                if video_queries:
                    print(f"           → using script queries: {video_queries}")
                candidates = fetch_candidates(n=5, extra_keywords=video_queries or None)
                video_path, scene = pick_best_video(
                    candidates, niche_cfg["llava_context"],
                    seed=seed, analysis=seed_analysis or None,
                )
                print(f"           → Pexels fallback: {video_path.name}\n")
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
        print(f"           → {output_path}\n")
    except Exception as e:
        print(f"           ✗ Step 5 failed: {e}")
        sys.exit(1)

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
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 7: Upload (with custom thumbnail) ─────────────────────────────────
    # Quality gate: only auto-upload videos whose script cleared the bar. A weak
    # script never reaches YouTube — it's saved locally for review instead.
    yt_url = None
    yt_id  = None   # guard: upload() may not be reached if quality gate holds
    min_score = int(os.environ.get("RUFUS_MIN_UPLOAD_SCORE",
                                   str(channel.upload.get("min_score", 8))))
    final_score = result.get("score", 0)
    if skip_upload:
        print("[ 7 / 7 ]  Upload skipped (--skip-upload)\n")
    elif final_score < min_score:
        print(f"[ 7 / 7 ]  Upload held — score {final_score}/10 < {min_score}/10 threshold.")
        print(f"           Video saved for review: {output_path}\n")
    else:
        print(f"[ 7 / 7 ]  Score {final_score}/10 ≥ {min_score} — generating thumbnail + uploading...")
        try:
            from thumbnail_gen    import make_thumbnail
            from youtube_uploader import upload, build_metadata

            thumb = None
            try:
                thumb = make_thumbnail(output_path, script)
                print(f"           thumbnail: {thumb.name}")
            except Exception as e:
                print(f"           ⚠ thumbnail generation skipped: {e}")

            # GPT title/description once here, persisted for CTR learning
            meta = None
            try:
                meta = build_metadata(script, active, niche_cfg)
                if db_id and meta.get("title"):
                    from db_manager import update_title
                    update_title(db_id, meta["title"])
            except Exception as e:
                print(f"           ⚠ metadata pre-build failed (uploader will retry): {e}")

            yt_url, yt_id = upload(output_path, script, thumbnail_path=thumb, metadata=meta)
            print(f"           → {yt_url}\n")

            if db_id and yt_id:
                try:
                    update_youtube_id(db_id, yt_id)
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
                except Exception:
                    pass

    # ── Done ────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"{'='*52}")
    print(f"  Done in {elapsed:.0f}s  |  {active}")
    if yt_url:
        print(f"  YouTube: {yt_url}")
    print(f"  File:    {output_path}")
    print(f"{'='*52}\n")

    return {"video": str(output_path), "youtube_url": yt_url, "script": script, "seed": seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rufus pipeline runner")
    parser.add_argument("--skip-upload", action="store_true", help="Render only, skip YouTube upload")
    parser.add_argument("--niche",       type=str,            help="Override niche (e.g. finance, motivation, mindset)")
    parser.add_argument("--scheduled",   action="store_true", help="Use today's niche from the channel/config schedule (for cron)")
    parser.add_argument("--rotate",      action="store_true", help="Run one video per unique niche in the schedule")
    parser.add_argument("--output-dir",  type=str,            help="Directory to write rendered mp4 files (overrides RUFUS_OUTPUT_DIR env var)")
    parser.add_argument("--channel",     type=str,            help="Channel id from config/channels.json (default: default_channel / legacy)")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.niche, args.scheduled, args.rotate)) > 1:
        print("Use only one of --niche, --scheduled, --rotate")
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
            output_dir=out_dir_arg, channel_id=args.channel)
    else:
        run(skip_upload=args.skip_upload, niche_override=args.niche,
            output_dir=out_dir_arg, channel_id=args.channel)
