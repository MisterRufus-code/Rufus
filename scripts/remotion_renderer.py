#!/usr/bin/env python3
"""
remotion_renderer.py — Render Rufus Shorts with Remotion (React-based engine).

Produces the same deliverable as audio_gen.render() — a 1080x1920 mp4 with
voice, word-pop captions, crossfades, Ken Burns, ducked music — but rendered
through Remotion for smoother, spring-physics animations and a retention
progress bar. CPU-only (headless Chrome), no GPU needed.

Enable with:
    export RUFUS_RENDERER=remotion
First-time setup:
    cd remotion && npm install

Reuses audio_gen for TTS, Whisper word timestamps, and music fetching, so the
audio path is identical between engines. Any failure here should be caught by
the caller, which falls back to the FFmpeg renderer.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import audio_gen  # same-directory import: TTS, whisper, music, duration logic
import video_format as _fmt

ROOT         = Path(__file__).parent.parent
REMOTION_DIR = ROOT / "remotion"
PUBLIC_DIR   = REMOTION_DIR / "public"

RENDER_TIMEOUT = 1800  # 30 min ceiling — CPU renders of 60s shorts stay well under


def _npx() -> str:
    """The npx executable to spawn, as an absolute path.

    On Windows npx is `npx.cmd`. `shutil.which` finds it — it consults PATHEXT —
    but CreateProcess does not, so passing the bare string "npx" to
    subprocess.run raises `[WinError 2] The system cannot find the file
    specified` even though the readiness check just passed. That is exactly how
    every render on the owner's box failed straight into the FFmpeg fallback:
    Remotion was installed and working the whole time, and the log line
    ("Remotion failed … falling back to FFmpeg") read like a Remotion problem.

    Spawning the resolved absolute path avoids the lookup entirely, on every OS.
    """
    found = shutil.which("npx")
    if found is None:
        raise RuntimeError("Node.js not installed (npx not found) — install Node.js "
                           "(Windows: https://nodejs.org, Linux: apt install nodejs npm)")
    return found


def _check_ready() -> None:
    _npx()
    if not (REMOTION_DIR / "node_modules").exists():
        raise RuntimeError(f"Remotion deps missing — run: cd {REMOTION_DIR} && npm install")


def _insert_style() -> str:
    """The channel's look for inserts — see insert_director.style_suffix.

    Both renderers ask the same function rather than each holding a copy, so
    the two paths cannot drift into two looks.
    """
    try:
        import insert_director
        return insert_director.style_suffix()
    except Exception:
        return ""


def _probe_duration(path: Path) -> float | None:
    """Clip duration in seconds via ffprobe, or None if probing fails."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def render(script: str, bg_paths: "Path | list[Path]", out_dir: Path,
           music_path: Path | None = None,
           voice_path: Path | None = None) -> Path:
    """Render the video. `voice_path` reuses an existing voiceover instead of
    synthesizing a new one.

    WHY REUSING THE VOICE MATTERS. A re-cut exists to change which picture is
    on screen, not when. Synthesizing again would produce audio a few
    milliseconds different, Whisper would time the new audio, and every cut in
    the video would move — so "I redrew beat 7" would silently reshuffle the
    other nine. Handed the same bytes, the transcription is the same and the
    cuts land exactly where they did before.
    """
    if isinstance(bg_paths, Path):
        bg_paths = [bg_paths]
    bg_paths = [Path(p) for p in bg_paths if Path(p).exists()]
    if not bg_paths:
        raise FileNotFoundError("No valid background video files found")

    _check_ready()

    niche_name = audio_gen._active_niche_name()
    if music_path is None:
        try:
            music_path = audio_gen._fetch_music(niche_name)
        except Exception as e:
            print(f"[music] {niche_name} mood fetch failed: {e}")
            music_path = None

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp   = int(time.time())
    job     = f"job_{stamp}"
    job_dir = PUBLIC_DIR / job
    job_dir.mkdir(parents=True, exist_ok=True)

    mp3 = job_dir / "voice.mp3"
    out = out_dir / f"short_{stamp}.mp4"

    try:
        if voice_path is not None and Path(voice_path).exists():
            print(f"[1/4] Reusing voice ({Path(voice_path).name})…")
            shutil.copy2(str(voice_path), str(mp3))
        else:
            print("[1/4] Generating voice…")
            audio_gen._tts(script, mp3)

        print("[2/4] Transcribing…")
        # Go through _transcribe, never the model object directly — the
        # CUDA→CPU fallback lives in that wrapper. Live cost of
        # skipping it: a box without cublas64_12.dll raised here, which aborted
        # the WHOLE Remotion render, fell back to FFmpeg, and re-ran the voice
        # from scratch — paying for a Kokoro load twice over a missing DLL the
        # FFmpeg path had already been shrugging off for weeks.
        segs, _ = audio_gen._transcribe(mp3)
        segments = list(segs)
        if not segments:
            raise RuntimeError("Whisper produced no segments")

        audio_dur = max((w.end for seg in segments for w in seg.words), default=0)
        audio_dur = min(max(audio_dur + 0.3, 1.0), audio_gen.MAX_DUR)
        if audio_dur < audio_gen.MIN_DUR:
            print(f"      ⚠ audio is only {audio_dur:.1f}s (target ≥{audio_gen.MIN_DUR:.0f}s)")

        # THE SAME CAPTIONS THE FFMPEG PATH BUILDS. Both renderers ship the
        # same channel, so the grouping and the casing come from one function
        # in audio_gen rather than being written a second time here — the two
        # producing different captions for the same script is the kind of
        # difference nobody notices until a viewer sees both.
        words = [
            {"text": text, "start": round(start, 3), "end": round(end, 3)}
            for start, end, text in audio_gen._cluster_words(segments, audio_dur)
        ]
        # HALF OF A CHOSEN CAPTION STYLE REACHES THIS ENGINE AND HALF DOES NOT,
        # and the person who chose it deserves to hear which. The grouping and
        # the casing are above, from the shared clusterer, so they are honoured
        # here exactly as on the FFmpeg path. The LOOK — size, height, outline,
        # box, colour — is written into an ASS style line that only FFmpeg
        # burns; Remotion draws its captions in TSX. Silently ignoring the
        # visual half is precisely the failure this repo keeps having, so it is
        # said in the log rather than discovered in the finished file.
        try:
            import caption_styles
            chosen = caption_styles.name()
            if chosen != caption_styles.DEFAULT:
                print(f"[captions] style {chosen!r}: {audio_gen.CLUSTER_SIZE} "
                      f"word(s) per card and the casing apply here; its size, "
                      f"position and colour do not — those are the FFmpeg "
                      f"renderer's, and this run is Remotion.")
        except Exception:
            pass
        # AND THE RAW STREAM, KEPT SEPARATELY. The captions above are grouped
        # and cased for READING — four words to a card in long-form. The insert
        # planner is not reading them: it looks up one noun at a time and needs
        # the second that noun was spoken, so handing it caption cards would
        # find "the palace held a" where it looked for "palace" and plan
        # nothing at all, silently, on this renderer only.
        spoken = [
            {"text": w.word.strip(), "start": round(w.start, 3),
             "end": round(w.end, 3)}
            for seg in segments for w in seg.words
            if w.word.strip() and w.start < audio_dur
        ]
        # The word stream for chapters, published from whichever renderer ran —
        # a description whose timestamps depend on which engine drew the frames
        # would be a bug that only shows up on half the runs.
        audio_gen.LAST_WORDS = [(w["start"], w["text"]) for w in spoken]

        print("[3/4] Staging assets…")
        # Self-hosted caption font: reuse the Anton ttf the FFmpeg path downloads.
        try:
            audio_gen._ensure_font()
            font_dst = PUBLIC_DIR / "fonts" / "Anton-Regular.ttf"
            if audio_gen.FONT_FILE.exists() and not font_dst.exists():
                font_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(audio_gen.FONT_FILE, font_dst)
        except Exception as e:
            print(f"[font] staging skipped ({e}) — Remotion falls back to Arial")

        clip_names = []
        clip_durs  = []
        for i, bg in enumerate(bg_paths):
            name = f"clip_{i}{bg.suffix or '.mp4'}"
            shutil.copy2(bg, job_dir / name)
            clip_names.append(name)
            clip_durs.append(_probe_duration(bg) or 8.0)  # 8s fallback; None → JSON null → crash

        music_name = None
        if music_path and Path(music_path).exists():
            music_name = f"music{Path(music_path).suffix or '.mp3'}"
            shutil.copy2(music_path, job_dir / music_name)

        # How each beat MOVES, decided from the script rather than by the
        # index%6 cycle Short.tsx falls back to. One beat per clip, because a
        # clip IS a beat here. None on any failure — see edit_director's
        # contract; the renderer's default cycle is a working edit, just not a
        # directed one.
        edit = None
        beats: list[str] = []
        try:
            import edit_director
            import main as _main
            beats = _main._split_beats(script, max_scenes=len(clip_names), grow=True)
            if len(beats) == len(clip_names):
                edit = edit_director.direct(beats)
        except Exception as e:
            print(f"[director] unavailable ({e}) — default motion cycle")

        # WHEN EACH PICTURE IS ON SCREEN, from the word timings rather than
        # from an even division of the runtime.
        #
        # Short.tsx sized every shot as ceil((total + (n-1)*trans) / n), so a
        # beat whose sentence takes two seconds and one whose sentence takes
        # six got the same time — the picture drifting further out of step with
        # the voice as the video went on. The FFmpeg path never had this; its
        # _plan_cuts snaps to sentence ends. Both renderers ship the same
        # channel and only one of them was cutting on the voice.
        #
        # Only when a beat IS a clip. With several stills to a beat the two
        # lists do not correspond, and a span list that does not line up with
        # the clips would be worse than no span list at all.
        beat_spans = []
        if beats and len(beats) == len(clip_names):
            try:
                beat_spans = audio_gen.beat_spans(beats, spoken, audio_dur)
            except Exception as e:
                print(f"[cuts] beat alignment unavailable ({e}) — even shots")
        if beat_spans:
            print(f"[cuts] {len(beat_spans)} shot(s) cut on the narration")

        # WORD-SYNCED INSERTS. Planned here rather than earlier because this is
        # the first point where the FINISHED voiceover has been transcribed —
        # an insert is pinned to the second its word is actually spoken, and
        # only Whisper knows that. Fail-open: any failure leaves `inserts`
        # empty and the video renders exactly as it did before this existed.
        inserts: list[dict] = []
        try:
            import insert_director
            if insert_director.enabled():
                planned = insert_director.plan_for(script, spoken, _insert_style())
                if planned:
                    print(insert_director.describe(planned))
                    import comfy_client
                    inserts = comfy_client.render_inserts(planned, job_dir)
        except Exception as e:
            print(f"[inserts] unavailable ({e}) — rendering without them")

        props = {
            "job":               job,
            "clips":             clip_names,
            "clipDurations":     clip_durs,
            "voice":             "voice.mp3",
            "music":             music_name,
            "words":             words,
            "durationInSeconds": round(audio_dur, 3),
            # The render's shape, from the active format profile. Root's
            # calculateMetadata reads these; without them a long-form job
            # renders at the vertical default and every frame is cropped.
            "width": _fmt.dimensions()[0],
            "height": _fmt.dimensions()[1],
            "edit":              edit,
            # [start, end] per clip in seconds. Absent or empty → Short.tsx
            # divides the runtime evenly, which is what it always did.
            "beatSpans":         beat_spans or None,
            # The words the director marked, resolved the same way the FFmpeg
            # path resolves them — including the share guard, so a plan that
            # marks half the script does not turn the captions green in one
            # renderer and leave them white in the other.
            "emphasis":          sorted(audio_gen.emphasis_words(
                edit, len(script.split()))),
            "inserts":           inserts or None,
        }
        props_file = job_dir / "props.json"
        props_file.write_text(json.dumps(props), encoding="utf-8")

        print(f"[4/4] Remotion render: {len(clip_names)} clip(s) → {audio_dur:.1f}s"
              f"{' + music' if music_name else ''}"
              f"{f' + {len(inserts)} insert(s)' if inserts else ''}…")
        cmd = [
            _npx(), "remotion", "render", "src/index.ts", "RufusShort", str(out),
            f"--props={props_file}",
        ]
        result = subprocess.run(cmd, cwd=REMOTION_DIR, capture_output=True,
                                text=True, timeout=RENDER_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(
                f"Remotion render failed (rc={result.returncode}):\n{result.stderr[-800:]}"
            )
        if not out.exists() or out.stat().st_size < 100_000:
            raise RuntimeError("Remotion render produced no/empty output file")

        # Lock loudness to the same -14 LUFS the FFmpeg path masters to.
        #
        # This was missing for as long as this renderer has existed, and nobody
        # could see it, because the renderer never actually ran — the npx spawn
        # failed on Windows and every render fell through to audio_gen, which
        # does normalise. The first successful Remotion render exposed it in one
        # line of QC: -25.4 dB mean, against -17.5 dB from the FFmpeg path on
        # the same pipeline. YouTube turns loud audio DOWN but never turns quiet
        # audio up, so an un-normalised Short just sounds weak beside every
        # other video in the feed.
        #
        # Fail-open like every other post-step: a skipped pass leaves the
        # quieter mix, which is still a finished video.
        try:
            audio_gen._normalize_loudness(out)
        except Exception as e:
            print(f"[remotion] loudness pass skipped ({e})")

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    print(f"Done → {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--bg",     required=True, nargs="+")
    p.add_argument("--out",    required=True)
    a = p.parse_args()
    result = render(a.script, [Path(b) for b in a.bg], Path(a.out))
    print(f"OUTPUT_PATH={result}")
