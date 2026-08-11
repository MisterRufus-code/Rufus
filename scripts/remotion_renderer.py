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
           music_path: Path | None = None) -> Path:
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

        words = [
            {"text": w.word.strip().upper(), "start": round(w.start, 3), "end": round(w.end, 3)}
            for seg in segments for w in seg.words
            if w.word.strip() and w.start < audio_dur
        ]

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
        try:
            import edit_director
            import main as _main
            beats = _main._split_beats(script, max_scenes=len(clip_names))
            if len(beats) == len(clip_names):
                edit = edit_director.direct(beats)
        except Exception as e:
            print(f"[director] unavailable ({e}) — default motion cycle")

        props = {
            "job":               job,
            "clips":             clip_names,
            "clipDurations":     clip_durs,
            "voice":             "voice.mp3",
            "music":             music_name,
            "words":             words,
            "durationInSeconds": round(audio_dur, 3),
            "edit":              edit,
        }
        props_file = job_dir / "props.json"
        props_file.write_text(json.dumps(props), encoding="utf-8")

        print(f"[4/4] Remotion render: {len(clip_names)} clip(s) → {audio_dur:.1f}s"
              f"{' + music' if music_name else ''}…")
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
