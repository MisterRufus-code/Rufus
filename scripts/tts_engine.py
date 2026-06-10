#!/usr/bin/env python3
"""
tts_engine.py — Pluggable text-to-speech for Rufus.

Two backends, selected via the RUFUS_TTS environment variable:

  RUFUS_TTS=edge   (default) — Microsoft Edge TTS. Free, fast, cloud, no GPU.
                               Slightly synthetic but reliable.
  RUFUS_TTS=xtts             — Coqui XTTS v2. Free forever, runs locally on a
                               GTX 1060 6GB (~3GB VRAM), near-ElevenLabs quality,
                               supports voice cloning from a 6-second sample.

XTTS voice cloning (optional):
  RUFUS_TTS_VOICE=/path/to/reference.wav   # 6-30s clean speech sample to clone
  If unset, XTTS uses a built-in studio speaker.

Both backends write to the exact output path requested (mp3). XTTS synthesizes
wav internally then transcodes to mp3 so the downstream Whisper/FFmpeg path is
identical regardless of backend. Any XTTS failure falls back to Edge TTS so a
render never breaks over a voice issue.
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

# Edge TTS defaults (kept in sync with audio_gen historical settings)
EDGE_VOICE = os.environ.get("RUFUS_EDGE_VOICE", "en-US-ChristopherNeural")
EDGE_RATE  = os.environ.get("RUFUS_EDGE_RATE", "+6%")

# XTTS defaults
XTTS_MODEL    = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = os.environ.get("RUFUS_TTS_LANG", "en")
# Built-in studio speaker used when no reference clip is provided.
XTTS_DEFAULT_SPEAKER = os.environ.get("RUFUS_XTTS_SPEAKER", "Damien Black")

_xtts_model = None   # lazy singleton — loading the model is expensive


def _backend() -> str:
    return os.environ.get("RUFUS_TTS", "edge").strip().lower()


# ── Edge TTS ──────────────────────────────────────────────────────────────────

async def _edge_async(script: str, out_path: Path) -> None:
    import edge_tts
    comm = edge_tts.Communicate(script, EDGE_VOICE, rate=EDGE_RATE)
    await comm.save(str(out_path))


def _edge(script: str, out_path: Path) -> None:
    asyncio.run(_edge_async(script, out_path))


# ── XTTS v2 (Coqui) ─────────────────────────────────────────────────────────────

def _load_xtts():
    """Load XTTS v2 once. Uses GPU if RUFUS_GPU is set and CUDA is available."""
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    from TTS.api import TTS  # heavy import — only when XTTS is actually used

    gpu = os.environ.get("RUFUS_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        import torch
        gpu = gpu and torch.cuda.is_available()
    except Exception:
        gpu = False

    _xtts_model = TTS(XTTS_MODEL, gpu=gpu)
    print(f"[tts] XTTS v2 loaded ({'GPU' if gpu else 'CPU'})")
    return _xtts_model


def _xtts(script: str, out_path: Path) -> None:
    """Synthesize with XTTS v2 → wav → transcode to the requested mp3 path."""
    model = _load_xtts()
    speaker_wav = os.environ.get("RUFUS_TTS_VOICE", "").strip()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        kwargs = {"text": script, "file_path": wav_path, "language": XTTS_LANGUAGE}
        if speaker_wav and Path(speaker_wav).exists():
            kwargs["speaker_wav"] = speaker_wav
        else:
            if speaker_wav:
                print(f"[tts] XTTS reference clip not found ({speaker_wav}) — using built-in speaker")
            kwargs["speaker"] = XTTS_DEFAULT_SPEAKER
        model.tts_to_file(**kwargs)

        # Transcode wav → mp3 at the requested path so downstream is unchanged.
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
             "-c:a", "libmp3lame", "-b:a", "192k", str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 5_000:
            raise RuntimeError(f"XTTS mp3 transcode failed: {r.stderr[-300:]}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(script: str, out_path: Path) -> None:
    """Generate speech for `script` at `out_path` (mp3). Backend per RUFUS_TTS.

    XTTS failures fall back to Edge TTS so a render never breaks over the voice.
    """
    out_path = Path(out_path)
    backend  = _backend()

    if backend == "xtts":
        try:
            print("[tts] backend: XTTS v2 (local)")
            _xtts(script, out_path)
            return
        except Exception as e:
            print(f"[tts] XTTS failed ({e}) — falling back to Edge TTS")

    _edge(script, out_path)


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a Rufus voice test. One, two, three."
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tts_test.mp3")
    synthesize(text, out)
    print(f"OUTPUT={out}  ({out.stat().st_size} bytes)")
