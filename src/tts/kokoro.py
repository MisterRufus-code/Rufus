"""
Free local text-to-speech using Kokoro TTS.

Install: pip install kokoro-onnx soundfile
Models downloaded automatically on first run (~300 MB, one-time).

Kokoro produces near-ElevenLabs quality at zero cost.
Voices: af_heart, af_bella, af_sarah, am_adam, am_michael, bf_emma, bm_george
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
SAMPLE_RATE = 24000


def _get_kokoro():
    try:
        from kokoro_onnx import Kokoro
        return Kokoro
    except ImportError:
        raise RuntimeError(
            "Kokoro TTS not installed.\n"
            "Run: pip install kokoro-onnx soundfile"
        )


def synthesize(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = "en-us",
) -> Path:
    """
    Convert *text* to speech and save as a WAV file at *output_path*.
    Returns the output path.
    """
    Kokoro = _get_kokoro()
    import soundfile as sf
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]TTS:[/cyan] generating voiceover ({len(text)} chars, voice={voice})")

    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)

    sf.write(str(output_path), samples, sample_rate)
    console.print(f"[green]✓ Voiceover saved:[/green] {output_path}")
    return output_path


def synthesize_sections(
    sections: list[dict],
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
) -> list[Path]:
    """
    Synthesize each script section separately.
    Returns list of WAV file paths in section order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i, section in enumerate(sections):
        text = section.get("script", "")
        if not text.strip():
            continue
        out = output_dir / f"section_{i:02d}.wav"
        synthesize(text, out, voice=voice, speed=speed)
        paths.append(out)

    return paths


def merge_audio_files(wav_paths: list[Path], output_path: Path, gap_seconds: float = 0.5) -> Path:
    """Concatenate multiple WAV files with a short silence gap between them."""
    import soundfile as sf
    import numpy as np

    output_path = Path(output_path)
    silence = np.zeros(int(SAMPLE_RATE * gap_seconds), dtype=np.float32)
    combined = np.array([], dtype=np.float32)

    for p in wav_paths:
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # mono
        combined = np.concatenate([combined, data, silence])

    sf.write(str(output_path), combined, SAMPLE_RATE)
    console.print(f"[green]✓ Full audio merged:[/green] {output_path}")
    return output_path


def list_voices() -> list[str]:
    return [
        "af_heart", "af_bella", "af_sarah", "af_nicole",
        "am_adam", "am_michael",
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis",
    ]
