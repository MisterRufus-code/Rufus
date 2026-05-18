"""
Free local text-to-speech using Kokoro TTS.

Install: pip install kokoro-onnx soundfile
Models downloaded automatically on first run (~300 MB, one-time).

Kokoro produces near-ElevenLabs quality at zero cost.
Voices: af_heart, af_bella, af_sarah, am_adam, am_michael, bf_emma, bm_george
"""

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

_CLIP_REF_RE = re.compile(r"(\[?(?:CLIP|footage|clip)\s*[\d_]+\]?:?\s*)", re.IGNORECASE)
_HEADING_RE = re.compile(r'\b(point|step|section|part|footage|scene)[_ ]?\d+[:\.\)]?\s*', re.IGNORECASE)


def _prep_tts_text(text: str) -> str:
    """Strip markdown formatting and normalise punctuation for clean TTS delivery."""
    text = _CLIP_REF_RE.sub("", text)             # strip footage_1:, [CLIP 2] etc.
    text = _HEADING_RE.sub("", text)              # strip "Point 1:", "Step 2" etc.
    text = re.sub(r'[*_#`~]', '', text)           # remove markdown symbols
    text = re.sub(r'\n+', ' ', text)              # collapse newlines to spaces
    text = re.sub(r'([.!?]){2,}', r'\1', text)   # deduplicate sentence-end punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # remove space before punctuation
    text = re.sub(r' {2,}', ' ', text)            # collapse multiple spaces
    return text.strip()



DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
SAMPLE_RATE = 24000

import os as _os
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = Path(_os.getenv("KOKORO_MODELS_DIR") or (_PROJECT_ROOT / "models"))


def _model_path(filename: str) -> str:
    p = _MODELS_DIR / filename
    if p.exists():
        return str(p)
    legacy = _PROJECT_ROOT / filename
    if legacy.exists():
        return str(legacy)
    raise FileNotFoundError(
        f"Kokoro model not found: {p}\n"
        f"Download from https://github.com/thewh1teagle/kokoro-onnx/releases\n"
        f"Or set KOKORO_MODELS_DIR=/path/to/models"
    )

# ---------------------------------------------------------------------------
# Niche → voice profiles
# ---------------------------------------------------------------------------
# Each key maps to a dict with two pools:
#   "default"  – balanced selection for that niche
#   "energetic" – higher-energy voices
#   "calm"      – softer / warmer voices
#
# Override at runtime by mutating this dict before calling pick_voice().
# ---------------------------------------------------------------------------
NICHE_VOICE_PROFILES: dict[str, dict[str, list[str]]] = {
    "finance": {
        "default":   ["am_michael", "bm_george", "af_sarah"],
        "energetic": ["am_michael", "am_adam"],
        "calm":      ["bm_george", "af_sarah"],
    },
    "business": {
        "default":   ["am_michael", "bm_george", "af_sarah"],
        "energetic": ["am_michael", "am_adam"],
        "calm":      ["bm_george", "af_sarah"],
    },
    "tech": {
        "default":   ["am_adam", "af_nicole"],
        "energetic": ["am_adam", "am_michael"],
        "calm":      ["af_nicole", "af_sarah"],
    },
    "health": {
        "default":   ["af_heart", "af_bella", "bf_emma"],
        "energetic": ["af_bella", "bf_emma"],
        "calm":      ["af_heart", "bf_isabella"],
    },
    "wellness": {
        "default":   ["af_heart", "af_bella", "bf_emma"],
        "energetic": ["af_bella", "bf_emma"],
        "calm":      ["af_heart", "bf_isabella"],
    },
    "dark_triad_ai": {
        "default":   ["bm_george", "am_michael", "am_adam"],
        "energetic": ["am_michael", "am_adam"],
        "calm":      ["bm_george", "bm_lewis"],
    },
    "general": {
        "default":   ["af_heart", "af_bella", "af_sarah", "af_nicole",
                      "am_adam", "am_michael", "bf_emma", "bf_isabella",
                      "bm_george", "bm_lewis"],
        "energetic": ["am_adam", "am_michael", "af_bella", "bf_emma"],
        "calm":      ["af_heart", "bf_isabella", "bm_george", "af_sarah"],
    },
}


def pick_voice(niche: str, style: str = "default") -> str:
    """
    Choose a voice string appropriate for *niche* and *style*.

    Args:
        niche:  Content niche (e.g. "finance", "tech", "health", "wellness",
                "business", "general").  Unknown niches fall back to "general".
        style:  One of "default", "energetic", or "calm".
                Unknown style values fall back to "default".

    Returns:
        A single voice string from :data:`NICHE_VOICE_PROFILES`, chosen
        at random from the matching pool so repeated calls give variety.
    """
    niche_key = niche.lower().strip()
    profile = NICHE_VOICE_PROFILES.get(niche_key, NICHE_VOICE_PROFILES["general"])

    style_key = style.lower().strip()
    pool = profile.get(style_key, profile.get("default", list_voices()))

    if not pool:
        pool = list_voices()

    return random.choice(pool)


def _get_kokoro():
    try:
        from kokoro_onnx import Kokoro
        return Kokoro
    except ImportError:
        raise RuntimeError(
            "Kokoro TTS not installed.\n"
            "Run: pip install kokoro-onnx soundfile"
        )


def _build_kokoro_instance():
    """Create a single Kokoro model instance (expensive — reuse across sections)."""
    Kokoro = _get_kokoro()
    return Kokoro(_model_path("kokoro-v1.0.onnx"), _model_path("voices-v1.0.bin"))


def synthesize(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = "en-us",
    _kokoro_instance=None,
) -> Path:
    """
    Convert *text* to speech and save as a WAV file at *output_path*.
    Pass a pre-built _kokoro_instance to avoid re-loading the model.
    Returns the output path.
    """
    valid = list_voices()
    if voice not in valid:
        console.print(f"[yellow]Unknown voice '{voice}', falling back to {DEFAULT_VOICE}[/yellow]")
        voice = DEFAULT_VOICE

    import soundfile as sf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]TTS:[/cyan] generating voiceover ({len(text)} chars, voice={voice})")

    kokoro = _kokoro_instance if _kokoro_instance is not None else _build_kokoro_instance()
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
    Synthesize each script section using a single shared Kokoro model instance.
    Sharing the instance prevents pacing inconsistencies caused by model re-initialization.
    Returns list of WAV file paths in section order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Build model once — reuse across all sections for consistent pacing
    kokoro_instance = _build_kokoro_instance()

    for i, section in enumerate(sections):
        text = _prep_tts_text(_CLIP_REF_RE.sub("", section.get("script", "")).strip())
        if not text:
            continue
        out = output_dir / f"section_{i:02d}.wav"
        synthesize(text, out, voice=voice, speed=speed, _kokoro_instance=kokoro_instance)
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


def get_section_durations(wav_paths: list[Path]) -> list[float]:
    """Return duration in seconds for each synthesised WAV section."""
    import soundfile as sf
    durations = []
    for p in wav_paths:
        try:
            info = sf.info(str(p))
            durations.append(float(info.duration))
        except Exception:
            durations.append(0.0)
    return durations
