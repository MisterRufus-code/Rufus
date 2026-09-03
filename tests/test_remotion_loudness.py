"""Both renderers master to the same loudness.

The first Remotion render that ever completed on the owner's machine exposed
this in one line of QC:

    FFmpeg path:   [audio] loudness locked to -14 LUFS   → QC -17.5 dB mean
    Remotion path: (no such line)                        → QC -25.4 dB mean

Eight decibels quieter, shipped. The normalisation existed the whole time in
audio_gen; remotion_renderer simply never called it, and nobody could see that
because the renderer never ran — the Windows npx spawn failed and every render
fell through to the path that does normalise. YouTube turns loud audio down and
never turns quiet audio up, so this is a retention bug, not a cosmetic one.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

SRC = Path(__file__).parent.parent / "scripts" / "remotion_renderer.py"


def test_the_remotion_path_normalises_loudness():
    assert "_normalize_loudness" in SRC.read_text(encoding="utf-8")


def test_it_normalises_after_the_output_is_validated():
    """Normalising a file that failed the size check would rewrite a broken
    render instead of raising."""
    src = SRC.read_text(encoding="utf-8")
    assert src.index("produced no/empty output file") < src.index("_normalize_loudness")


def test_the_loudness_pass_cannot_break_a_finished_render():
    """Fail-open like every other post-step: a skipped pass leaves the quieter
    mix, which is still a publishable video."""
    src = SRC.read_text(encoding="utf-8")
    tail = src[src.index("_normalize_loudness"):]
    assert "except Exception" in tail[:400]
    assert "loudness pass skipped" in tail[:500]


def test_both_renderers_target_the_same_number():
    import audio_gen

    body = Path(audio_gen.__file__).read_text(encoding="utf-8")
    targets = set(re.findall(r"-14 LUFS", body))
    assert targets, "audio_gen must state its target so the two paths cannot drift"
