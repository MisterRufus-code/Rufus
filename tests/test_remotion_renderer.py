"""Short.tsx is what actually draws these videos.

RUFUS_RENDERER=remotion is what the owner's saved settings select, so this
component — not audio_gen's ffmpeg path — has drawn the captions on every
video the channel has published. Its numbers were 1080x1920 numbers, and the
format switch made that a problem nobody would have seen until a nine-minute
render came out with the words floating two-thirds of the way up the picture.
"""

def _short_tsx() -> str:
    from pathlib import Path as _P
    return (_P(__file__).parent.parent / "remotion" / "src" / "Short.tsx"
            ).read_text(encoding="utf-8")


def test_the_captions_are_sized_from_the_frame():
    """fontSize 96 and paddingBottom 700 are right for a phone: big words,
    lifted clear of the Shorts UI covering the bottom fifth. On a 1080-tall
    landscape frame the same two numbers are a caption floating 65% up the
    picture."""
    src = _short_tsx()
    assert "fontSize: 96" not in src
    assert "paddingBottom: 700" not in src
    assert "const portrait = height >= width" in src


def test_the_portrait_ratios_reproduce_the_shipped_numbers():
    """A refactor that moves the existing channel by a pixel has to be
    explained to somebody watching the videos. 0.05 x 1920 = 96 and
    0.3646 x 1920 = 700, both exactly."""
    import re
    src = _short_tsx()
    fs = float(re.search(r"height \* \(portrait \? ([\d.]+)", src).group(1))
    pb = float(re.search(r"height \* \(portrait \? ([\d.]+)",
                         src[src.index("paddingBottom ="):]).group(1))
    assert round(1920 * fs) == 96
    assert round(1920 * pb) == 700


def test_landscape_captions_sit_near_the_bottom():
    """No app UI to avoid, and no reason to cover the frame."""
    import re
    src = _short_tsx()
    pb = float(re.search(r": ([\d.]+)\)\);",
                         src[src.index("paddingBottom ="):]).group(1))
    assert 0.04 <= pb <= 0.12, pb


def test_the_composition_takes_its_shape_from_the_run():
    """Without this a long-form job renders at the vertical default and every
    landscape frame comes out cropped."""
    from pathlib import Path as _P
    root = (_P(__file__).parent.parent / "remotion" / "src" / "Root.tsx"
            ).read_text(encoding="utf-8")
    assert "props.width" in root and "props.height" in root
