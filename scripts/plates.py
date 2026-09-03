"""Background plates: one fixed picture of a place, figures composited onto it.

A diffusion model renders every beat from noise with no memory of the others,
so "the same room" is a thing it can be asked for and cannot deliver. The
storyboard already knows this — _pin_setting's own docstring says so — and its
answer is to restate three words of the room into a quarter of the shots. That
narrows the drift; it cannot remove it.

A plate removes it. The place is drawn once, by hand, and kept: every shot set
there is generated as a FIGURE ALONE and pasted onto the same pixels. The room
is then identical across every frame of every video that uses it.

Compositing usually gives itself away through a missing contact shadow and
light that does not match. This channel's style forbids shading, ambient
occlusion and photographic lighting anywhere in the frame, so there is nothing
left to mismatch — which is why cut-and-paste is the correct method here rather
than a compromise.

Fail-open throughout, like every other optional stage in this pipeline: no
library, no match, or a cutout that does not look like a cutout, and the caller
renders the beat exactly the way it did before.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PLATES_DIR = Path(__file__).parent.parent / "assets" / "plates"

# How much of the frame the flood fill must claim for the render to count as
# "a figure on a plain background". Below the floor the background was not
# plain and we would be cutting into scenery; above the ceiling the fill ate
# the figure too and there is nothing left to paste.
_MIN_BACKGROUND = 0.12
_MAX_BACKGROUND = 0.995

# How far a pixel may sit from the background colour and still be background.
# The renders are flat fills, so this only has to absorb JPEG-ish ringing and
# the anti-aliased edge of an outline.
_FILL_TOLERANCE = 30

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "his", "her",
    "its", "their", "one", "two", "into", "over", "out", "up", "down",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


class Plate:
    """One place, plus where a figure stands in it."""

    def __init__(self, slug: str, meta: dict, directory: Path):
        self.slug = slug
        self.dir = directory
        self.name = str(meta.get("name") or slug)
        self.keywords = {str(k).lower() for k in meta.get("keywords") or []}
        self.light = str(meta.get("light") or "")
        when = meta.get("when")
        self.when: tuple[int, int] | None = None
        if isinstance(when, (list, tuple)) and len(when) == 2:
            try:
                self.when = (int(when[0]), int(when[1]))
            except (TypeError, ValueError):
                self.when = None
        self.ground_y = float(meta.get("ground_y", 0.74))
        self.figure_height = float(meta.get("figure_height", 0.44))
        safe = meta.get("safe_x") or [0.2, 0.8]
        self.safe_x = (float(safe[0]), float(safe[1]))

    @property
    def back(self) -> Path:
        return self.dir / "back.png"

    @property
    def front(self) -> Path:
        return self.dir / "front.png"

    def fits_era(self, year: int | None) -> bool:
        """Whether this place belongs to `year`. A plate with no range fits
        anything, and an unknown year fits everything — the era check is here
        to rule a place OUT, never to be the reason nothing matches."""
        if self.when is None or year is None:
            return True
        return self.when[0] <= year <= self.when[1]

    def score(self, shot: str, year: int | None) -> float:
        if not self.fits_era(year):
            return 0.0
        hits = self.keywords & _words(shot)
        if not hits:
            return 0.0
        # Normalised by the plate's own vocabulary, so a plate that lists
        # twenty keywords does not win every shot by breadth alone.
        return len(hits) / max(4, len(self.keywords))


def library(root: Path | None = None) -> list[Plate]:
    """Every usable plate on disk. A directory missing back.png or carrying a
    broken plate.json is skipped rather than raised on — a half-made plate
    someone is still drawing must not take the pipeline down with it."""
    base = root or PLATES_DIR
    out: list[Plate] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "plate.json"
        if not meta_file.is_file() or not (d / "back.png").is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        out.append(Plate(d.name, meta, d))
    return out


# A shot has to look like this place and not merely mention one of its words.
# One keyword out of a four-word plate clears it; one out of twelve does not.
MIN_SCORE = 0.24


def pick(shot: str, year: int | None = None,
         root: Path | None = None) -> Plate | None:
    """The best plate for this shot, or None to render it the old way."""
    best, best_score = None, 0.0
    for p in library(root):
        s = p.score(shot, year)
        if s > best_score:
            best, best_score = p, s
    return best if best_score >= MIN_SCORE else None


def cutout(image):
    """A figure-only render with its flat background made transparent.

    Flood-filled inward from the border rather than keyed on colour, because
    the figures are WHITE FILL inside a black outline: "remove everything pale"
    would erase the body along with the background. Only pale that connects to
    the edge of the frame is background.

    Returns an RGBA image cropped to the figure, or None when the result does
    not look like a cutout — too little filled means the background was not
    plain, too much means the fill leaked through the outline and ate the
    figure. Either way the caller falls back.
    """
    from PIL import Image, ImageDraw

    rgb = image.convert("RGB")
    w, h = rgb.size
    sentinel = (255, 0, 255)
    probe = rgb.copy()
    # The background colour is whatever the corners agree on. Seeding blindly
    # from the edge midpoints eats the figure whenever it touches the frame —
    # and it touches the bottom in most shots, because that is where feet go —
    # so a seed is only used if it already looks like the background.
    corners = [rgb.getpixel(c) for c in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    bg = max(set(corners), key=corners.count)
    near = lambda c: max(abs(a - b) for a, b in zip(c, bg)) <= _FILL_TOLERANCE
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for xy in seeds:
        if not near(rgb.getpixel(xy)):
            continue
        try:
            ImageDraw.floodfill(probe, xy, sentinel, thresh=_FILL_TOLERANCE)
        except ValueError:
            continue

    import numpy as np
    arr = np.asarray(probe)
    hit = ((arr[..., 0] == sentinel[0]) & (arr[..., 1] == sentinel[1])
           & (arr[..., 2] == sentinel[2]))
    mask = Image.fromarray(np.where(hit, 0, 255).astype("uint8"), "L")
    share = float(hit.mean())
    if share < _MIN_BACKGROUND or share > _MAX_BACKGROUND:
        return None

    out = rgb.convert("RGBA")
    out.putalpha(mask)
    box = mask.getbbox()          # bbox of the NON-zero (kept) pixels
    if box is None:
        return None
    return out.crop(box)


def compose(plate: Plate, figure, size: tuple[int, int] | None = None,
            seed: int = 0):
    """The figure standing in the plate's place, as one finished RGB image.

    The figure is scaled by the plate's own `figure_height` and set down with
    its feet on `ground_y`, which is what makes two shots of the same room
    agree about how big a person is and where the floor runs. Its position
    across the frame is jittered inside `safe_x` by the beat's seed: dead
    centre every time is the composition that gives the trick away.
    """
    from PIL import Image

    back = Image.open(plate.back).convert("RGBA")
    if size and back.size != size:
        back = back.resize(size, Image.LANCZOS)
    W, H = back.size

    target_h = max(1, int(H * plate.figure_height))
    scale = target_h / float(figure.height)
    fig = figure.resize((max(1, int(figure.width * scale)), target_h),
                        Image.LANCZOS)

    lo, hi = plate.safe_x
    span = max(0.0, hi - lo)
    # Deterministic, so a re-run of the same beat puts the figure back in the
    # same place rather than sliding it around between attempts.
    t = ((seed * 2654435761) % 1000) / 1000.0
    centre = (lo + span * t) * W
    x = int(centre - fig.width / 2)
    x = max(0, min(W - fig.width, x))
    y = int(H * plate.ground_y - fig.height)

    back.alpha_composite(fig, (x, max(0, y)))

    if plate.front.is_file():
        try:
            front = Image.open(plate.front).convert("RGBA")
            if front.size != back.size:
                front = front.resize(back.size, Image.LANCZOS)
            back.alpha_composite(front, (0, 0))
        except OSError:
            pass                  # a missing foreground layer is not a failure

    return back.convert("RGB")
