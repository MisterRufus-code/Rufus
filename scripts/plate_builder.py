"""Background plates, drawn as vector rather than generated.

Same argument that settled the mascot: a place that has to be identical in
every frame cannot be re-derived from noise each time. It is stronger here,
because the channel's style is *already* what a vector drawing is — thin black
line of uniform weight, flat saturated fill, no gradient, no shading, no
texture. There is nothing in it a renderer supplies that this cannot.

Everything is built from one small set of props so that twelve places look like
twelve rooms in one world instead of twelve drawings. That consistency is most
of what a viewer reads as "this channel", and it is the thing a generator
cannot hold across a library.

Run this module to (re)build every plate under assets/plates/.
"""

from __future__ import annotations

import json
from pathlib import Path

# The frame size comes from video_format, never from a second copy of the
# numbers — the sweep in test_video_format exists because seven modules once
# each wrote 1080x1920 down for themselves and two of the copies were wrong.
import video_format

W, H = video_format.dimensions()

INK = "#141414"
SW = 6

# The figure's lane. Nothing goes here: a plate is only useful if the thing it
# was drawn for is not standing in front of the half that says where this is.
GROUND_Y = 0.74
FIG_H = 0.44
LANE = (300, 780)

C = {
    "sky": "#7FC4E8", "cloud": "#FFFFFF",
    "grass": "#4CA352", "earth": "#A9793F", "dirt": "#96703C",
    "water": "#3E86B5", "stone": "#B9B2A6", "stone_d": "#8E8579",
    "brick": "#B4614A", "plaster": "#E8DFCC", "plank": "#C08B4E",
    "wood": "#A9773F", "wood_d": "#7A5327",
    "metal": "#8E97A3", "metal_d": "#5F6874",
    "cloth": "#C9503F", "cloth2": "#D9A441", "cloth3": "#4E7FA8",
    "fire": "#F0872B", "ember": "#F6C445", "coal": "#3A3733",
    "gold": "#E8B93C", "paper": "#F3EBD8", "glass": "#CFE6F2",
    "green_d": "#35753C", "night": "#22314A", "lamp": "#F6DE8B",
}


def _r(x, y, w, h, fill, rx=0, stroke=True):
    s = f' stroke="{INK}" stroke-width="{SW}"' if stroke else ""
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'rx="{rx}" fill="{fill}"{s}/>')


def _p(d, fill, stroke=True, sw=SW):
    s = f' stroke="{INK}" stroke-width="{sw}" stroke-linejoin="round"' if stroke else ""
    return f'<path d="{d}" fill="{fill}"{s}/>'


def _c(cx, cy, r, fill, stroke=True):
    s = f' stroke="{INK}" stroke-width="{SW}"' if stroke else ""
    return f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{fill}"{s}/>'


def _l(x1, y1, x2, y2, color=INK, w=SW):
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')


# ── props, each anchored bottom-centre at (x, y) ────────────────────────────
def crate(x, y, s=1.0, fill=None):
    w, h = 150 * s, 130 * s
    f = fill or C["wood"]
    return (_r(x - w / 2, y - h, w, h, f)
            + _l(x - w / 2, y - h * 0.62, x + w / 2, y - h * 0.62)
            + _l(x, y - h, x, y))


def barrel(x, y, s=1.0):
    w, h = 130 * s, 165 * s
    d = (f"M{x-w/2:.0f},{y-h*0.86:.0f} q{w*0.12:.0f},{-h*0.16:.0f} {w/2:.0f},{-h*0.14:.0f} "
         f"q{w*0.38:.0f},{h*0.02:.0f} {w/2:.0f},{h*0.14:.0f} "
         f"l{w*0.06:.0f},{h*0.72:.0f} q{-w*0.3:.0f},{h*0.2:.0f} {-w*1.12:.0f},0 Z")
    return _p(d, C["wood_d"]) + _l(x - w * 0.52, y - h * 0.55, x + w * 0.52, y - h * 0.55) \
        + _l(x - w * 0.55, y - h * 0.3, x + w * 0.55, y - h * 0.3)


def sacks(x, y, s=1.0):
    out = []
    for i, (dx, dy, k) in enumerate(((0, 0, 1.0), (-70, 0, 0.85), (-32, -95, 0.8))):
        w, h = 120 * s * k, 130 * s * k
        cx, cy = x + dx * s, y + dy * s
        d = (f"M{cx-w/2:.0f},{cy:.0f} q{-w*0.06:.0f},{-h*0.7:.0f} {w*0.3:.0f},{-h*0.86:.0f} "
             f"q{w*0.2:.0f},{-h*0.14:.0f} {w*0.4:.0f},0 "
             f"q{w*0.36:.0f},{h*0.16:.0f} {w*0.3:.0f},{h*0.86:.0f} Z")
        out.append(_p(d, C["paper"] if i % 2 else "#DFCFA6"))
    return "".join(out)


def table(x, y, s=1.0, fill=None):
    w, h = 320 * s, 130 * s
    f = fill or C["wood"]
    return (_r(x - w / 2, y - h, w, 26 * s, f)
            + _r(x - w / 2 + 14 * s, y - h + 26 * s, 22 * s, h - 26 * s, f)
            + _r(x + w / 2 - 36 * s, y - h + 26 * s, 22 * s, h - 26 * s, f))


def stool(x, y, s=1.0):
    w, h = 100 * s, 110 * s
    return (_r(x - w / 2, y - h, w, 20 * s, C["wood_d"])
            + _l(x - w / 3, y - h + 20 * s, x - w / 2.4, y)
            + _l(x + w / 3, y - h + 20 * s, x + w / 2.4, y))


def counter(x, y, s=1.0, fill=None):
    w, h = 520 * s, 190 * s
    f = fill or C["wood_d"]
    return (_r(x - w / 2, y - h, w, h, f)
            + _r(x - w / 2 - 16 * s, y - h - 22 * s, w + 32 * s, 26 * s, C["wood"])
            + _l(x - w / 6, y - h + 26 * s, x - w / 6, y)
            + _l(x + w / 6, y - h + 26 * s, x + w / 6, y))


def shelf(x, y, s=1.0, rows=3):
    w, h = 260 * s, 330 * s
    out = [_r(x - w / 2, y - h, w, h, C["wood"])]
    for i in range(1, rows):
        yy = y - h + h * i / rows
        out.append(_l(x - w / 2, yy, x + w / 2, yy))
    for i in range(rows):
        yy = y - h + h * (i + 1) / rows
        for j in range(3):
            bw = w / 4.4
            bx = x - w / 2 + 18 * s + j * (bw + 10 * s)
            bh = h / rows * 0.62
            out.append(_r(bx, yy - bh, bw, bh,
                          (C["cloth3"], C["cloth"], C["green_d"])[(i + j) % 3]))
    return "".join(out)


def window(x, y, s=1.0, view="sky"):
    """Wall opening. `y` is the BOTTOM of the window, not the floor."""
    w, h = 250 * s, 300 * s
    inner = C["sky"] if view == "sky" else C["night"]
    out = [_r(x - w / 2, y - h, w, h, inner)]
    if view == "sky":
        out.append(_c(x + w * 0.18, y - h * 0.72, 26 * s, C["cloud"], stroke=False))
        out.append(_c(x - w * 0.06, y - h * 0.68, 34 * s, C["cloud"], stroke=False))
        out.append(_r(x - w / 2 + SW, y - h * 0.28, w - 2 * SW, h * 0.28 - SW,
                      C["grass"], stroke=False))
    else:
        out.append(_c(x + w * 0.22, y - h * 0.74, 22 * s, C["lamp"], stroke=False))
    out.append(_l(x, y - h, x, y))
    out.append(_l(x - w / 2, y - h * 0.5, x + w / 2, y - h * 0.5))
    out.append(_r(x - w / 2, y - h, w, h, "none"))
    return "".join(out)


def arch_window(x, y, s=1.0):
    w, h = 230 * s, 420 * s
    d = (f"M{x-w/2:.0f},{y:.0f} L{x-w/2:.0f},{y-h*0.62:.0f} "
         f"A{w/2:.0f},{w/2:.0f} 0 0 1 {x+w/2:.0f},{y-h*0.62:.0f} L{x+w/2:.0f},{y:.0f} Z")
    return _p(d, C["glass"]) + _l(x, y - h * 0.9, x, y) \
        + _l(x - w / 2, y - h * 0.32, x + w / 2, y - h * 0.32)


def door(x, y, s=1.0):
    w, h = 220 * s, 400 * s
    return (_r(x - w / 2, y - h, w, h, C["wood_d"], rx=6)
            + _l(x - w / 2 + 18 * s, y - h + 18 * s, x - w / 2 + 18 * s, y - 18 * s)
            + _c(x + w * 0.28, y - h * 0.5, 12 * s, C["metal"]))


def hanging_lamp(x, y_top, s=1.0):
    """Hangs DOWN from y_top."""
    drop = 150 * s
    w = 150 * s
    d = (f"M{x-w/2:.0f},{y_top+drop:.0f} L{x-w*0.16:.0f},{y_top+drop-70*s:.0f} "
         f"L{x+w*0.16:.0f},{y_top+drop-70*s:.0f} L{x+w/2:.0f},{y_top+drop:.0f} Z")
    return _l(x, y_top, x, y_top + drop - 70 * s) + _p(d, C["metal_d"]) \
        + _c(x, y_top + drop + 6 * s, 26 * s, C["lamp"])


def column(x, y, s=1.0):
    w, h = 120 * s, 900 * s
    return (_r(x - w / 2, y - h, w, h, C["stone"])
            + _r(x - w * 0.66, y - h - 26 * s, w * 1.32, 34 * s, C["stone_d"])
            + _r(x - w * 0.66, y - 34 * s, w * 1.32, 34 * s, C["stone_d"])
            + _l(x - w * 0.2, y - h + 30 * s, x - w * 0.2, y - 44 * s)
            + _l(x + w * 0.2, y - h + 30 * s, x + w * 0.2, y - 44 * s))


def furnace(x, y, s=1.0):
    w, h = 420 * s, 520 * s
    out = [_r(x - w / 2, y - h, w, h, C["brick"], rx=10)]
    for i in range(4):
        out.append(_l(x - w / 2, y - h + h * (i + 1) / 5, x + w / 2, y - h + h * (i + 1) / 5))
    mw, mh = w * 0.44, h * 0.4
    d = (f"M{x-mw/2:.0f},{y-40*s:.0f} L{x-mw/2:.0f},{y-mh:.0f} "
         f"A{mw/2:.0f},{mw/2:.0f} 0 0 1 {x+mw/2:.0f},{y-mh:.0f} L{x+mw/2:.0f},{y-40*s:.0f} Z")
    out.append(_p(d, C["fire"]))
    out.append(_c(x, y - mh * 0.7, mw * 0.22, C["ember"], stroke=False))
    return "".join(out)


def anvil(x, y, s=1.0):
    w, h = 220 * s, 150 * s
    d = (f"M{x-w*0.3:.0f},{y:.0f} L{x-w*0.22:.0f},{y-h*0.42:.0f} "
         f"L{x-w/2:.0f},{y-h*0.6:.0f} L{x-w/2:.0f},{y-h:.0f} L{x+w*0.3:.0f},{y-h:.0f} "
         f"L{x+w/2:.0f},{y-h*0.8:.0f} L{x+w*0.26:.0f},{y-h*0.6:.0f} "
         f"L{x+w*0.18:.0f},{y-h*0.42:.0f} L{x+w*0.3:.0f},{y:.0f} Z")
    return _p(d, C["metal_d"])


def chest(x, y, s=1.0, open_=False):
    w, h = 230 * s, 150 * s
    out = [_r(x - w / 2, y - h, w, h, C["wood_d"], rx=6)]
    d = (f"M{x-w/2:.0f},{y-h:.0f} q{w/2:.0f},{-h*0.55:.0f} {w:.0f},0 Z")
    out.append(_p(d, C["wood"]))
    out.append(_r(x - 26 * s, y - h * 0.72, 52 * s, 46 * s, C["gold"], rx=4))
    if open_:
        out.append(_c(x, y - h * 1.15, 40 * s, C["gold"]))
    return "".join(out)


def safe(x, y, s=1.0):
    w, h = 260 * s, 300 * s
    return (_r(x - w / 2, y - h, w, h, C["metal_d"], rx=8)
            + _r(x - w / 2 + 24 * s, y - h + 24 * s, w - 48 * s, h - 48 * s, C["metal"], rx=4)
            + _c(x + w * 0.16, y - h * 0.5, 34 * s, C["gold"]))


def cart(x, y, s=1.0):
    w, h = 380 * s, 150 * s
    return (_r(x - w / 2, y - h - 40 * s, w, h, C["wood"])
            + _c(x - w * 0.28, y - 46 * s, 58 * s, C["wood_d"])
            + _c(x + w * 0.28, y - 46 * s, 58 * s, C["wood_d"])
            + _l(x + w / 2, y - h - 20 * s, x + w * 0.78, y - h * 1.3))


def tree(x, y, s=1.0):
    return (_r(x - 26 * s, y - 240 * s, 52 * s, 240 * s, C["wood_d"])
            + _c(x, y - 300 * s, 130 * s, C["green_d"])
            + _c(x - 100 * s, y - 250 * s, 90 * s, C["grass"])
            + _c(x + 100 * s, y - 258 * s, 84 * s, C["grass"]))


def bush(x, y, s=1.0):
    return (_c(x, y - 46 * s, 62 * s, C["green_d"])
            + _c(x - 54 * s, y - 30 * s, 46 * s, C["grass"])
            + _c(x + 54 * s, y - 32 * s, 44 * s, C["grass"]))


def rope_coil(x, y, s=1.0):
    return (_c(x, y - 30 * s, 62 * s, C["cloth2"])
            + _c(x, y - 30 * s, 34 * s, C["earth"])
            + _c(x, y - 30 * s, 12 * s, C["cloth2"]))


def bollard(x, y, s=1.0):
    w, h = 70 * s, 130 * s
    return (_r(x - w / 2, y - h, w, h, C["metal_d"], rx=8)
            + _c(x, y - h, w * 0.62, C["metal_d"]))


def hull(x, y, s=1.0):
    w, h = 620 * s, 200 * s
    d = (f"M{x-w/2:.0f},{y-h:.0f} L{x+w/2:.0f},{y-h:.0f} "
         f"q{-w*0.1:.0f},{h:.0f} {-w*0.42:.0f},{h:.0f} "
         f"L{x-w*0.28:.0f},{y:.0f} q{-w*0.3:.0f},0 {-w*0.22:.0f},{-h:.0f} Z")
    return _p(d, C["wood_d"]) + _l(x - w * 0.44, y - h * 0.55, x + w * 0.44, y - h * 0.55)


def mast(x, y, s=1.0):
    top = y - 900 * s
    return (_r(x - 14 * s, top, 28 * s, 900 * s, C["wood"])
            + _p(f"M{x+14*s:.0f},{top+70*s:.0f} L{x+250*s:.0f},{top+130*s:.0f} "
                 f"L{x+14*s:.0f},{top+430*s:.0f} Z", C["paper"]))


def awning(x, y_top, s=1.0, fill=None):
    w = 460 * s
    f = fill or C["cloth"]
    d = [f"M{x-w/2:.0f},{y_top:.0f}"]
    n = 5
    for i in range(n):
        x0 = x - w / 2 + w * i / n
        d.append(f"Q{x0+w/(2*n):.0f},{y_top+70*s:.0f} {x0+w/n:.0f},{y_top+40*s:.0f}")
    d.append(f"L{x+w/2:.0f},{y_top:.0f} Z")
    return _p(" ".join(d), f) + _l(x - w / 2, y_top, x + w / 2, y_top)


def stall(x, y, s=1.0, fill=None):
    out = [table(x, y, s * 1.1, fill=C["wood"])]
    top = y - 420 * s
    out.append(_l(x - 210 * s, y - 140 * s, x - 210 * s, top))
    out.append(_l(x + 210 * s, y - 140 * s, x + 210 * s, top))
    out.append(awning(x, top, s, fill))
    for i, cc in enumerate((C["cloth2"], C["green_d"], C["cloth"])):
        out.append(_c(x - 90 * s + i * 90 * s, y - 160 * s, 30 * s, cc))
    return "".join(out)


def desk(x, y, s=1.0):
    out = [table(x, y, s, fill=C["wood_d"])]
    out.append(_p(f"M{x-60*s:.0f},{y-140*s:.0f} l{120*s:.0f},0 l0,{-14*s:.0f} "
                  f"l{-120*s:.0f},0 Z", C["paper"]))
    return "".join(out)


def scales(x, y, s=1.0):
    top = y - 200 * s
    return (_r(x - 10 * s, top, 20 * s, 200 * s, C["metal_d"])
            + _l(x - 120 * s, top + 14 * s, x + 120 * s, top + 14 * s, INK, SW + 2)
            + _p(f"M{x-160*s:.0f},{top+50*s:.0f} l{80*s:.0f},0 l{-20*s:.0f},{40*s:.0f} "
                 f"l{-40*s:.0f},0 Z", C["gold"])
            + _p(f"M{x+80*s:.0f},{top+50*s:.0f} l{80*s:.0f},0 l{-20*s:.0f},{40*s:.0f} "
                 f"l{-40*s:.0f},0 Z", C["gold"])
            + _l(x - 120 * s, top + 14 * s, x - 120 * s, top + 50 * s)
            + _l(x + 120 * s, top + 14 * s, x + 120 * s, top + 50 * s))


def machine(x, y, s=1.0):
    w, h = 420 * s, 460 * s
    out = [_r(x - w / 2, y - h, w, h, C["metal"], rx=10)]
    out.append(_c(x - w * 0.2, y - h * 0.62, 80 * s, C["metal_d"]))
    out.append(_c(x - w * 0.2, y - h * 0.62, 30 * s, C["gold"]))
    out.append(_c(x + w * 0.22, y - h * 0.42, 54 * s, C["metal_d"]))
    out.append(_r(x - w * 0.1, y - h * 0.22, w * 0.5, 60 * s, C["coal"], rx=6))
    return "".join(out)


def bed(x, y, s=1.0):
    w, h = 420 * s, 150 * s
    return (_r(x - w / 2, y - h, w, h, C["wood"])
            + _r(x - w / 2, y - h - 60 * s, w * 0.42, 60 * s, C["paper"])
            + _r(x - w / 2 - 14 * s, y - h - 150 * s, 28 * s, 150 * s, C["wood_d"])
            + _r(x + w / 2 - 14 * s, y - h - 110 * s, 28 * s, 110 * s, C["wood_d"]))


def stove(x, y, s=1.0):
    w, h = 240 * s, 280 * s
    return (_r(x - w / 2, y - h, w, h, C["metal_d"], rx=6)
            + _c(x, y - h * 0.42, 52 * s, C["fire"])
            + _r(x - 24 * s, y - h - 420 * s, 48 * s, 420 * s, C["metal"]))


def plant(x, y, s=1.0):
    return (_p(f"M{x-46*s:.0f},{y-90*s:.0f} l{92*s:.0f},0 l{-14*s:.0f},{90*s:.0f} "
               f"l{-64*s:.0f},0 Z", C["brick"])
            + _c(x - 40 * s, y - 140 * s, 46 * s, C["green_d"])
            + _c(x + 36 * s, y - 152 * s, 42 * s, C["grass"])
            + _c(x, y - 190 * s, 44 * s, C["green_d"]))


def hill(x, y, s=1.0, fill=None):
    w, h = 700 * s, 180 * s
    return _p(f"M{x-w/2:.0f},{y:.0f} q{w/2:.0f},{-h*2:.0f} {w:.0f},0 Z",
              fill or C["green_d"])


def cloud(x, y, s=1.0):
    return (_c(x, y, 60 * s, C["cloud"], stroke=False)
            + _c(x - 62 * s, y + 16 * s, 44 * s, C["cloud"], stroke=False)
            + _c(x + 66 * s, y + 14 * s, 48 * s, C["cloud"], stroke=False))


# ── assembling a place ──────────────────────────────────────────────────────
GY = GROUND_Y * H          # 1420: where feet land


def _svg(parts: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}">' + "".join(parts) + "</svg>")


def _room(wall: str, floor: str, extra_floor: str = "") -> list[str]:
    """Wall to the horizon, floor from it down. The floor line IS the ground
    line the figure stands on, so the two are never allowed to disagree."""
    out = [_r(-4, -4, W + 8, GY + 4, wall, stroke=False),
           _r(-4, GY, W + 8, H - GY + 8, floor, stroke=False),
           _l(0, GY, W, GY)]
    if extra_floor:
        out.append(extra_floor)
    return out


def _outdoor(sky: str, ground: str) -> list[str]:
    return [_r(-4, -4, W + 8, GY + 4, sky, stroke=False),
            _r(-4, GY, W + 8, H - GY + 8, ground, stroke=False),
            _l(0, GY, W, GY)]


def _boards(n=6):
    return "".join(_l(0, GY + (H - GY) * i / n, W, GY + (H - GY) * i / n, INK, 3)
                   for i in range(1, n))


# Each entry: what the place is, and what is drawn in it. Props keep out of the
# figure's lane above the floor — a plate is only useful if the half that says
# where this is has not been covered by the thing it was drawn for.
PLACES: list[dict] = [
    dict(slug="counting_house", name="a merchant's counting room",
         when=[1400, 1880],
         keywords=["ledger", "counting", "clerk", "loan", "debt", "interest",
                   "merchant", "account", "desk", "strongbox", "bookkeeping"],
         light="window light from the left",
         back=lambda: _room(C["plaster"], C["plank"], _boards())
         + [window(210, GY - 560, 1.0), shelf(930, GY, 1.0),
            hanging_lamp(760, 0, 0.9), chest(170, GY, 0.9),
            scales(905, GY - 350, 0.7), plant(1010, GY, 0.7)],
         front=lambda: [desk(250, H - 40, 1.5)]),

    dict(slug="market_street", name="a market street",
         when=None,
         keywords=["market", "stall", "trader", "bought", "sold", "price",
                   "buyer", "seller", "bread", "grain", "wares", "haggle"],
         light="flat overcast daylight",
         back=lambda: _outdoor("#CFE0EA", "#B9AFA0")
         + [_r(-4, GY - 520, 420, 520, C["plaster"]),
            _r(700, GY - 560, 420, 560, "#E0CDB2"),
            window(150, GY - 300, 0.8), window(900, GY - 330, 0.8),
            stall(920, GY, 0.85, C["cloth2"]), barrel(120, GY, 0.9),
            crate(255, GY, 0.8)],
         front=lambda: [stall(300, H - 30, 1.25, C["cloth"])]),

    dict(slug="bank_hall", name="a bank's public hall",
         when=[1700, 2030],
         keywords=["bank", "teller", "deposit", "withdraw", "queue", "vault",
                   "branch", "run", "savings", "account", "cashier"],
         light="daylight from the tall windows",
         back=lambda: _room("#DCD3C2", "#C9C2B4",
                            "".join(_r(i * 180, GY + (j * 180), 180, 180,
                                       "#B5AC9C" if (i + j) % 2 else "#D5CDBE",
                                       stroke=False)
                                    for i in range(6) for j in range(3)))
         + [arch_window(150, GY - 480, 1.0), arch_window(930, GY - 480, 1.0),
            column(60, GY, 0.8), column(1020, GY, 0.8),
            hanging_lamp(540, 0, 1.0)],
         front=lambda: [counter(540, H - 20, 2.0, C["stone_d"])]),

    dict(slug="mint_forge", name="a coin mint",
         when=[600, 1900],
         keywords=["mint", "minted", "struck", "coin", "silver", "gold",
                   "forge", "furnace", "anvil", "die", "metal", "melted"],
         light="firelight from the furnace",
         back=lambda: _room("#6E5A48", "#7A6A57", _boards(4))
         + [furnace(880, GY, 1.0), anvil(180, GY, 1.0),
            crate(330, GY, 0.7, C["wood_d"]), hanging_lamp(430, 0, 0.8),
            barrel(1010, GY, 0.7)],
         front=lambda: []),

    dict(slug="quay", name="a harbour quay",
         when=[1200, 1950],
         keywords=["ship", "port", "harbour", "harbor", "cargo", "trade",
                   "fleet", "sailed", "dock", "quay", "crate", "voyage"],
         light="early daylight",
         back=lambda: [_r(-4, -4, W + 8, GY - 260, C["sky"], stroke=False),
                       _r(-4, GY - 260, W + 8, 264, C["water"], stroke=False),
                       _r(-4, GY, W + 8, H - GY + 8, C["stone"], stroke=False),
                       _l(0, GY, W, GY), cloud(240, 240, 1.0), cloud(820, 180, 0.8),
                       hull(760, GY - 40, 0.9), mast(760, GY - 200, 0.7),
                       bollard(120, GY, 1.0), rope_coil(1000, GY, 0.9)],
         front=lambda: [crate(190, H - 30, 1.7, C["wood_d"]),
                        crate(960, H - 60, 1.5)]),

    dict(slug="field", name="an open farmed field",
         when=None,
         keywords=["harvest", "crop", "grain", "wheat", "farm", "field",
                   "famine", "soil", "peasant", "plough", "land", "acre"],
         light="broad afternoon sunlight",
         back=lambda: _outdoor(C["sky"], C["grass"])
         + [cloud(200, 220, 1.0), cloud(880, 300, 0.85),
            hill(180, GY, 1.0, "#3E8A46"), hill(920, GY, 0.8, "#47A04F"),
            tree(120, GY, 1.0), bush(980, GY, 1.0), cart(950, GY, 0.7)],
         front=lambda: []),

    dict(slug="mine", name="a mine head",
         when=None,
         keywords=["mine", "mined", "ore", "seam", "shaft", "dig", "pit",
                   "silver", "gold", "rush", "prospector", "claim"],
         light="daylight at the shaft mouth",
         back=lambda: _room("#5E4B3C", "#6B5745", "")
         + [_p(f"M240,{GY:.0f} L240,{GY-420:.0f} A160,160 0 0 1 560,{GY-420:.0f} "
               f"L560,{GY:.0f} Z", C["coal"]),
            _l(280, GY, 520, GY - 300),
            cart(880, GY, 0.8), crate(1010, GY, 0.7, C["wood_d"]),
            hanging_lamp(760, 0, 0.7)],
         front=lambda: []),

    dict(slug="tavern", name="an inn's common room",
         when=[1200, 1900],
         keywords=["tavern", "inn", "alehouse", "wage", "paid", "drink",
                   "gathered", "rumour", "rumor", "news", "crowd"],
         light="firelight and one small window",
         back=lambda: _room("#8A6B4E", C["wood"], _boards(5))
         + [window(160, GY - 520, 0.75, view="night"),
            stove(950, GY, 0.9), table(880, GY, 0.8), stool(760, GY, 0.9),
            barrel(60, GY, 0.9), hanging_lamp(560, 0, 0.85),
            shelf(300, GY - 480, 0.6, rows=2)],
         front=lambda: [table(220, H - 30, 1.6, C["wood_d"])]),

    dict(slug="treasury", name="a royal treasury",
         when=[300, 1800],
         keywords=["treasury", "king", "queen", "crown", "tribute", "tax",
                   "hoard", "royal", "coffer", "empire", "throne", "levy"],
         light="lamplight on stone",
         back=lambda: _room(C["stone_d"], C["stone"],
                            "".join(_l(0, GY + i * 120, W, GY + i * 120, INK, 3)
                                    for i in range(1, 5)))
         + [column(90, GY, 0.9), column(990, GY, 0.9),
            arch_window(540, GY - 900, 0.7),
            chest(210, GY, 1.0, open_=True), chest(880, GY, 0.9),
            hanging_lamp(300, 0, 0.8), hanging_lamp(790, 0, 0.8)],
         front=lambda: [chest(120, H - 40, 1.5, open_=True)]),

    dict(slug="vault", name="a bank vault",
         when=[1850, 2030],
         keywords=["vault", "reserve", "bullion", "safe", "locked", "held",
                   "gold", "store", "deposit", "guard"],
         light="hard electric light",
         back=lambda: _room("#9AA3AE", "#7E8791",
                            "".join(_l(0, GY + i * 130, W, GY + i * 130, INK, 3)
                                    for i in range(1, 5)))
         + [safe(150, GY, 1.0), safe(930, GY, 1.0),
            _c(540, GY - 760, 150, C["metal_d"]), _c(540, GY - 760, 60, C["metal"]),
            hanging_lamp(250, 0, 0.7), hanging_lamp(830, 0, 0.7)],
         front=lambda: []),

    dict(slug="workshop", name="a workshop",
         when=[1500, 1980],
         keywords=["factory", "workshop", "worker", "machine", "made",
                   "production", "labour", "labor", "shift", "wage", "mill"],
         light="high window light",
         back=lambda: _room("#B7AE9E", "#8E8579", _boards(4))
         + [window(180, GY - 700, 0.9), window(900, GY - 700, 0.9),
            machine(880, GY, 0.9), crate(120, GY, 0.9, C["wood_d"]),
            barrel(255, GY, 0.7), hanging_lamp(540, 0, 0.9)],
         front=lambda: [table(300, H - 30, 1.5, C["wood_d"])]),

    dict(slug="home", name="a plain home interior",
         when=None,
         keywords=["home", "family", "household", "rent", "kitchen", "table",
                   "meal", "bread", "children", "wife", "husband", "afford"],
         light="one window, plain daylight",
         back=lambda: _room("#E3D5BC", C["plank"], _boards(5))
         + [window(190, GY - 620, 0.85), bed(930, GY, 0.8),
            stove(90, GY, 0.7), plant(1030, GY, 0.6),
            shelf(330, GY - 560, 0.55, rows=2), hanging_lamp(620, 0, 0.7)],
         front=lambda: [table(280, H - 30, 1.5)]),

    dict(slug="trading_floor", name="a modern trading floor",
         when=[1970, 2030],
         keywords=["market", "trader", "stock", "crash", "index", "shares",
                   "sold", "bought", "panic", "exchange", "broker", "screen"],
         light="flat electric light",
         back=lambda: _room("#3C4757", "#2E3743", "")
         + [_r(60, GY - 700, 300, 200, "#1E2733"),
            _r(720, GY - 700, 300, 200, "#1E2733"),
            _l(90, GY - 560, 330, 560 - 640 + GY - GY + 560 - 560 + (GY - 560), "#4CA352", 6),
            desk(180, GY, 1.0), desk(900, GY, 1.0),
            hanging_lamp(540, 0, 0.7)],
         front=lambda: [counter(540, H - 20, 1.9, "#232C36")]),

    dict(slug="road", name="an open road between towns",
         when=None,
         keywords=["road", "carried", "travelled", "traveled", "route",
                   "caravan", "journey", "toll", "wagon", "border", "carried"],
         light="clear daylight",
         back=lambda: _outdoor(C["sky"], "#B49A6E")
         + [cloud(260, 200, 1.0), cloud(860, 280, 0.8),
            hill(240, GY, 1.1, "#7E9E5E"), hill(880, GY, 0.9, "#8FAA6B"),
            tree(1000, GY, 0.9), bush(90, GY, 0.9), cart(170, GY, 0.8),
            _p(f"M{W*0.32:.0f},{H:.0f} L{W*0.44:.0f},{GY:.0f} "
               f"L{W*0.56:.0f},{GY:.0f} L{W*0.68:.0f},{H:.0f} Z",
               "#C9B489", stroke=False)],
         front=lambda: []),
]


def build(root: Path | None = None) -> list[str]:
    """Draw every place and write it as a plate. Returns the slugs written."""
    import cairosvg

    base = root or (Path(__file__).parent.parent / "assets" / "plates")
    base.mkdir(parents=True, exist_ok=True)
    written = []
    for place in PLACES:
        d = base / place["slug"]
        d.mkdir(exist_ok=True)
        parts = place["back"]()
        up = UPPER.get(place["slug"])
        if up:
            # AFTER the room and BEFORE the props would bury it; the band lives
            # above head height, so ordering against the floor props is moot.
            parts = parts[:3] + up() + parts[3:]
        cairosvg.svg2png(bytestring=_svg(parts).encode(),
                         write_to=str(d / "back.png"),
                         output_width=W, output_height=H)
        front = place["front"]()
        if front:
            cairosvg.svg2png(bytestring=_svg(front).encode(),
                             write_to=str(d / "front.png"),
                             output_width=W, output_height=H,
                             background_color="transparent")
        elif (d / "front.png").exists():
            (d / "front.png").unlink()
        (d / "plate.json").write_text(json.dumps({
            "name": place["name"],
            "when": place["when"],
            "keywords": place["keywords"],
            "light": place["light"],
            "ground_y": GROUND_Y,
            "figure_height": FIG_H,
            "safe_x": [0.30, 0.72],
        }, indent=2) + "\n", encoding="utf-8")
        written.append(place["slug"])
    return written




# ── the upper band ──────────────────────────────────────────────────────────
# A standing figure reaches from the floor to about a third of the way down the
# frame, so everything ABOVE that is free across the FULL width — and the first
# pass left it empty, which is the failure this file's own README names: a
# vertical plate with nothing above head height reads as a close-up of a wall.

def beams(y, n=3, fill=None):
    f = fill or C["wood_d"]
    out = [_r(-4, y, W + 8, 44, f)]
    for i in range(n):
        x = W * (i + 0.5) / n
        out.append(_r(x - 26, y + 44, 52, 120, f))
    return "".join(out)


def rafters(y):
    out = [_p(f"M-4,{y+150:.0f} L{W/2:.0f},{y:.0f} L{W+4:.0f},{y+150:.0f} "
              f"L{W+4:.0f},{y+40:.0f} L{W/2:.0f},{y-110:.0f} L-4,{y+40:.0f} Z",
              C["wood_d"])]
    for i in (-1, 1):
        out.append(_l(W / 2, y + 20, W / 2 + i * 360, y + 150))
    return "".join(out)


def wall_row(y, kinds=("pan", "tool", "pan", "tool", "pan")):
    """Small things hung high on a wall, spanning the width."""
    out = [_l(0, y, W, y, INK, 4)]
    for i, k in enumerate(kinds):
        x = W * (i + 0.5) / len(kinds)
        if k == "pan":
            out.append(_c(x, y + 60, 44, C["metal"]))
            out.append(_l(x, y, x, y + 20))
        else:
            out.append(_r(x - 12, y, 24, 130, C["wood"]))
            out.append(_p(f"M{x-40:.0f},{y+130:.0f} l80,0 l0,40 l-80,0 Z", C["metal_d"]))
    return "".join(out)


def high_shelf(y, n=6):
    out = [_r(-4, y, W + 8, 26, C["wood"])]
    for i in range(n):
        x = W * (i + 0.5) / n
        h = 90 + (i % 3) * 30
        out.append(_r(x - 40, y - h, 80, h,
                      (C["cloth3"], C["cloth"], C["green_d"], C["gold"])[i % 4]))
    return "".join(out)


def skyline(y, fill="#9AA7B4"):
    out = []
    xs = 0
    i = 0
    while xs < W + 60:
        w = 130 + (i % 3) * 60
        h = 150 + ((i * 7) % 4) * 70
        out.append(_r(xs, y - h, w, h, fill))
        out.append(_p(f"M{xs-14:.0f},{y-h:.0f} L{xs+w/2:.0f},{y-h-70:.0f} "
                      f"L{xs+w+14:.0f},{y-h:.0f} Z", fill))
        xs += w + 16
        i += 1
    return "".join(out)


def treeline(y, fill=None):
    f = fill or "#2F6B37"
    out = []
    x = -40
    i = 0
    while x < W + 60:
        r = 80 + (i % 3) * 26
        out.append(_c(x, y - r * 0.5, r, f, stroke=False))
        x += r * 1.3
        i += 1
    out.append(_r(-4, y - 10, W + 8, 40, f, stroke=False))
    return "".join(out)


def hanging_goods(y, n=5):
    out = [_l(0, y, W, y, INK, 4)]
    for i in range(n):
        x = W * (i + 0.5) / n
        out.append(_l(x, y, x, y + 50))
        out.append(sacks(x, y + 200, 0.5) if i % 2 else
                   _c(x, y + 110, 56, (C["cloth2"], C["green_d"])[i % 2]))
    return "".join(out)


def arch_row(y, n=3):
    out = []
    for i in range(n):
        x = W * (i + 0.5) / n
        out.append(_p(f"M{x-130:.0f},{y+220:.0f} L{x-130:.0f},{y+60:.0f} "
                      f"A130,130 0 0 1 {x+130:.0f},{y+60:.0f} L{x+130:.0f},{y+220:.0f} Z",
                      C["stone_d"]))
    return "".join(out)


# The upper band each place hangs above the figure's head. Applied after the
# place is drawn so it always sits behind nothing and in front of the wall.
UPPER = {
    "counting_house": lambda: [beams(120), high_shelf(430)],
    "market_street": lambda: [skyline(340, "#A9B4BE"), hanging_goods(70)],
    "bank_hall": lambda: [arch_row(60), beams(400, 4, C["stone_d"])],
    "mint_forge": lambda: [rafters(150), wall_row(430, ("tool", "pan", "tool", "pan", "tool"))],
    "quay": lambda: [skyline(300, "#8FA0AE")],
    "field": lambda: [treeline(GY - 150)],
    "mine": lambda: [beams(140, 4), wall_row(440, ("tool", "tool", "pan", "tool", "tool"))],
    "tavern": lambda: [beams(160), hanging_goods(430)],
    "treasury": lambda: [arch_row(80), beams(430, 3, C["stone_d"])],
    "vault": lambda: [beams(120, 4, C["metal_d"]), high_shelf(450)],
    "workshop": lambda: [rafters(120), wall_row(470)],
    "home": lambda: [beams(150), high_shelf(460)],
    "trading_floor": lambda: [beams(110, 4, "#2A323C"), high_shelf(460)],
    "road": lambda: [treeline(GY - 170, "#5E7F45"), skyline(GY - 210, "#94A48E")],
}


if __name__ == "__main__":
    for slug in build():
        print(f"[plates] wrote {slug}")
