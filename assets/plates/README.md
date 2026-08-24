# Background plates

A plate is one picture of a place with **nobody in it**. Every shot set in that
place is drawn as a figure alone and composited onto the plate, so the room is
pixel-identical in every frame of every video that uses it — which is the one
thing a diffusion model cannot do, because it renders each beat from noise with
no memory of the others.

Compositing normally betrays itself through a missing contact shadow and
mismatched light. This channel's style forbids shading, ambient occlusion and
photographic lighting anywhere in the frame, so there is nothing to mismatch.
That is why cut-and-paste is the right method here and not a compromise.

## Layout

```
assets/plates/<slug>/
    back.png      the empty place. Required.
    front.png     optional: only the things a figure can stand BEHIND
    plate.json    where a figure goes and what this place is for
```

`front.png` must be transparent everywhere except those foreground objects. It
is what lets a figure stand behind a counter, a crate or a railing; without it
every figure floats in front of everything.

## plate.json

```json
{
  "name": "counting house",
  "when": [1500, 1850],
  "keywords": ["ledger", "loan", "merchant", "counting", "debt", "interest"],
  "ground_y": 0.74,
  "figure_height": 0.44,
  "safe_x": [0.24, 0.76],
  "light": "window light from the left"
}
```

- `ground_y` — where feet land, as a fraction of the frame height
- `figure_height` — a standing figure's height, as a fraction of frame height
- `safe_x` — the horizontal band a figure can stand in without colliding with
  furniture, as fractions of the width
- `when` — the year range this place is right for; `null` for any era
- `keywords` — matched against the shot text to pick the plate

## Making a plate

Any tool will do — the plate is a fixed asset, so it is worth iterating on until
it is right. Produce it at **1080×1920** (vertical) and, where the place is also
wanted for long-form, **1920×1080**.

### The prompt

> Minimalist cartoon illustration in thin, clean black line art of uniform
> weight. Flat, saturated colour filled edge to edge — no gradients, no
> shading, no texture, no photographic lighting. **An empty PLACE with no
> people and no animals in it.** `<THE PLACE>`. A floor across the lower third
> for someone to stand on, and whatever closes off the distance behind it.
> Eight to twelve specific things that say where this is, each drawn properly
> with its own true shape and colour. `<THE LIGHT>`. The middle of the frame is
> clear and unobstructed. Vertical composition.
>
> Not in the picture, at all: any writing, lettering, numerals or inscription
> in any alphabet; any blank sign, signboard, placard or framed notice; any
> clock or watch; any screen or monitor that the place itself does not need.

That last paragraph is not boilerplate. A 6-minute stickman video made with an
off-the-shelf tool was measured frame by frame: **clocks in roughly 15 of 24
sampled frames, blank signboards in 17, screens in 17, stacked books in 14** —
in a video about money, where none of the four was in the script. They are the
generator's own furniture, and it draws them whenever a place is asked for.
Naming them as forbidden is the only thing that keeps them out.

### Starter set

| slug | THE PLACE | THE LIGHT |
|---|---|---|
| `counting_house` | a merchant's counting room: a heavy oak desk, an open blank ledger, brass scales, a strongbox, wooden shelves of rolled documents, a small window with panes | daylight through the window from the left |
| `market_street` | a market street of cobblestones: cloth awnings over wooden stalls, crates of produce, hanging baskets, barrels, shuttered houses behind | flat overcast daylight |
| `bank_hall` | a bank's public hall: a long marble counter, brass grilles above it, stone columns, a chequerboard floor, tall arched windows | daylight from the tall windows |
| `mint_forge` | a coin mint: a brick furnace with an open glowing mouth, an anvil, iron tongs and crucibles, a screw press, timber roof beams | firelight from the furnace |
| `quay` | a stone quay at a harbour: stacked crates and barrels, coiled rope, a bollard, a moored hull and mast behind, water beyond | early daylight |
| `field` | an open farmed field: furrowed earth, a low stone wall, a wooden cart, a distant treeline, open sky | broad afternoon sunlight |

Make `back.png` first for all of them. Add `front.png` only where something
really should pass in front of a figure — the counting-house desk, the market
stall, the bank counter, the quay crates.
