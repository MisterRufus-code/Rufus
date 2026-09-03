# The Paper Trails — mascot

`mascot.svg` is the master. Everything else is derived from it, and the master
is vector on purpose: the mark has to land on the same pixels in every frame of
every video, and a diffusion model cannot draw the same character twice. The
mascot is composited (Remotion `<Img>`, `thumbnail_gen` badge), never generated.

## What the character is

A green banknote that IS the body — there is no head. The face lives inside the
note's portrait oval, in the channel's own face vocabulary: two dot eyes, two
short brow strokes, one curved mouth, nothing else. Arms and legs come straight
out of the note. One hand holds a magnifying glass; the character is the one
following the money, which is what the channel does.

## The fixed symbol

Not a currency sign. A footprint, in all four corner rosettes — the channel's
own mark, tied to no country and no era, and readable at 32px. The same
footprint is the trail the character leaves behind it.

## The history tell

A vintage bowler. It dates the character without dating the money, and it pairs
with the magnifying glass: a Victorian clerk who investigates where the money
went. Brown and brass are the only non-green colours, so the green stays the
thing you recognise.

## No lettering

The note is blank, as the style rules require everywhere else in the frame:
wordless guilloche waves and ruled lines, no numerals, no serial, no denomination.
That is also what keeps it from reading as any real banknote.

## Files

- `mascot.svg` — master, full figure with the footprint trail
- `mascot_avatar.svg` — arms tucked in so nothing is clipped by a round crop
- `*.png` — 512px renders of each
