# DIRECTION.md

Creative direction for the content this pipeline makes. **This file is yours.**
Edit it in plain English and it reaches the script writer and the storyboard on
the very next run — no code, no restart.

`AGENTS.md` tells coding agents how to work on this repo. This tells the
writing and storyboard models how you want the videos made.

**Where it goes:** injected into the script writer's system prompt right after
the niche description, and into the storyboard prompt. A per-channel file at
`config/direction/<channel>.md` is appended after this one, so channel-specific
direction adds to this rather than replacing it.

---

## Three things worth knowing before you edit

**1. Examples beat rules.** `config/gold_examples.json` says it in its own
note: *"the model mimics these more than any instruction."* If you can show the
thing you want with a two-line example instead of describing it in a paragraph,
do that — put it there, and it will outperform anything written here.

**2. Numbers here lose.** Word counts, sentence lengths and durations are
enforced by `config/script_standards.json` and by deterministic checks in
`script_writer.py`. Writing "keep it to 60 words" here will not shorten
anything; it will produce scripts rejected for being under the minimum. The run
log warns you if this file mentions a length.

**3. Keep it under 400 words.** The system prompt is already long. Past the cap
this file is truncated and the log says by how much. Two instructions that
disagree do not average out — the model drops both. This has happened here: the
prompt once said "split it into short sentences" while a gate demanded a
15-word sentence, and scripts were rejected for obeying the instruction.

---

## The direction

### Open on a moment, not a summary

The strongest line in any script is a thing that happened: a date, a place, one
named person doing one specific thing. Totals, shares and net worths are
**evidence you cut to afterwards**, never the way in.

> "In 1523 Jakob Fugger sent Charles V a letter demanding repayment."
> — a moment. You can see it, and you can film it.
>
> "Jakob Fugger held two percent of Europe's GDP."
> — true, specific, and nobody is doing anything.

A script that never lands in a place with a person in it is an encyclopedia
entry, however good its numbers are.

### Feeling comes from the event

Never state what someone felt, feared or intended. Show what they did, and let
the viewer supply the feeling. This is also the only version that survives the
fact-check: sources record what people **did**, almost never what they thought.

> "People carted wheelbarrows of worthless notes to the shops and still went
> home hungry" — vivid, and factual.

### Specifics over adjectives

A number, a name, a date or an object beats any amount of description. Cut
"significant", "dramatic", "remarkable" and put a figure there instead.

### Pictures show the sentence, not the topic

Every shot is the literal thing its line names. If the line is about copper,
the shot has copper in it. A picture of the general subject — a castle for
"power", a map for "influence" — is the most common way this goes wrong, and it
is worse than a plain shot of the actual object.

### One shot moves

Motion is an accent, not a texture. The beat carrying the moment gets the
moving shot; everything else holds. Nine moving shots stop being motion by the
third one.
