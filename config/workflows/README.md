# Candidate workflows

Drop a ComfyUI **API export** in this folder and it becomes a column in
`python scripts/workflow_bench.py`. `config/stills_api.json` — whatever the
channel is rendering with today — is always the first column, so every
candidate is measured against what already ships rather than against nothing.

    python scripts/workflow_bench.py --check    validate the exports, no GPU
    python scripts/workflow_bench.py            render the grid
    dashboard → 🔬 Bench                        look at it

## Exporting one

In ComfyUI: build the graph, set the **positive prompt text to the single word
`RUFUS_PROMPT`**, then *Workflow → Export (API)*. Not the plain Export — that
saves the editor's layout, which this cannot run.

Three things `--check` will tell you are missing, and each one costs you
something real:

- **no `RUFUS_PROMPT`** — the bench has nowhere to put the prompt.
- **no negative text node** wired to a sampler's `negative` input — the
  negative prompt lands nowhere. This is not theoretical: a real gallery came
  back with readable sentences written across two frames while the negative
  led with `text, letters, words`.
- **no plain width/height** on any node — the frame size cannot be set, so the
  export's own resolution wins and the pictures come out the wrong shape.

## Change ONE thing at a time

Name the file after the thing that is different:

    zimage-turbo.json
    zimage-turbo-inklora-0.8.json
    zimage-turbo-inklora-0.8-steps12.json

Two changes at once produce a winner you cannot explain, and a winner you
cannot explain is one you cannot reproduce next month. The knobs worth moving,
roughly in order of how much they change the look:

| knob | what it decides |
|---|---|
| checkpoint | almost everything — the model's own idea of what a drawing is |
| style LoRA + weight | whether the look is the model's or yours |
| steps | detail and line confidence; turbo models want few, others want many |
| CFG | how literally the prompt is obeyed. Too high burns the colour out |
| sampler / scheduler | the texture of the line; worth trying last |

## What the probes are for

The six test prompts are not a pretty scene. Each one is a defect this project
has actually shipped, so a workflow gets measured against the ways this channel
has been let down:

| probe | the failure it looks for |
|---|---|
| `face` | ten shots of a country losing its money came back as ten mild smiles |
| `animal` | the stick-figure/real-animal contrast that IS this style |
| `action` | thirteen frames of sixteen had nobody doing anything |
| `writing_surface` | readable lettering got through twice in one gallery |
| `crowd` | two frames came back as six-panel contact sheets |
| `weather_place` | every background pale beige, against an explicit instruction |

The grid gives you a number per column — how many probes passed the automatic
checks, and the mean seconds — but the pictures decide. The numbers exist so
the obvious failures are not something to argue about.
