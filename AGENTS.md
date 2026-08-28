# AGENTS.md

Guidance for any coding agent working in this repository. This is the canonical
file; `CLAUDE.md` points here so the two cannot drift apart.

Rufus is an autonomous YouTube Shorts pipeline: it researches a real source,
writes and scores a script, generates or fetches footage, renders a vertical
video, and queues it for a human to approve. It runs unattended on the owner's
Windows box, which shapes almost every convention below.

## Commands

```bash
# Tests — no ComfyUI, A1111, GPU or API keys needed (pure-function + mocked HTTP)
python -m pytest -q                                     # full suite
python -m pytest tests/test_comfy_client.py -q          # one file
python -m pytest tests/test_comfy_client.py::test_render_image_returns_bytes_on_success -q

# Run the pipeline
python scripts/main.py --skip-upload      # render only, nothing leaves the machine
python scripts/main.py                    # full run: research → render → queue for approval
python scripts/main.py --niche money_history --topic "Bretton Woods"

# Dashboard (approval queue, settings, live status)
python scripts/dashboard.py               # http://localhost:8765

# Preflight — what is actually installed and reachable right now
python scripts/health_check.py

# Why is a ComfyUI engine "off"? Names which of the four causes it is:
# server down, nodes missing, weights in a folder ComfyUI doesn't read,
# or the API export simply not done yet. Read-only.
python scripts/comfy_doctor.py            # every engine
python scripts/comfy_doctor.py wan_t2v    # one
```

There is no lint or type-check config. **`python -m pytest -q` passing is the
correctness bar** (1390+ tests). Windows is the deployment target (RTX 3090,
16GB system RAM); `run.bat` / `run_scheduled.bat` / `run_dashboard.bat` set the
env vars a real run needs — read one before assuming a flag or default.

## The one rule that explains most of this codebase

**Every external step degrades through an explicit fallback chain instead of
raising.** Research source, TTS backend, image and motion engines, render
engine, notification channel — a missing API key or an offline local server
must never kill a run, only downgrade its quality.

**The corollary is the part that gets forgotten: fail-open without fail-loud is
fail-silent.** Every bug of consequence found in this project has been the same
shape — a degraded path that nobody could see. Remotion failed on a Windows
`npx` spawn for months and the log said "falling back to FFmpeg", which read
like a Remotion problem. Reddit returned HTML to a JSON parser and the log said
"unreachable", which read like a network blip. An edit plan was computed on
every run and thrown away because only the renderer that never ran consumed it.

So when you add a fallback, also add the line that says **which** path was
taken and **why**, in words that name the fix. `warn("X unavailable")` is not
enough; `"X needs OAuth now — add reddit_client_id to config/keys.json"` is.

## Architecture

**7-step pipeline, all in `scripts/main.py`'s `run()`**: research a seed →
footage (stock or generated) → vision-pick the best clip (stock sources only) →
write and score the script → render → save to `rufus.db` → queue for approval.
Steps are numbered in console output (`[ 1/7 ]`) and mirrored to
`run_progress.py` for the dashboard's live status.

**Nothing uploads automatically.** Every video lands `upload_status='pending'`
in `rufus.db`; a human clicks Approve in the dashboard. `dashboard.py`'s
`approve_video` is the *only* code path that calls `youtube_uploader.upload` in
production — `main.py`'s direct-upload branch fires only under
`RUFUS_AUTO_UPLOAD=1`, off by default. Both funnel through the same `upload()`,
so anything wired there (CTA comments, source citations, cross-posting) applies
regardless of which path triggered it.

**Config-driven per niche and channel, not code-driven.** `config/niches.json`
defines each niche (voice and system prompt, `style_suffix`, `video_source`,
`character`, hashtags, CTA pool). `channel_config.py` layers a channel's
`niche_overrides` on top for multi-channel installs (`config/channels.json`,
legacy-shimmed to a synthesized `main_en` channel when absent). `paths.py` is
the single source of truth for every filesystem root, all overridable by env
var — **never hardcode a path that `paths.py` already resolves.**

**ComfyUI integration is template-only, never hand-wired.**
`comfy_template.py`'s module docstring is load-bearing. A new ComfyUI-backed
engine is never built by wiring a graph from documentation. The owner runs the
workflow in ComfyUI once, sets the positive prompt to the literal placeholder
`RUFUS_PROMPT`, exports via "Export (API)", and drops the JSON in `config/`.
`comfy_template.prepare()` substitutes prompt, image, seed and dims into that
exact proven graph. Every comfy-backed engine (`comfy_client.py`,
`hunyuan_client.py`, `wan_client.py`, `ltx_client.py`) follows this contract and
is **inert until its template file exists — intentional, not a bug to fix.**
`wan_client.py` is the deliberate exception: it builds its graph from native
ComfyUI nodes, which is why it needs no export and why a GGUF swap there needs
a code change rather than a filename change.

**The recurring-character system (`character_engine.py`) cuts across three
files.** A niche's `character` block drives text-level consistency
(`main.py`'s `_build_sd_prompts`), image-level consistency (`comfy_client.py`
bootstraps a reference portrait and feeds it through
`config/character_stills_api.json`), and thumbnail branding
(`thumbnail_gen.py`). All three degrade independently and silently to "no
character" — check `character_engine.enabled(niche)` before assuming any is
live. Naming a character in a prompt is **not** describing them: an image model
renders each beat from noise with no memory of the others, so a beat that says
only "the narrator appears again" will render a different person.

**No niche currently has a character switched on.** money_history ran one (a
hooded narrator) and the owner removed it: it took the first, middle and last
shot of every sequence, so three frames in ten went to a mascot instead of the
story. The mechanism is intact and the five SD niches still ship starter
mascots with `"enabled": false` — turning one on is a `config/niches.json`
edit, not a code change. Do not re-add a character to a niche without being
asked for one.

**Fact-checking and quality gates live in `script_writer.py`**, layered rather
than single: hook grounding against the source seed, body pre-checks (banned
phrases, repeated numbers, em-dash overuse), and a story-architect plan
grounding pass before the draft starts.

**The emotional map (`emotional_map.py`) is the shared read on how a beat
feels.** `edit_director.py` returns a tone per beat alongside its camera move;
`audio_gen.py` turns that into per-clip color grading, film grain, SFX
weighting, and tone-sized pauses in the voice. It is a lookup table, not an
agent — no extra model call. Everything in it degrades to `neutral`, which
grades to the niche's own base look.

**Auth and roles**: `auth.py` defines `owner`/`partner`/`viewer`, consumed by
`dashboard.py`'s route guards. Only `owner` may approve. **This boundary does
not move without explicit, unambiguous instruction.** Loopback (`127.0.0.1`) is
not proof of identity once `config/users.json` exists, because `tailscale serve`
proxies all traffic through localhost — auth checks must not special-case it.

**Env-var feature toggles** are the standing pattern for anything optional or
swappable (`RUFUS_STILLS_ONLY`, `RUFUS_BEAT_MOTION`, `RUFUS_CHARACTER_MODE`,
`RUFUS_RENDERER`, `RUFUS_FILM_GRAIN`, …). See the README's "Modes" table before
inventing a new on/off mechanism.

## Where the channel owner instructs the content

Four surfaces, in order of leverage. Reach for the strongest one that fits
before adding prose to a prompt in code — and note that **nothing written in
1-3 can beat 4**, because 4 is enforced by deterministic checks.

1. **`config/gold_examples.json`** — two full example scripts per niche,
   injected as few-shot. The strongest by a wide margin, and the file says so
   itself: *"the model mimics these more than any instruction, so they define
   the voice."* If a rule can be replaced by an example, replace it.
2. **`DIRECTION.md`** + **`config/direction/<channel>.md`** — the owner's
   standing creative direction in plain English, layered shared-then-channel
   (the same shape `channel_config.py` uses for `niche_overrides`). Reaches the
   script writer's system prompt and the storyboard prompt.
   `script_writer.load_direction()` is the single reader. Everything above the
   `## The direction` heading is for the human and is never sent.
3. **`config/niches.json`** → `gpt_system` (prose, per niche) and
   `style_suffix` (the visual look).
4. **`config/script_standards.json`** — word counts, sentence lengths,
   banned phrases, the opinion pool. Enforced in code by
   `script_writer._body_violations`, so it overrides all prose. Direction that
   states a length is warned about at load time for exactly this reason: "keep
   it to 60 words" does not shorten anything, it produces scripts rejected for
   being under `min_words`.

Adding another LLM stage is almost never the answer. The pipeline already runs
six with structured handoffs (hook factory → story architect → body writer →
fact gate → storyboard → edit director). Every quality gain in recent work came
from **enforcing instructions that already existed**, not from new roles — see
the storyboard's relevance check and the cadence/DELIVERY contradiction below.

## Conventions that have already cost a real bug

**Always state `encoding="utf-8"` on `read_text`, `write_text` and `open`.**
Without it Python uses the ANSI code page, which on the owner's Hebrew-locale
Windows box is cp1255. Every config file here is UTF-8 and several hold
em-dashes:

```python
"—".encode("utf-8").decode("cp1255") == "ג€”"
```

A CTA read out of `niches.json` reached the TTS backend as a Hebrew letter and
was **read aloud** in a finished English video, while every QC check reported
pass. `tests/test_file_encoding.py` fails the suite if a new bare call appears.

**Spawn executables by the path `shutil.which` returns, never by bare name.**
On Windows `npx` is `npx.cmd`; `which` finds it via PATHEXT, `CreateProcess`
does not. A readiness check passes and the spawn raises `WinError 2`. This
silently disabled Remotion and HyperFrames. `tests/test_npx_resolution.py`
guards it.

**Prefer a prompt-level nudge over a new hard-fail gate.** This codebase has hit
real "wasted-generation rejection ladder" bugs from stacking deterministic gates
for stylistic preferences. Hard gates are reserved for factual and grounding
correctness. A pacing or aesthetic check should be a warning that is impossible
to miss, not a rejection.

**Distinguish executable fields from cosmetic ones when validating model
output.** An unknown camera motion is a plan the renderer cannot perform, so the
whole plan is refused. An unknown tone is cosmetic, so it degrades to neutral.
Refusing good work over a cosmetic field trades a working feature for a new one.

**A number that depends on the video's shape belongs in
`scripts/video_format.py`, never in the module that needs it.** There are two
formats — a 40-second vertical Short and a nine-minute landscape explainer —
and every constant that differs between them is a profile field the readers
import. `1080, 1920` was written into seven modules before the second format
existed, and the sweep to remove it kept finding more: the alternative image
backends, the thumbnail composer, the manual image tool. The test file for it
now parses every script and fails on an assignment holding both numbers, and
imports every module under both formats to catch a profile key that does not
exist.

Three shapes of this bug, all of which have happened here:

- *The measurement contradicting the feature.* `run_review` scored runs
  against a beat rule the pipeline had stopped using, then against the Shorts
  ceiling on a nine-minute script, where it measured 24 pictures as generous.
- *The gate asking for what the generator was told not to produce.* The script
  rubric disqualified any script without a Shorts loop line, capping every
  long-form script at 4/10 — a device `longform_writer` is explicitly
  instructed not to write.
- *Taking the first N of something.* `insert_director` cut its candidate list
  at 28, which is the whole video at forty seconds and the first ninety
  seconds at nine minutes. When a limit binds, SPREAD rather than truncate;
  the part a truncation drops is the part nobody scrolls to.

**Two renderers ship this channel, and they must agree.** `audio_gen`'s FFmpeg
path and `remotion_renderer` both produce finished videos, so a caption rule,
a caption *casing*, an emphasis colour or a word-timing stream written into one
of them is a video that looks different depending on which engine drew it —
and nobody watches both. Put the decision in one function and have both call
it. Note the shape of that fix: the captions are grouped for READING, while the
insert planner needs one word at a time, so they are two lists built from one
transcript, not one list used twice.

**A per-video decision belongs on the page that makes the video; a
per-channel one belongs in Settings.** The caption look sat in
`video_format.PROFILES` next to the frame size and the QC bounds, so changing
it changed every future video at once and nobody found it. It is now a preset
in `scripts/caption_styles.py` picked on the render page, with Settings holding
only the default for unattended runs. Two rules keep that honest: the default
preset overrides *nothing*, so the per-format numbers still stand and the cron
renders are unchanged; and preset sizes are shares of the FRAME HEIGHT rather
than multiples of the format's own number — long-form already ships the
broadcast look, so scaling its 58px by 0.45 to "make it broadcast" lands at
26px, produced by asking for the thing it was already doing.

**When only one of several endings makes a sound, the silent ones are where
work goes to die.** A render can finish six ways — queued for review, uploaded,
held by QC, held on facts, held on the scene plan, held on score — and exactly
one of them notified anybody. A video the QC held finished in complete silence
and sat on disk until somebody thought to look, which is not something you find
out about, it is something you eventually notice. Every branch now records
which ending it reached and one place announces it, with a test asserting the
map covers every value `main.py` can set.

## Writing tests here

Tests run with no GPU, no ComfyUI, no API keys, and no network. Mock at the HTTP
boundary. Prefer testing the *contract a failure must honour* — "a broken plan
grades every beat neutral", "a missing template returns None and the chain
continues" — over asserting on a happy path that a live run would catch anyway.

When a test encodes a real incident, say so in the docstring, with the log line
that reported it. Several tests in this repo exist because a symptom was
misdiagnosed for weeks; the docstring is what stops the next agent from
re-deriving the wrong cause.

## The word-synced insert layer

A second visual format, off nothing and additive: instead of one picture per
sentence, a small picture per **noun**, landing on the second that noun is
spoken. `insert_director.py` plans it, `comfy_client.render_inserts` draws it,
`Short.tsx`'s `InsertLayer` pops it in, `sfx_gen`'s `pop` marks it.

**The pipeline can do this because it already transcribes its own voiceover.**
`remotion_renderer` runs Whisper over the finished audio for captions, so
word-level timings exist before the planner runs — the expensive half of this
format is already paid for.

**The planner never renders.** `python scripts/insert_director.py "<script>"`
prints the plan in a second with no GPU, no ComfyUI and no network, which is
the point: argue with a bad plan before spending GPU on it.

**Insert images are drawn while the stills model is still loaded**, between the
beats and the `/free` that precedes any motion engine. Twenty-eight extra
renders on a warm model, not twenty-eight model loads — the measured cost on
this box is loading, not sampling (see wan_t2v: 2 seconds of sampling inside a
330-second clip).

Every rejection rule in `insert_director` exists because a real script leaked
something undrawable, and each names what leaked. Do not "simplify" them
without re-running the planner on a real script first.
