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
only "the Chronicler appears again" will render a different person.

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

## Writing tests here

Tests run with no GPU, no ComfyUI, no API keys, and no network. Mock at the HTTP
boundary. Prefer testing the *contract a failure must honour* — "a broken plan
grades every beat neutral", "a missing template returns None and the chain
continues" — over asserting on a happy path that a live run would catch anyway.

When a test encodes a real incident, say so in the docstring, with the log line
that reported it. Several tests in this repo exist because a symptom was
misdiagnosed for weeks; the docstring is what stops the next agent from
re-deriving the wrong cause.
