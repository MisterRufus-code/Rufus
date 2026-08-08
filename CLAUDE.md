# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Tests (no ComfyUI/A1111/GPU/API keys needed — pure-function + mocked-HTTP level)
python -m pytest -q                                    # full suite
python -m pytest tests/test_comfy_client.py -q          # one file
python -m pytest tests/test_comfy_client.py::test_render_image_returns_bytes_on_success -q   # one test

# Run the pipeline
python scripts/main.py --skip-upload      # render only, nothing leaves the machine
python scripts/main.py                    # full run: research → render → queue for approval
python scripts/main.py --niche money_history --topic "Bretton Woods"   # forced niche/topic

# Dashboard (approval queue, settings, live status)
python scripts/dashboard.py               # http://localhost:8765
```

Windows is the target deployment platform (RTX 3090); `run.bat` /
`run_scheduled.bat` / `run_dashboard.bat` wrap the same `main.py`/
`dashboard.py` entry points with the env vars a real run needs — read one
before assuming a flag or default. There is no lint/type-check config in
this repo; `python -m pytest -q` passing (990+ tests) is the correctness bar.

## Architecture

**7-step pipeline, all in `scripts/main.py`'s `run()`**: research a seed →
footage (stock or generate) → vision-pick the best clip (stock sources
only) → write+score the script → render → save to `rufus.db` → queue for
approval. Steps are numbered in the console output (`[ 1/7 ]` etc.) and
mirrored to `run_progress.py` for the dashboard's live status. Every
external step (research source, TTS backend, image/motion engine, render
engine, notification channel) degrades through an explicit fallback chain
instead of raising — a missing API key or offline local server should
never kill a run, only downgrade its quality. Read the fallback chain
printed at each step rather than assuming one backend is in use.

**Nothing uploads automatically.** Every video lands `upload_status='pending'`
in `rufus.db`; a human clicks Approve in the dashboard (`dashboard.py`'s
`approve_video`, the *only* code path that calls `youtube_uploader.upload`
in production — `main.py`'s own direct-upload branch only fires under
`RUFUS_AUTO_UPLOAD=1`, off by default). Both paths funnel through the same
`upload()` so anything wired there (CTA/source-citation comments,
cross-posting) applies regardless of which path triggered it.

**Config-driven, not code-driven, per niche/channel.** `config/niches.json`
defines each content niche (voice/system-prompt, `style_suffix`,
`video_source`, `character`, hashtags, CTA pool, etc.); `channel_config.py`
layers a channel's `niche_overrides` on top for multi-channel installs
(`config/channels.json`, legacy-shimmed to a synthesized `main_en` channel
when absent). `paths.py` is the single source of truth for every
filesystem root (media/debug/output/logs), all overridable via env vars —
never hardcode a path that paths.py already resolves.

**ComfyUI integration is template-only, never hand-wired.** `comfy_template.py`'s
module docstring is load-bearing: a new ComfyUI-backed engine (stills,
image-to-video, character-consistency) is never built by wiring a graph
from documentation. The channel owner runs the workflow in ComfyUI once,
sets the positive prompt to the literal placeholder `RUFUS_PROMPT`,
exports via "Export (API)", and drops the JSON in `config/`.
`comfy_template.prepare()` then substitutes prompt/image/seed/dims into
that exact proven graph. Every comfy-backed engine (`comfy_client.py`,
`hunyuan_client.py`, `wan_client.py`, `ltx_client.py`) follows this same
contract and is inert (returns `None`/`[]`, falls through the chain) until
its template file exists — that's intentional, not a bug to fix.

**Recurring-character system (`character_engine.py`) is generic per-niche,
cuts across three files.** A niche's `character` block in `niches.json`
drives: (1) text-level consistency — `main.py`'s `_build_sd_prompts`
injects a "describe the same person" clause into both the FLUX/comfy and
SD/token prompt branches; (2) image-level consistency — `comfy_client.py`
bootstraps a persistent reference portrait once per niche and feeds it
through an image-conditioning template (`config/character_stills_api.json`,
same proven-template contract as above) when one exists; (3) thumbnail
branding — `thumbnail_gen.py` badges the reference portrait into a corner
of every thumbnail once it's been bootstrapped. All three degrade
independently and silently to "no character" if their prerequisite is
missing — check `character_engine.enabled(niche)` before assuming any of
them is live for a given niche.

**Fact-checking and quality gates live in `script_writer.py`**, layered
rather than a single gate: hook grounding against the source seed, body
pre-checks (banned phrases/patterns, repeated numbers, em-dash overuse),
and a story-architect plan grounding pass before the draft even starts.
When adding a new script-quality rule, prefer a prompt-level nudge over a
new hard-fail gate — this codebase has hit real "wasted-generation
rejection ladder" bugs from stacking deterministic gates for stylistic
(non-correctness) preferences; hard gates are reserved for factual/
grounding correctness.

**Auth/roles**: `auth.py` defines `owner`/`partner`/`viewer` permissions,
consumed by `dashboard.py`'s route guards. Only `owner` may `approve`
(publish to YouTube) — this boundary should not move without explicit,
unambiguous instruction. Loopback (`127.0.0.1`) is not proof of identity
once `config/users.json` exists, since `tailscale serve` proxies all
traffic through localhost — auth checks must not special-case it.

**Env-var feature toggles** are the standing pattern for anything optional
or swappable (`RUFUS_STILLS_ONLY`, `RUFUS_CHARACTER_MODE`,
`RUFUS_VIDEO_SOURCE`, `RUFUS_RENDERER`, ...) — see the README's "Modes"
table for the full list before inventing a new on/off mechanism.
