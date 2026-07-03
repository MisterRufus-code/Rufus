# Rufus — Autonomous YouTube Shorts Machine

Rufus produces faceless YouTube Shorts end to end: it researches a real story,
writes a hook-first script, picks or generates matching footage, voices it,
captions it, scores it, and (if it clears the quality bar) uploads it to
YouTube — all from one command.

```bash
python scripts/main.py            # full run: research → render → upload
python scripts/main.py --skip-upload   # render only, nothing leaves the machine
```

---

## Pipeline (7 steps)

1. **Research** — a real seed: Reddit story → Hacker News → wisdom quote fallback.
2. **Footage** — Pexels stock or local Stable Diffusion (see below).
3. **Vision pick** — GPT-4o Vision picks the clip that matches the *story*, not just the niche.
4. **Script** — hook-first, 3-beat arc (Setup → Turn → Payoff), scored 0–10.
5. **Render** — TTS voice + Whisper word-captions + Ken Burns + music → 1080×1920 mp4.
6. **Save** — every video + script + score logged to `rufus.db` (SQLite).
7. **Upload** — YouTube **private**, scheduled to the next peak ET hour. Only scripts ≥ the quality bar upload.

---

## Setup

**Linux / macOS:**
```bash
bash scripts/setup.sh                 # venv, deps, DB, font, health check
# then edit config/keys.json with your real keys
source venv/bin/activate
python scripts/health_check.py        # verify everything is ready
```

**Windows 11 (+ RTX 3090):**
```powershell
.\setup_windows.ps1                   # venv, deps, ffmpeg check, prints GPU-stack cmds, health check
# then edit config\keys.json with your real keys
.\run.bat --skip-upload               # daily run (ComfyUI/FLUX + GPU by default)
```
Fully cross-platform: locking uses `filelock` (no POSIX `os.kill`/`fcntl`), fonts resolve
`C:\Windows\Fonts` plus the bundled `assets/fonts/Anton-Regular.ttf`. You need **ffmpeg on
PATH** (the script tells you how) and, for the FLUX engine, **ComfyUI** running.

Dependencies are split: `requirements.txt` is the lean core (always installed);
`requirements-optional.txt` holds the heavy local-ML extras (in-process Kokoro voice,
MusicGen music, Diffusers images, pytrends, praw) — **not needed** for the
ComfyUI + Docker-Kokoro setup, install only what you want.

`config/keys.json` (never commit it — it's gitignored) needs at minimum:

```json
{ "openai": "sk-...", "pexels": "your_pexels_key" }
```

Optional keys: `vimeo`, `pixabay`, `jamendo_client_id` (music), `reddit_client_id`/`reddit_client_secret` (better Reddit research).

**Music needs no key**: the chain is Jamendo (if key set) → archive.org (free) →
**locally synthesized ambient bed** (`music_gen.py` — per-niche chord-progression pads
generated with FFmpeg, zero APIs, zero licenses). Every render gets music under the voice.

---

## Modes (environment variables)

Everything is free except OpenAI credits. Mix and match:

| Variable | Values | Default | What it does |
|---|---|---|---|
| `RUFUS_VIDEO_SOURCE` | `comfy` / `sd` / `diffusers` / `pexels` | per-niche (`niches.json`) | Footage source — overrides the niche's `video_source`; falls back down the chain |
| `COMFY_HOST` / `COMFY_MODEL` | URL / checkpoint | `http://localhost:8188` / `flux1-dev-fp8.safetensors` | ComfyUI server + FLUX checkpoint |
| `RUFUS_RENDERER` | `ffmpeg` / `remotion` | `ffmpeg` | Render engine (see below) |
| `RUFUS_TTS` | `edge` / `kokoro` / `kokoro_api` / `xtts` / `elevenlabs` | auto (`kokoro` if installed, else `edge`) | Voice engine (see below) |
| `KOKORO_API_URL` | URL | `http://localhost:8880` | Kokoro-FastAPI service (for `kokoro_api`) |
| `RUFUS_GPU` | `1` / unset | unset | Whisper CUDA + FFmpeg NVENC |
| `RUFUS_MIN_UPLOAD_SCORE` | `0`–`10` | `8` | Quality gate — only ≥N auto-uploads |
| `RUFUS_NICHE_OVERRIDE` | niche name | — | Force a niche for one run |

Each niche picks its own source via `"video_source"` in `config/niches.json`
(default: all niches → `sd`). `RUFUS_VIDEO_SOURCE` overrides it for one run.

### Footage sources
- **`comfy`** — local **ComfyUI + FLUX.1-dev**, the top-quality engine (needs ~24GB VRAM, e.g. **RTX 3090**). One photoreal image per beat at 832×1472 → Lanczos upscale → crop 1080×1920 → Ken Burns, with perceptual-hash dedup so no scene repeats. Start ComfyUI with `--listen` and drop `flux1-dev-fp8.safetensors` in `models/checkpoints/`. Tune via `COMFY_HOST`/`COMFY_MODEL`. Falls back: **comfy → sd → diffusers → pexels**.
- **`diffusers`** — in-process HuggingFace Diffusers (SDXL-Turbo by default) — no A1111 server needed. Lighter than FLUX; good when ComfyUI/A1111 aren't running. `RUFUS_DIFFUSERS_MODEL` selects the model.
- **`sd`** (default) — local Stable Diffusion (Automatic1111). Splits the script into **spoken beats** and generates **one content-matched image per beat, in order** — so when the narrator talks about stocks, the screen shows stocks (the renderer cuts on sentence boundaries, keeping image and voice in sync). Each image is upscaled 2× with Real-ESRGAN, cropped to 1080×1920, and animated with Ken Burns. Every image is **perceptual-hash de-duplicated** (aHash + regenerate) so none visibly repeats within a video. Ultra-detailed prompts tuned for Realistic Vision v5.1 with a rotating camera anchor (macro → wide → medium → aerial). **Free forever, runs on a GTX 1060 6GB.** Start A1111 with `./webui.sh --api --xformers --medvram`, then set `SD_HOST` if not on localhost. `SD_CLIPS` caps the scene count (default 6).
- **`pexels`** — free stock footage, 7 candidates, GPT-4o Vision picks the best match. Needs a Pexels key. Automatic fallback when A1111 isn't running.

Fallback chain so a run never dies on footage: **comfy → sd → diffusers → pexels**.

> _Optional/unwired:_ `scripts/hyperframes_client.py` (HeyGen HyperFrames HTML→MP4 motion-graphics) stays on disk for a possible future data-viz channel but is **not** in the active source routing — the focus is photoreal SD.


### Render engines
- **`ffmpeg`** (v4.0 "cinematic edit") — cuts snap to **sentence boundaries** from Whisper timestamps with a punchy ~3s hook cut; synthesized **SFX layer** (sub-bass hit on the hook, whoosh on every cut, riser into the final beat — generated locally by `sfx_gen.py`, zero APIs); music **ducked dynamically** under the voice via sidechain compression; voice runs through a highpass → compressor → presence-EQ chain; final mix mastered to **-14 LUFS** (YouTube reference); retention progress bar + captions accented in the per-niche `accent_color`. Falls back to a simple hard-concat mix if the full graph errors, so renders never break.
- **`remotion`** — React engine: spring-physics word captions, smooth crossfades, retention progress bar, edge fades. Run `cd remotion && npm install` once. Falls back to FFmpeg on any error.

### Voice engines
- **`edge`** — Microsoft Edge TTS. Free, fast, cloud, no GPU. Reliable but slightly synthetic.
- **`kokoro`** — Kokoro-82M in-process (CPU, free, natural). Auto-selected if the `kokoro` package is installed. `pip install kokoro soundfile`.
- **`kokoro_api`** — **Kokoro-FastAPI** over HTTP — same voice, zero native install (ideal on Windows). Run once: `docker run -d -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2` (or the `-gpu` image). Tune with `KOKORO_API_URL` / `RUFUS_KOKORO_VOICE` (default `am_adam`). Falls back to Edge on any error.
- **`xtts`** — Coqui XTTS v2, local. Near-ElevenLabs quality, free forever, ~3GB VRAM. Voice-clone from a 6s sample with `RUFUS_TTS_VOICE=/path/to/ref.wav`. Install: `pip install TTS`. Falls back to Edge on any error.
- **`elevenlabs`** — cloud, most natural (~$0.10/video). Needs `elevenlabs` key in `config/keys.json`.

---

## Common recipes

```bash
# Highest quality on a GTX 1060, fully free, no upload
RUFUS_VIDEO_SOURCE=sd RUFUS_RENDERER=remotion RUFUS_TTS=xtts \
  python scripts/main.py --skip-upload

# Default fast path with stock footage
python scripts/main.py --skip-upload

# One video per niche in the schedule
python scripts/main.py --rotate --skip-upload

# Cron: today's scheduled niche, auto-upload if it scores ≥8
python scripts/main.py --scheduled

# GPU instance (CUDA Whisper + NVENC encoding)
RUFUS_GPU=1 python scripts/main.py
```

---

## Niches

Configured in `config/niches.json` (`finance`, `motivation`, `mindset`,
`business`, `personal_development`). Switch the active niche:

```bash
python scripts/switch_niche.py motivation
python scripts/switch_niche.py --list
```

---

## Operations

```bash
python scripts/health_check.py        # pre-flight: deps, keys, config, disk
python scripts/review_scripts.py      # browse generated scripts from the DB
python scripts/analyze_scripts.py     # script-writer funnel/cost/score analysis
python scripts/analytics_fetcher.py   # pull YouTube metrics into the DB (cron daily)
python scripts/feedback_analyzer.py   # turn metrics into config/learnings.json
python -m pytest tests/ -q            # test suite
```

---

## Security invariants

- `config/keys.json` is **gitignored** — never commit real keys.
- YouTube uploads default to **private**.
- The quality gate (`RUFUS_MIN_UPLOAD_SCORE`, default 8) holds weak scripts back for review.
