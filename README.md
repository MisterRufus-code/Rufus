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

```bash
bash scripts/setup.sh                 # venv, deps, DB, font, health check
# then edit config/keys.json with your real keys
source venv/bin/activate
python scripts/health_check.py        # verify everything is ready
```

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
| `RUFUS_VIDEO_SOURCE` | `pexels` / `sd` | `pexels` | Footage source (see below) |
| `RUFUS_RENDERER` | `ffmpeg` / `remotion` | `ffmpeg` | Render engine (see below) |
| `RUFUS_TTS` | `edge` / `xtts` | `edge` | Voice engine (see below) |
| `RUFUS_GPU` | `1` / unset | unset | Whisper CUDA + FFmpeg NVENC |
| `RUFUS_MIN_UPLOAD_SCORE` | `0`–`10` | `8` | Quality gate — only ≥N auto-uploads |
| `RUFUS_NICHE_OVERRIDE` | niche name | — | Force a niche for one run |

### Footage sources
- **`pexels`** — free stock footage, 7 candidates, GPT-4o Vision picks the best match. Needs a Pexels key.
- **`sd`** — local Stable Diffusion (Automatic1111). Generates images matching the script, upscales 2× with Real-ESRGAN, crops to 1080×1920, animates with Ken Burns. **Free forever, runs on a GTX 1060 6GB.** Start A1111 with `./webui.sh --api --xformers --medvram`, then set `SD_HOST` if not on localhost.

Both fall back to Pexels if they produce nothing, so a run never dies on footage.

### Render engines
- **`ffmpeg`** (v4.0 "cinematic edit") — cuts snap to **sentence boundaries** from Whisper timestamps with a punchy ~3s hook cut; synthesized **SFX layer** (sub-bass hit on the hook, whoosh on every cut, riser into the final beat — generated locally by `sfx_gen.py`, zero APIs); music **ducked dynamically** under the voice via sidechain compression; voice runs through a highpass → compressor → presence-EQ chain; final mix mastered to **-14 LUFS** (YouTube reference); retention progress bar + captions accented in the per-niche `accent_color`. Falls back to a simple hard-concat mix if the full graph errors, so renders never break.
- **`remotion`** — React engine: spring-physics word captions, smooth crossfades, retention progress bar, edge fades. Run `cd remotion && npm install` once. Falls back to FFmpeg on any error.

### Voice engines
- **`edge`** — Microsoft Edge TTS. Free, fast, cloud, no GPU. Reliable but slightly synthetic.
- **`xtts`** — Coqui XTTS v2, local. Near-ElevenLabs quality, free forever, ~3GB VRAM. Voice-clone from a 6s sample with `RUFUS_TTS_VOICE=/path/to/ref.wav`. Install: `pip install TTS`. Falls back to Edge on any error.

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
