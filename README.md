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
| `RUFUS_SUPERVISOR` | `0`/`1` | `1` | Per-stage retry judge (see below) — set `0` to skip the extra GPT calls |
| `RUFUS_IMG2VID` | `0`/`1` | `1` | SVD image-to-video: generated stills become real motion clips (see below) |
| `RUFUS_IMG2VID_ENGINE` | `auto` / `comfy` / `diffusers` | `auto` | SVD engine — `auto` prefers ComfyUI, falls back to in-process diffusers |
| `RUFUS_FACE_RESTORE` | `auto` / `0` / `1` | `auto` | Face restoration on FLUX stills if a restore node is installed (see below); `0` forces off |
| `RUFUS_FACE_RESTORE_MODEL` | weights filename | `GFPGANv1.4.pth` | Restore model file (e.g. `codeformer-v0.1.0.pth`) |
| `RUFUS_DEBUG` | `1` / unset | unset | Save every run's script, raw voiceover, and FLUX keyframes+prompts to `media_library/debug/<run_id>/` — one folder to review the whole pipeline before it reaches YouTube. Auto-cleaned after ~30 days. |

Each niche picks its own source via `"video_source"` in `config/niches.json`
(default: all niches → `sd`). `RUFUS_VIDEO_SOURCE` overrides it for one run.

### Footage sources
- **`comfy`** — local **ComfyUI + FLUX.1-dev**, the top-quality engine (needs ~24GB VRAM, e.g. **RTX 3090**). One photoreal image per beat at 832×1472 → Lanczos upscale → crop 1080×1920, with perceptual-hash dedup so no scene repeats. Start ComfyUI with `--listen` and drop `flux1-dev-fp8.safetensors` in `models/checkpoints/`. Tune via `COMFY_HOST`/`COMFY_MODEL`. Falls back: **comfy → sd → diffusers → pexels**.
  - **Image-to-video (Wan 2.2 — primary)**: when the six ComfyUI-template files are installed (`wan2.2_i2v_{high,low}_noise_14B_fp8_scaled` in `models/diffusion_models/`, the two `lightx2v_4steps` LoRAs in `models/loras/`, `umt5_xxl_fp8` in `models/text_encoders/`, `wan_2.1_vae` in `models/vae/` — ~35GB one-time via the ComfyUI "Wan 2.2 14B Image to Video" template), every still is animated by **Wan 2.2 14B** (`wan_client.py`): dramatically better temporal consistency than SVD (no warping/melting of fine detail on rigid objects), and each beat gets a **text motion prompt** derived from its image prompt — steered toward camera/ambient motion only (never a one-way completing action like a page turn or gesture, since every clip loops forward-then-reversed and a one-way action played backward visibly undoes itself each cycle). Default mode uses real classifier-free guidance with no speed-up LoRA (`cfg 3.5`, `12` steps split evenly across the high/low-noise experts, the model's own tuned negative prompt) — verified against an actual API export of a proven-clean test render (that export's default toggle was "no LoRA"), trimmed from its 20-step default for a ~1.5-2h/video motion-generation budget instead of 3h+. A faster opt-in path (`RUFUS_WAN_LORA=1`, the lightx2v 4-step/cfg-1.0 distillation, ~5-6 min/clip) trades some of that verified quality for speed. Faces still blur/glitch under Wan motion even though rigid objects don't (a live report caught this) — the same face-skip heuristics as SVD apply here too, opt back in with `RUFUS_WAN_FACE_MOTION=1`. Knobs: `RUFUS_WAN=0` disables, `RUFUS_WAN_FRAMES` (81), `RUFUS_WAN_STEPS` (12), `RUFUS_WAN_CFG` (3.5), `RUFUS_WAN_LORA` (0), `RUFUS_WAN_FACE_MOTION` (0), `RUFUS_WAN_TIMEOUT` (1800s). Engine chain per clip: **wan → svd → Ken Burns** — any failure walks down, a clip is never lost.
  - **Image-to-video (SVD — fallback)**: every generated still is animated into a **real motion clip** — 25 frames at 576×1024, interpolated to 30fps, ping-pong-looped seamlessly, upscaled to 1080×1920 (`svd_client.py`). Two engines, resolved automatically once per run: **ComfyUI** (drop `svd_xt.safetensors` in `models/checkpoints/` — one-time ~9GB download from [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)) or **in-process diffusers** (`pip install diffusers transformers accelerate`, needs CUDA; downloads the same weights to the HF cache — no manual file placement). Applies to the `comfy` AND `diffusers` footage sources. Adds ~30-60s GPU time per beat on a 3090. Any per-image failure falls back to the classic **Ken Burns** zoom so a clip is never lost. Knobs: `RUFUS_IMG2VID=0` disables, `RUFUS_IMG2VID_ENGINE` forces an engine, `COMFY_SVD_MODEL` / `RUFUS_SVD_DIFFUSERS_MODEL`, `RUFUS_SVD_MOTION` (1-255, more = more motion), `RUFUS_SVD_FRAMES`/`RUFUS_SVD_FPS`/`RUFUS_SVD_STEPS`. Note: SVD conditions on the image only — it takes **no text prompt** (the model card's generic `DiffusionPipeline` + prompt snippet is misleading; the correct class is `StableVideoDiffusionPipeline`). SVD also isn't face-aware — animating a shot with a visible face can warp/melt facial features across frames — so beats whose own generation prompt reads as a person shot (contains "portrait" or "face") skip SVD entirely and use Ken Burns instead, which can't distort anything.
  - **Face restoration (optional)**: diffusion models render human faces worst, so — beyond the prompt steering that already avoids the hardest close-ups — each FLUX still can be routed through a **CodeFormer / GFPGAN** restore node to clean up eyes/teeth/skin before animation. **Strictly opt-in and fail-safe**: it activates only if a supported node is installed in ComfyUI, and any failure falls back to the plain graph with the same seed, so restoration can only add quality, never cost a clip. Install one restore node via **ComfyUI-Manager** (search "Facerestore CF" → provides `FaceRestoreCFWithModel`, or the ReActor pack → `ReActorRestoreFace`) and put the weights (`GFPGANv1.4.pth` / `codeformer-v0.1.0.pth`) in `models/facerestore_models/`. Rufus auto-detects it and prints `face restore: ON via …`. Tune with `RUFUS_FACE_RESTORE` (`0` to force off), `RUFUS_FACE_RESTORE_MODEL`, `RUFUS_FACE_FIDELITY` (0-1, higher = truer to the original).
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
- **`kokoro`** — Kokoro-82M in-process (CPU, free, natural). Auto-selected if the `kokoro` package is installed. `pip install kokoro soundfile`. Kokoro has no SSML/prosody control, so delivery comes from punctuation: a silence is inserted after each line sized to its trailing punctuation (longest after an em-dash/ellipsis "beat", shortest after a comma) — `RUFUS_KOKORO_SPEED` (default `1.0`) tunes playback rate. `script_writer.py`'s prompt is written to lean on this (dashes/ellipses before a reveal, hard stops instead of comma run-ons).
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

## Supervisor (per-stage retry judge)

The quality gate at upload time catches a bad *script*, but a thin research
seed or off-target image prompts burn a full render before anything catches
them. `scripts/supervisor.py` adds two early gpt-4o-mini judge calls that can
reject and force **one** retry of just that stage:

- **After research** — rejects a seed with no concrete facts to build a story on (retries `get_seed`).
- **After scripting** — a **fact-check**: compares the finished script's names/numbers/dates against the source seed (the script prompt forbids inventing them; this verifies GPT complied). On rejection the script is rewritten once with the objection fed back; if the rewrite is *still* flagged, the video renders but the **upload is held** with the specific claim printed — wrong facts never publish themselves. Wisdom-quote seeds keep their documented-history allowance.
- **After beat-prompt writing, before FLUX/SD/diffusers generation** — rejects near-duplicate or off-topic image prompts (retries `_build_sd_prompts`, which is non-deterministic so a retry actually differs).

Each call is a few hundred tokens (a fraction of a cent) and fails **open** —
no key, an API error, or a malformed reply always approves, so a broken judge
can never block a render. Set `RUFUS_SUPERVISOR=0` to skip it entirely.

---

## Operations

```bash
python scripts/health_check.py        # pre-flight: deps, keys, config, disk, ComfyUI+FLUX checkpoint
python scripts/review_scripts.py      # browse generated scripts from the DB
python scripts/analyze_scripts.py     # script-writer funnel/cost/score analysis
python scripts/analytics_fetcher.py   # pull YouTube metrics into the DB (cron daily)
python scripts/feedback_analyzer.py   # turn metrics into config/learnings.json
python -m pytest tests/ -q            # test suite
```

### Go live (daily automation, Windows)

One command turns the machine autonomous — one Short per day, hands-off:

```powershell
.\schedule_daily.ps1                 # every day at 13:00
.\schedule_daily.ps1 -Time "09:30"   # or pick the hour
.\schedule_daily.ps1 -Unregister     # stop
```

Requirements at run time: PC on, ComfyUI running (FLUX engine). The scheduled
run uses `run_scheduled.bat` — full pipeline **including upload**, protected by
the built-in gates: script must score ≥8/10, output must pass QC, and uploads
are **private** (scheduled to the next peak hour) so nothing goes public
without you.

**Daily 2-minute checklist:** open `media_library\output\` → skim the newest
video + its `.qc.json` → if it's good, it publishes itself at the scheduled
peak hour (or flip it public in YouTube Studio); if it's held, the log says
exactly why (`logs\rufus_YYYYMMDD.log`).

---

## Security invariants

- `config/keys.json` is **gitignored** — never commit real keys.
- YouTube uploads default to **private**.
- The quality gate (`RUFUS_MIN_UPLOAD_SCORE`, default 8) holds weak scripts back for review.
- Every upload sets `status.containsSyntheticMedia=True` (`youtube_uploader.py`) — YouTube's altered/synthetic-content disclosure policy requires self-declaring this for realistic AI-generated video, and every Rufus video qualifies (GPT script, FLUX/SVD imagery and motion, synthesized voice). Not optional/configurable — it's always true for this pipeline's output.
