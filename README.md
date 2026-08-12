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
7. **Review queue** — nothing uploads automatically. Every video lands `pending` in the dashboard; approving there uploads to YouTube **private**, scheduled to the next peak ET hour. (`RUFUS_AUTO_UPLOAD=1` restores the old fully-automatic behavior, gated on the same quality bar.)

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
.\setup_windows.ps1                   # venv, deps, ffmpeg check, optional Remotion install, GPU-stack cmds, health check
# then edit config\keys.json with your real keys
.\run.bat --skip-upload               # daily run (ComfyUI + GPU by default)
```
Fully cross-platform: locking uses `filelock` (no POSIX `os.kill`/`fcntl`), fonts resolve
`C:\Windows\Fonts` plus the bundled `assets/fonts/Anton-Regular.ttf`. You need **ffmpeg on
PATH** (the script tells you how) and, for the `comfy` stills/motion engines, **ComfyUI** running.

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
| `COMFY_HOST` | URL | `http://localhost:8188` | ComfyUI server address |
| `RUFUS_STILLS_ONLY` | `0`/`1` | `0` | One switch for images-only mode: overrides Wan, HunyuanVideo, AND SVD at once — every beat is just its stills-model image with a Ken Burns zoom. No motion-model GPU time at all. |
| `RUFUS_RENDERER` | `ffmpeg` / `remotion` | `ffmpeg` | Render engine — `remotion` uses the React/Node engine (`remotion/`) for spring-physics captions, crossfades, and a retention progress bar; falls back to `ffmpeg` on any failure. Needs `cd remotion && npm install` once (both setup scripts do this automatically if Node.js is present). |
| `RUFUS_TTS` | `edge` / `kokoro` / `kokoro_api` / `xtts` / `elevenlabs` | auto (`kokoro` if installed, else `edge`) | Voice engine (see below) |
| `KOKORO_API_URL` | URL | `http://localhost:8880` | Kokoro-FastAPI service (for `kokoro_api`) |
| `RUFUS_GPU` | `1` / unset | unset | Whisper CUDA + FFmpeg NVENC |
| `RUFUS_MIN_UPLOAD_SCORE` | `0`–`10` | `8` | Quality gate — only ≥N auto-uploads |
| `RUFUS_NICHE_OVERRIDE` | niche name | — | Force a niche for one run |
| `RUFUS_SUPERVISOR` | `0`/`1` | `1` | Per-stage retry judge (see below) — set `0` to skip the extra GPT calls |
| `RUFUS_IMG2VID` | `0`/`1` | `1` | SVD image-to-video: generated stills become real motion clips (see below) |
| `RUFUS_IMG2VID_ENGINE` | `auto` / `comfy` / `diffusers` | `auto` | SVD engine — `auto` prefers ComfyUI, falls back to in-process diffusers |
| `RUFUS_DEBUG` | `1` / unset | unset | Also print verbose per-clip progress. Every run's script, raw voiceover, and keyframes+prompts are ALWAYS saved to `media_library/debug/<run_id>/` regardless of this flag — the permanent quality-review record, never auto-deleted. |
| `RUFUS_KENBURNS_ZOOM` | `0.0`–`1.0` | `0.06` | Ken Burns zoom range on a still (e.g. `0.06` = 1.00x→1.06x); pan drift scales with it too. Kept deliberately subtle by default — a heavy pan/zoom reads as fake on top of strong stills. |
| `RUFUS_FRESH_IMAGES` | `0`/`1` | `1` | Cross-run visual freshness: recent runs' image prompts are fed to the prompt-writer as a do-not-repeat list, and perceptual hashes of past images pre-seed the dup check so look-alike frames regenerate. `0` disables both. |
| `RUFUS_NTFY_TOPIC` | topic | — | **Phone ping when a video needs approval.** Free, no account: install the ntfy app, subscribe to a RANDOM topic (the topic is the shared secret), set it here. Pushover (`RUFUS_PUSHOVER_TOKEN`/`_USER`) and Telegram (`RUFUS_TELEGRAM_TOKEN`/`_CHAT`) also supported; `RUFUS_NOTIFY=0` disables. |
| `RUFUS_DASHBOARD_URL` | URL | — | Included in the notification so the phone deep-links straight to that video's review page, e.g. `http://192.168.1.20:8765`. |
| `RUFUS_PUBLISH_TIKTOK` | `0`/`1` | `0` | Cross-post to TikTok after a successful YouTube upload on Approve. Best-effort — a TikTok failure never turns a good YouTube upload into an error. |
| `RUFUS_MEDIA_DIR` | path | `<repo>/media_library` | **Send bulky output to another drive.** Moves the whole media tree (debug record, finished mp4s, cache/temp/music) — e.g. `W:\Rufus\media` to keep a small SSD free. |
| `RUFUS_DEBUG_DIR` / `RUFUS_OUTPUT_DIR` / `RUFUS_LOG_DIR` | path | under `RUFUS_MEDIA_DIR` | Split the review record / finished videos / logs across drives individually; the narrower var wins over `RUFUS_MEDIA_DIR`. |
| `RUFUS_STILLS_DETAIL` | text | (built-in) | Photographic direction appended to every stills prompt — lens/aperture, lighting falloff, and the micro-surface detail (grain, dust, worn edges, micro-scratches) that makes a frame read as a real photograph. Written as prose on purpose: the stills encoder is an LLM (Qwen3-4B), which reads description, **not** the SD1.5 `8k, masterpiece` tag idiom — that's out-of-distribution here and crowds out the subject. Set empty to disable. |
| `RUFUS_SCRIPT_CYCLES` | `1`+ | `3` | **Loop until the script is good.** Each cycle is a COMPLETE fresh attempt (new hook → new angle → new body → fact gate), and the previous cycle's rejection is fed forward so the hook factory stops reaching for the same unsupportable claim. Stops as soon as a cycle clears both the score bar and the fact gate; `1` = old single-pass behavior. |
| `RUFUS_SCRIPT_MAX_COST` | USD | `0.30` | Hard ceiling across all cycles — a topic whose source genuinely can't support an interesting claim can't run up a bill. |
| `RUFUS_SCRIPT_ARCHITECT` | `0`/`1` | `1` | One extra cheap GPT call before drafting: plans the spine fact, the turn, and why the story matters *now* — the draft writes to that plan instead of blind. `0` skips it (draft goes straight from pre-analysis, as before this feature). |
| `RUFUS_AUTO_UPLOAD` | `0`/`1` | `0` | `0` (default): nothing uploads automatically — every video is queued `pending` for approval in the dashboard. `1`: restores the old fully-automatic behavior (uploads immediately once score/QC/facts all pass). |
| `RUFUS_LTX` | `0`/`1` | `1` | **LTX-2.3 — the fast motion engine.** Tried before Hunyuan in the chain. Needs a one-time ComfyUI export to `config/ltx_i2v_api.json` (prompt set to `RUFUS_PROMPT`); inert until then. Knobs: `RUFUS_LTX_W/H` (832×1472), `RUFUS_LTX_FRAMES` (121), `RUFUS_LTX_TIMEOUT` (900s). ⚠ Verify LTX-2.3's licence covers commercial use before publishing with it. |
| `RUFUS_HUNYUAN` | `0`/`1` | `1` | HunyuanVideo 1.5 as the FACE motion engine (animates the face shots Wan skips). Needs a one-time ComfyUI template export to `config/hunyuan_i2v_api.json` — see `hunyuan_client.py` header. Knobs: `RUFUS_HUNYUAN_W/H` (480×832), `RUFUS_HUNYUAN_FRAMES` (121), `RUFUS_HUNYUAN_TIMEOUT` (1800s). |
| `RUFUS_STILLS_TEMPLATE` | `0`/`1` | `1` | **The stills model — required, no built-in fallback.** Export a commercial-safe ComfyUI image workflow (Z-Image-Turbo recommended, Apache 2.0) with the positive prompt set to `RUFUS_PROMPT` to `config/stills_api.json`. There's deliberately no hardcoded fallback model here — a "safety net" that silently renders a non-commercial model into a monetized video isn't safe. With no export, `comfy` mode renders nothing and falls through to `sd`/`diffusers`/`pexels`. |
| `RUFUS_FLUX2` | `0`/`1` | `1` | Back-compat alias: an existing `config/flux2_api.json` is still honored as a stills template. Prefer `config/stills_api.json` for new setups. |
| `RUFUS_CHARACTER_MODE` | `0`/`1` | `1` | Global kill switch for the recurring-character feature (`character_engine.py`) — `0` forces every niche back to the ordinary, no-character pipeline even if a niche has one configured and enabled. See "Recurring character" below. |
| `RUFUS_CHARACTER_TEMPLATE` | `0`/`1` | `1` | Whether to use an exported `config/character_stills_api.json` for image-level character consistency. `0` keeps text-level consistency (the character clause in prompts) but never attempts the IPAdapter/PuLID render path. |
| `RUFUS_BEAT_MOTION` | `i2v`/`i2i`/`cut`/`hero`/`kenburns` | — | **How a beat moves — one selector instead of four interacting flags.** `i2v`: motion model per still (Wan/Hunyuan/LTX/SVD) — best-looking, but measured **600-1800s per clip** on a 3090, i.e. hours a video. `i2i`: each frame is img2img'd from the previous one at low denoise so the frames genuinely continue each other, then **motion-interpolated to 30fps** — real smooth motion for ~1-2s a frame plus ~40-60s of CPU interpolation per beat. **`config/stills_i2i_api.json` is already exported and committed** (z_image_turbo at denoise 0.4 / 14 steps), so this mode works with no setup; if you re-export it, keep **denoise ~0.4 and 10-12 steps** — Z-Image-Turbo's 8-step default leaves only ~3 effective steps at that denoise, too few to move the picture); falls back to plain stills if it's missing. `cut`: several independent stills on one seed, hard cut, no extra setup. `hero`: **exactly ONE beat gets a real motion clip; every other beat is a cut still.** The beat chosen is the one carrying the story architect's `THE SCENE` — the single line in a script that is already a motion prompt, because it names a date, a place and a person doing something. Every other beat is evidence (a total, a share, a consequence), and a video model handed an abstraction produces a slow drift over generic scenery. One clip instead of nine is minutes instead of hours, and one moving shot among stills reads as a deliberate accent rather than wallpaper the viewer stops noticing by beat three. Defaults `RUFUS_HUNYUAN_FRAMES=61` (~2.5s generated, then freeze-extended to fill the beat) — with the exported template's 12 steps that is what puts one clip near ~5 min instead of the ~21 min measured at 30 steps/121 frames. No beat matches the scene → nothing is animated and the run is the stills run. `kenburns`: one still, zoom only. Unset keeps the historical behaviour exactly. |
| `RUFUS_SHOT_CHAIN` | `0`/`1` | `1` | **Cross-beat visual continuity — the picture carries forward between shots, not just within one.** When `storyboard.py` says a beat continues the last one ("the same coin from shot 1, now thinner"), that beat is generated *from the previous beat's image* by an image-**edit** model instead of from fresh noise — a model that has never seen the coin cannot draw "the same coin". Needs a one-time `config/shot_chain_api.json` export from an edit workflow (**Qwen-Image-Edit-2509**: Apache-2.0, so safe for a monetised channel, and its Q4_K_M GGUF is ~13GB). Inert and silent without that file. Refuses a template sampling at **denoise < 0.9** from the loaded image — that is img2img, which can only redraw its input, and is precisely what made all ten beats come back as the character reference portrait once. |
| `RUFUS_SHOT_CHAIN_TEMPLATE` | path | `config/shot_chain_api.json` | Override the edit-workflow export used for shot chaining. |
| `RUFUS_SMOOTH_SCALE` | px | `540` | Width at which `i2i` runs its motion interpolation before upscaling back to 1080. Measured on a 4.8s beat: **117.9s at full 1080×1920 vs 27.6s at 540** — lower is faster, and flat 2D illustration upscales cleanly. |
| `RUFUS_FRAMES_PER_BEAT` | `1`-`4` | `1` | **Animate by cutting between stills instead of by a motion model.** `3` renders three stills per beat — the same scene a moment earlier, at the peak, and a moment later, all on the *same seed* so the composition holds — Ken Burns's each for a third of the beat and hard-cuts them together. A motion model costs ~10 min/video; Z-Image-Turbo renders a still in seconds, so this buys an animated feel far cheaper and sidesteps every current motion-engine glitch. **Mutually exclusive with the motion chain** — Wan/Hunyuan/LTX/SVD are bypassed when this is >1, and say so in the log. |

Each niche picks its own source via `"video_source"` in `config/niches.json`
(default: all niches → `sd`). `RUFUS_VIDEO_SOURCE` overrides it for one run.

### Footage sources
- **`comfy`** — local **ComfyUI**, the top-quality engine (needs ~24GB VRAM, e.g. **RTX 3090**). One photoreal image per beat at 832×1472 → Lanczos upscale → crop 1080×1920, with perceptual-hash dedup so no scene repeats. Start ComfyUI with `--listen`; the stills model itself is whatever's exported to `config/stills_api.json` (see "Swappable stills model" below — required, there is no built-in fallback model). Tune the server address via `COMFY_HOST`. Falls back: **comfy → sd → diffusers → pexels**.
  - **Image-to-video (Wan 2.2 — primary)**: when the six ComfyUI-template files are installed (`wan2.2_i2v_{high,low}_noise_14B_fp8_scaled` in `models/diffusion_models/`, the two `lightx2v_4steps` LoRAs in `models/loras/`, `umt5_xxl_fp8` in `models/text_encoders/`, `wan_2.1_vae` in `models/vae/` — ~35GB one-time via the ComfyUI "Wan 2.2 14B Image to Video" template), every still is animated by **Wan 2.2 14B** (`wan_client.py`): dramatically better temporal consistency than SVD (no warping/melting of fine detail on rigid objects), and each beat gets a **text motion prompt** derived from its image prompt — steered toward camera/ambient motion only, never a one-way completing action like a page turn or gesture. Wan's clips play **one-way, then freeze-extend** to fill the requested duration (`svd_client._assemble(..., ping_pong=False)`) rather than looping forward-then-reversed like SVD's — a live glitch report showed a directional action visibly undoing itself every reversed cycle; a single forward pass with the last frame held has no loop point to go wrong at all. Default mode uses real classifier-free guidance with no speed-up LoRA (`cfg 3.5`, `12` steps split evenly across the high/low-noise experts, the model's own tuned negative prompt) — verified against an actual API export of a proven-clean test render (that export's default toggle was "no LoRA"), trimmed from its 20-step default for a ~1.5-2h/video motion-generation budget instead of 3h+. A faster opt-in path (`RUFUS_WAN_LORA=1`, the lightx2v distill) trades some verified quality for a big speed win — roughly `24 min → ~5-8 min/clip` on a 3090. The distill runs at `cfg 1.0` and now defaults to **8 steps (4+4), not 4**: 4-step lightx2v produces documented slow/reduced motion (worst on the high-noise expert), and 8 steps is the community fix, at a fraction of the quality-path's time. The high-noise LoRA is also applied at **reduced strength (0.8)** to restore motion range while the low-noise LoRA stays at full strength for detail. Tune with `RUFUS_WAN_LORA_STEPS` (8) and `RUFUS_WAN_LORA_HIGH_STRENGTH` (0.8). Faces still blur/glitch under Wan motion even though rigid objects don't (a live report caught this) — the same face-skip heuristics as SVD apply here too, opt back in with `RUFUS_WAN_FACE_MOTION=1`. Knobs: `RUFUS_WAN=0` disables, `RUFUS_WAN_FRAMES` (81), `RUFUS_WAN_STEPS` (12), `RUFUS_WAN_CFG` (3.5), `RUFUS_WAN_LORA` (0), `RUFUS_WAN_FACE_MOTION` (0), `RUFUS_WAN_TIMEOUT` (1800s). Engine chain per clip: **wan → hunyuan → svd → Ken Burns** — any failure walks down, a clip is never lost.
  - **Image-to-video (HunyuanVideo 1.5 — the face engine)**: Wan skips face shots (they blur under its motion), which used to leave them as static Ken Burns. HunyuanVideo 1.5 (Tencent, 8.3B — reported best facial detail of any local video model) slots in right after Wan in the chain: Wan declines a face shot → Hunyuan animates it. **Template-driven, not blind-wired** (`comfy_template.py`): update ComfyUI (Nov 2025+ build), open the built-in "Hunyuan Video 1.5 Image to Video" template (ComfyUI auto-downloads the ~15GB of models — pick fp8/480p variants), run it once on a test image, set the positive prompt text to exactly `RUFUS_PROMPT`, then Workflow → Export (API) → save as `config/hunyuan_i2v_api.json`. Rufus replays exactly that verified graph, substituting only prompt/image/seed/dims per clip. Engine chain per clip: **wan → hunyuan → svd → Ken Burns**.
  - **Recurring character (opt-in, `character_engine.py`)**: per the channel-owner's Calliope-Labs-style direction — one character carried across every beat of a video, and across videos/topics too, not regenerated per-video. Two independent layers:
    1. **Text-level (live today, zero ComfyUI setup, every niche — not just money_history)**: `config/niches.json` → `niches.<name>.character` (`name`, `short_description`, `description`, `enabled`) makes both prompt builders (FLUX's natural-language branch AND the SD/Realistic-Vision token branch) tell GPT to describe the SAME person — same face/hair/build/wardrobe, identical wording — in every beat that shows one, varying only pose/action/framing. money_history ships **"the Chronicler"**, `"enabled": true`: a deliberately *timeless* hooded storyteller-guide (per the channel-owner's Calliope-Labs direction) rather than period dress — chosen specifically so it can stay on across every era (ancient Lydia, Weimar Germany, today) without fighting the PERIOD ACCURACY rule the rest of every scene still follows. The five SD niches (finance/motivation/mindset/business/personal_development) each ship their own distinct starter mascot too ("the Strategist", "the Grinder", "the Observer", "the Builder", "the Climber") — all `"enabled": false` until reviewed; flip one on (and rewrite `description` to taste) to try it. **Two description fields, on purpose**: `short_description` (~15 words) is what goes into *every beat prompt*, and `description` (long) is used only once, to bootstrap the reference portrait. They were the same field originally, and that broke the feature — a ~100-word description demanded in every prompt cannot coexist with the builder's "2 to 4 sentences per prompt" budget, so the model silently dropped the character from all 10 prompts of a live run. Keep `short_description` compact, and keep words like "ledger"/"document"/"sign" out of it (they trip the no-readable-text rule and push the character's own prop out of focus). Note the SD niches only get this text-level layer — the reference-portrait bootstrap and image-conditioning render below are comfy-only (`comfy_client.py`), so an SD niche's character never gets a thumbnail badge or true image-level consistency unless that niche also moves to `video_source: comfy`.
    2. **Image-level (opt-in, needs a one-time ComfyUI export)**: same "proven template" pattern as everything else in this section — build a workflow with an image-conditioning node chain (IPAdapter Plus, IPAdapter FaceID, or PuLID — whichever ComfyUI Manager has installed) feeding a `LoadImage` reference portrait into the sampler, positive prompt set to `RUFUS_PROMPT`, verify the face/outfit actually holds across a few prompts/seeds, then Export (API) → `config/character_stills_api.json`. The FIRST time character mode runs for a niche, Rufus auto-bootstraps the reference portrait itself (one plain-template render from a character-sheet prompt built out of `description`), saves it to `character.reference_image` (default `config/character_reference_<niche>.png`), and reuses that same file for every beat and every future video from then on — that persistence, not per-video regeneration, is what gives the cross-topic continuity. Until the template is exported, this layer is a no-op and Rufus silently uses the ordinary stills pipeline (layer 1's text-level clause still applies if enabled).
  - **Swappable stills model (required — no built-in fallback)**: the image model is a one-time ComfyUI export to `config/stills_api.json` (positive prompt = `RUFUS_PROMPT`, portrait ~832×1472); Rufus deliberately does NOT fall back to a hardcoded model on failure — a "safety net" that silently renders a non-commercial model into a monetized video isn't safe at all, so a failed render just retries/reuses the previous still instead. Recommended for this rig (RTX 3090, 16GB RAM), in order: **Z-Image-Turbo** (Alibaba Tongyi, Apache 2.0, 6B, ~8GB fp8 — fits fully in VRAM with no offload, portrait specialist, ~10× faster at 8 steps; the best fit here because Rufus bans in-image text anyway, so the main edge of a bigger model is moot, and its portrait strength matches the content) or **Qwen-Image-2512** (Apache 2.0, 20B, fp8 ~20GB — better prompt adherence on complex multi-element scenes and best in-image text, at ~10× the generation time; the plan-B when composition on busy scenes matters more than portrait quality/speed). Both run cleanly because stills are a *separate phase* before any video model loads (see two-phase note below). ⚠ Avoid FLUX.1/FLUX.2 entirely: both are non-commercial-licensed, and FLUX.2-dev's 24B text encoder additionally wants 32-64GB system RAM (16GB is the blocker). **Two-phase generation** (built in): all stills render first with the image model resident, then one ComfyUI `/free` unloads it, then the video model loads into a clean card — so a 20GB image model and a 28GB Wan never fight for 24GB, and ComfyUI swaps models once per video instead of ten times.
  - **Image-to-video (SVD — fallback)**: every generated still is animated into a **real motion clip** — 25 frames at 576×1024, interpolated to 30fps, ping-pong-looped seamlessly, upscaled to 1080×1920 (`svd_client.py`). Two engines, resolved automatically once per run: **ComfyUI** (drop `svd_xt.safetensors` in `models/checkpoints/` — one-time ~9GB download from [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)) or **in-process diffusers** (`pip install diffusers transformers accelerate`, needs CUDA; downloads the same weights to the HF cache — no manual file placement). Applies to the `comfy` AND `diffusers` footage sources. Adds ~30-60s GPU time per beat on a 3090. Any per-image failure falls back to the classic **Ken Burns** zoom so a clip is never lost. Knobs: `RUFUS_IMG2VID=0` disables, `RUFUS_IMG2VID_ENGINE` forces an engine, `COMFY_SVD_MODEL` / `RUFUS_SVD_DIFFUSERS_MODEL`, `RUFUS_SVD_MOTION` (1-255, more = more motion), `RUFUS_SVD_FRAMES`/`RUFUS_SVD_FPS`/`RUFUS_SVD_STEPS`. Note: SVD conditions on the image only — it takes **no text prompt** (the model card's generic `DiffusionPipeline` + prompt snippet is misleading; the correct class is `StableVideoDiffusionPipeline`). SVD also isn't face-aware — animating a shot with a visible face can warp/melt facial features across frames — so beats whose own generation prompt reads as a person shot (contains "portrait" or "face") skip SVD entirely and use Ken Burns instead, which can't distort anything.
- **`diffusers`** — in-process HuggingFace Diffusers (SDXL-Turbo by default) — no A1111 server needed. Lighter than the ComfyUI stills path; good when ComfyUI/A1111 aren't running. `RUFUS_DIFFUSERS_MODEL` selects the model.
- **`sd`** (default) — local Stable Diffusion (Automatic1111). Splits the script into **spoken beats** and generates **one content-matched image per beat, in order** — so when the narrator talks about stocks, the screen shows stocks (the renderer cuts on sentence boundaries, keeping image and voice in sync). Each image is upscaled 2× with Real-ESRGAN, cropped to 1080×1920, and animated with Ken Burns. Every image is **perceptual-hash de-duplicated** (aHash + regenerate) so none visibly repeats within a video. Ultra-detailed prompts tuned for Realistic Vision v5.1 with a rotating camera anchor (macro → wide → medium → aerial). **Free forever, runs on a GTX 1060 6GB.** Start A1111 with `./webui.sh --api --xformers --medvram`, then set `SD_HOST` if not on localhost. `SD_CLIPS` caps the scene count (default 6).
- **`pexels`** — free stock footage, 7 candidates, GPT-4o Vision picks the best match. Needs a Pexels key. Automatic fallback when A1111 isn't running.

Fallback chain so a run never dies on footage: **comfy → sd → diffusers → pexels**.

**Stills-only mode** (`RUFUS_STILLS_ONLY=1`): a single switch that overrides Wan, HunyuanVideo, AND SVD at once — every beat is just its stills-model image animated with a subtle Ken Burns zoom (`RUFUS_KENBURNS_ZOOM`, default 0.06). Much faster per video (no motion-model GPU time at all) and sidesteps every current motion-engine glitch, at the cost of no real camera/subject motion — a deliberate trade when the stills alone are already the strongest part of the output. (The old per-engine knobs — `RUFUS_WAN=0`, `RUFUS_HUNYUAN=0`, `RUFUS_IMG2VID=0` — still work individually if you want motion from only one engine rather than none at all.)

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

## Free local AI critique of a finished video

`scripts/video_critique.py` — standalone, **not** wired into the pipeline,
run it by hand whenever you want a second opinion on a rendered video.
Samples frames evenly across the video, sends them + the script to a
**local** vision model via [Ollama](https://ollama.com/download) (free,
zero per-video API cost, runs on the same RTX 3090 — same philosophy as
local Whisper/Realistic Vision/Z-Image elsewhere in this project), and
prints a structured report (hook strength, pacing, whether the images
actually match the narration, visible AI artifacts, caption legibility).

```powershell
# One-time setup
# 1. install Ollama, then:
ollama pull llama3.2-vision      # or: ollama pull llava (lighter/faster)

# Every time you want a critique
python scripts\video_critique.py media_library\output\some.mp4 "the script text"
```

Saves the report next to the video as `<video>.critique.txt`. `OLLAMA_HOST`
(default `http://localhost:11434`) and `OLLAMA_VISION_MODEL` (default
`llama3.2-vision`) are overridable. Purely advisory for now — it does not
touch the score, the approval gate, or anything automatic; if that changes
later it'll be a deliberate, separate decision.

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

- **After research** — rejects a seed with no concrete facts to build a story on, **or** one that passes a "knowledge gap" test: does it contain a counter-intuitive fact that would break a typical viewer's mental model? A seed can be accurate and on-topic and still get rejected here for being a flat, expected restatement of common knowledge with no surprise in it (retries `get_seed`).
- **After scripting** — a **fact-check**: compares the finished script's names/numbers/dates against the source seed (the script prompt forbids inventing them; this verifies GPT complied). On rejection the script is rewritten once with the objection fed back; if the rewrite is *still* flagged, the video renders but the **upload is held** with the specific claim printed — wrong facts never publish themselves. Wisdom-quote seeds keep their documented-history allowance.
- **After beat-prompt writing, before stills/SD/diffusers generation** — rejects near-duplicate or off-topic image prompts (retries `_build_sd_prompts`, which is non-deterministic so a retry actually differs).

Each call is a few hundred tokens (a fraction of a cent) and fails **open** —
no key, an API error, or a malformed reply always approves, so a broken judge
can never block a render. Set `RUFUS_SUPERVISOR=0` to skip it entirely.

---

## Script depth (beyond grounding)

Grounding (never inventing a fact) and interest (making a viewer actually
care) are different problems — a script can pass every accuracy check and
still be flat. Four checks target the second problem specifically:

- **Sensory-anchor disqualifier** — the body critic (`_score`) rejects
  (caps score ≤4) a script with zero concrete physical detail — nothing a
  viewer could see, hear, feel, smell, or taste. Abstract summary, however
  accurate, is a critic failure now, not just a missed opportunity.
- **Cadence pattern-interrupt** (`_cadence_violation`) — a script whose
  sentences are all a similar length reads as monotone even with perfect
  content; the pre-filter chain now requires at least one short, punchy
  sentence (≤6 words) and one longer, flowing one (≥15 words) somewhere in
  the body before it's accepted.
- **Hook-opener diversity** (`_overused_hook_openers`) — a long-term
  semantic-decay guard: if a small set of opening words dominates the last
  30 shipped hooks in a niche (≥15% share), those words are named and
  banned in the next hook-factory prompt. Catches slow convergence toward
  the model's favorite few hook shapes across dozens/hundreds of videos —
  something nobody notices without actually reading the DB.
- **Bottleneck breakdown** — the dashboard's `/failures` page groups every
  rejected attempt, all-time, into a fixed taxonomy (safety / accuracy /
  weak_hook / loose_structure / boring / weak_seed / footage_drift)
  instead of counting distinct free-text strings — after enough volume,
  this answers "which stage of the pipeline is actually the bottleneck"
  at a glance. This now covers ALL five gates, not just the two inside
  `script_writer.py` — `supervisor.py`'s three judge calls (seed
  knowledge-gap, fact-check, footage-prompt drift) previously verdicted
  silently to the console and left no queryable trace at all.
- **Story Architect, strengthened** — the plan now includes a STAKES GAP
  ("what does the viewer specifically lose by not knowing this") alongside
  the spine fact and the turn, and the turn is explicitly required to be a
  *direct consequence* of the spine fact, not a separate idea grafted on.
- **Sensory anchor, timed** — the disqualifier now requires the sensory
  detail specifically in the first third of the body, not just present
  anywhere — a detail buried near the end doesn't stop the swipe.
- **Topic clustering** (`check_topic_similarity` / `add_topic_embedding`)
  — a second dedup layer alongside the full-script similarity gate. Two
  scripts can be semantically distinct (different examples, different
  framing) and still cover the same underlying topic within a couple of
  weeks — this embeds just the pre-analysis CORE line and checks it
  against a **time-windowed** (14-day, not count-windowed) history per
  channel, so the same topic is fair game again once it's not recent.

---

## Operations

```bash
python scripts/health_check.py        # pre-flight: deps, keys, config, disk, ComfyUI+stills template
python scripts/review_scripts.py      # browse generated scripts from the DB
python scripts/analyze_scripts.py     # script-writer funnel/cost/score analysis
python scripts/analytics_fetcher.py   # pull YouTube metrics into the DB (cron daily) + Discord digest
python scripts/feedback_analyzer.py   # turn metrics into config/learnings.json
python scripts/image_gen.py "..."     # generate a thumbnail image on the GPU
python scripts/auth.py list           # who can reach the dashboard
python scripts/watchdog.py            # keep the dashboard answering (serve.ps1 runs this at boot)
python -m pytest tests/ -q            # test suite
```

### Go live (daily automation, Windows)

One command turns the machine autonomous, one or several Shorts per day, hands-off:

```powershell
.\schedule_daily.ps1                              # one run/day at 13:00
.\schedule_daily.ps1 -Time "09:30"                # one run/day, your hour
.\schedule_daily.ps1 -Times "09:00,13:00,17:00,20:00,23:00"   # 5 runs/day
.\schedule_daily.ps1 -Unregister                  # stop ALL Rufus daily runs
```

Each trigger is registered as its **own** Windows Task Scheduler task (`Rufus
Daily Short 1`, `2`, ...) rather than an in-process loop — one crashing or
running long can't take the rest of the day down with it, since Task
Scheduler fires each one independently. Re-running `-Times` with a different
count clears any stale extra tasks from a previous run automatically.

At higher daily counts, one channel burns through its Wikipedia topic pool
proportionally faster (money_history's ~155 topics ≈ 1 month at 5/day
instead of 5 months at 1/day) — **the pool now auto-replenishes**: whenever
a niche's unused-topic count drops under 30, GPT proposes ~40 more real
article titles, each one individually *validated* by actually fetching its
Wikipedia summary before being trusted (GPT invents plausible-sounding
titles that don't exist — those are silently dropped, never added). Costs
a few cents, happens automatically inline in a run, fails open (no key /
GPT error / everything invented → the pool just doesn't grow that run,
never blocks the video being made from whatever topics remain).

Requirements at run time: PC on, ComfyUI running (stills/motion engine). The scheduled
run uses `run_scheduled.bat` — full render pipeline through scoring/QC, then
queues for review (see Dashboard below) rather than uploading itself.

**Daily 2-minute checklist:** open the dashboard → approve or reject whatever
landed in "Awaiting your review" (edit title/description first if you want)
→ approving uploads **private**, scheduled to the next peak hour. If a video
scored low or failed a gate, that reason is right there on its page instead
of buried in `logs\rufus_YYYYMMDD.log`.

### Dashboard — approval queue

**Nothing uploads automatically.** Every rendered video lands as `pending`;
a human has to click **Approve** in the dashboard for it to actually
publish to YouTube. This is the intended workflow for handing upload/review
off to someone else (e.g. a channel partner) without giving them shell
access to the machine.

```powershell
python scripts\dashboard.py
```

Open `http://localhost:8765`. The homepage leads with **"Awaiting your
review"** — the actual to-do list. Per video: score, the critic's full
reasoning, the auto-gate's opinion (shown as a note, not a decision anymore),
an editable **title/description** (persisted before any upload happens, so
edits are never lost or overwritten), **Approve & Upload** / **Reject**
buttons, a full recent-videos table, a score trend line, the most common
script-rejection reasons, and — per video — direct links to that run's
actual keyframes/voiceover mp3 (when `RUFUS_DEBUG=1` was on) from
`media_library\debug\<run_id>\`.

Approving reconstructs the video's own original channel/niche context (not
whatever's active today) so the right voice/CTA-pool/category still apply
even if you approve days after it rendered.

**Escape hatch**: `RUFUS_AUTO_UPLOAD=1` restores the old fully-automatic
behavior (uploads immediately if score/QC/facts all pass, exactly as
before this feature existed) for anyone who decides they don't want the
manual step after all.

Self-contained: no external CSS/JS. Reads/writes `rufus.db` (safe alongside
a live run — WAL mode), never crashes on a missing or partial DB row. The
approve action is the only place this app talks to the network (the real
YouTube upload call) — everything else is local.

**Make a video about a specific topic** (backlog item #6): a box right on
the homepage, or `python scripts\main.py --topic "Bretton Woods"` from the
command line. Your input is resolved to a real Wikipedia article (exact
title first, then a search fallback for an imprecise phrase like "bretton
woods conference thing") instead of being handed to the script writer as
free text — a raw string with no real source would just get its claims
rejected by the fact-gate anyway, so this keeps the same grounding
guarantee as an auto-picked topic. From the dashboard it runs in the
background (`subprocess.Popen`, non-blocking — a render can take a while)
and lands in the normal pending-review queue like anything else; it never
auto-uploads either.

**Failures & rejected attempts** (`/failures`, linked top-right): every
mistake the automation made, not just its successes. Two sections: **crashed
runs** — `RUFUS_DEBUG` folders with no matching database row at all, i.e. a
run that started but died before Step 6 (bad script, failed render) and
would otherwise leave zero trace anywhere in the app; and **rejected script
attempts** — the full, filterable (by niche/phase) list of every hook/body
`script_writer.py` tried and threw away, with its exact rejection reason
(already logged to `script_attempts` for every run; the homepage only ever
showed the top-8 aggregate, this is the full browser).

**Loopback-only by default**: it binds to `127.0.0.1`, so it's reachable
*only from this PC* — not your home WiFi, not any other device. This is
deliberate: `/system` has routes that can start and kill processes on this
machine, so "no login" alone isn't enough once those exist.

Never set `RUFUS_DASHBOARD_HOST=0.0.0.0` and never port-forward this —
either one puts the process-control routes on your open WiFi or the
internet. Publish it with `tailscale serve` instead (below), which keeps
the dashboard on loopback and still reaches your phone.

Knobs: `RUFUS_DASHBOARD_PORT` (8765), `RUFUS_DASHBOARD_HOST` (`127.0.0.1`).

---

## Remote access & sharing with a second person

### Why loopback is not a login

`tailscale serve` terminates the connection and proxies to `127.0.0.1`, so
**every tailnet visitor arrives at the dashboard as loopback**. Any check of
the form "is this request from 127.0.0.1" therefore passes for anyone you
share the tailnet URL with — including the routes that start and kill
processes and the one that publishes to YouTube. Sharing the URL is not the
same as sharing read access.

So access is decided by a **token the caller holds**, not by where the
request appears to come from.

### Roles

| Role | Can | Cannot |
|---|---|---|
| `owner` | everything | — |
| `partner` | make videos, generate thumbnails, edit title/description, download media | approve/upload, settings, start or kill processes |
| `viewer` | browse and download | generate anything |

`config/users.json` is **gitignored** — the tokens in it are real
credentials. With no such file the dashboard stays in legacy mode (loopback
= owner), so an existing single-user setup is unchanged until you opt in.

```powershell
python scripts\auth.py init                      # create the file + owner link
python scripts\auth.py add james --role partner  # prints james's sign-in link
python scripts\auth.py list
python scripts\auth.py link james                # reprint james's link (same token)
python scripts\auth.py revoke james               # kills that link immediately
```

Each command prints a `.../?token=...` URL. **The link is the password** —
send it privately. Opening it once on a phone stores an HttpOnly,
SameSite=Strict cookie, so the token stops trailing in URLs afterward.

**Run `serve.ps1 -Tailscale` (below) before adding anyone.** The link's
domain comes from `config/dashboard_url.txt`, which only exists once
Tailscale has actually published the dashboard — `auth.py add` run before
that prints a `localhost` link that does nothing on a phone. If you already
added someone before running `-Tailscale`, don't `revoke` — their token is
still valid, just re-print it with the right domain:
`python scripts\auth.py link james`.

### Managing users from the dashboard itself

Everything above is also a page, at **Settings → Manage users** (owner-only,
`/settings/users`) — add or revoke a partner/viewer without a terminal. It
calls the exact same `add_user()`/`revoke_user()` functions the CLI does, so
the two can't drift into enforcing different rules. Adding someone there
shows their sign-in link right on the page; revoking takes effect on their
very next request.

### Google Sign-In (optional, instead of a token link)

A token link works everywhere with zero setup, and stays the default. Google
Sign-In is for when you'd rather your partner log in with their own Google
account than hold onto a link — nothing to lose in a screenshot, nothing to
forward by mistake.

**How it decides who gets in**: Google only vouches for *identity* (which
verified email just signed in) — it grants nothing by itself. Access is still
whatever's in `config/users.json`: an email with no matching `google_email`
entry is refused exactly like a wrong token, no account is silently created.

**One-time setup**, in the **same** Google Cloud project you may already have
for YouTube (see below) — this is a *second*, differently-typed OAuth client;
Google doesn't let a Desktop-app client (YouTube's) and a Web-app client
(this) share one registration:

1. Run `serve.ps1 -Tailscale` first — you need the real tailnet URL for step 3.
2. [Google Cloud Console](https://console.cloud.google.com) → your project
   → **Credentials → Create credentials → OAuth client ID → Web application**.
3. **Authorized redirect URIs** → add exactly:
   `https://<your-machine>.<your-tailnet>.ts.net/auth/google/callback`
   (the domain `tailscale serve status` printed, from `config/dashboard_url.txt`).
   Must match byte-for-byte or Google refuses with `redirect_uri_mismatch`.
4. Download the client ID + secret, save as `config/google_oauth.json`
   (gitignored — see `config/google_oauth.json.template` for the shape).
5. Add a user with their Google email attached:
   ```powershell
   python scripts\auth.py add james --role partner --google james@gmail.com
   ```
   (or the same thing from **Settings → Manage users** in the dashboard —
   the "Google email" field is optional on that form).

`/login` now shows a **Sign in with Google** button automatically once
`config/google_oauth.json` exists. James signs in with his own account —
no link to send at all.

### Always-on server (Windows)

```powershell
.\serve.ps1 -Tailscale     # dashboard + watchdog start at boot, published to your tailnet
.\serve.ps1 -Status        # what's registered, what's answering, who has access
.\serve.ps1 -Unregister    # remove the boot tasks
```

Two Task Scheduler entries run at startup (before anyone logs in): the
dashboard, and `scripts/watchdog.py`, which polls `/healthz` and restarts the
dashboard if it stops answering — a crashed Flask process otherwise leaves
the tailnet URL dead until someone happens to try it. Set
`RUFUS_WATCHDOG_COMFY=1` plus `COMFY_START_CMD` to have it revive ComfyUI too.

For the box to answer at 3am it must not sleep:
`powercfg /change standby-timeout-ac 0`.

### What a partner can actually do

- **`/generate`** — describe a topic, start a real run on your RTX 3090. It
  goes through every existing gate and lands in *your* review queue. It is
  not published.
- **`/thumbnails`** — describe an image, get it rendered on the GPU through
  the same ComfyUI stills workflow the videos use, then **⬇ Save to phone**.
  1280×720 (YouTube's thumbnail shape) or portrait.
- **⬇ Download mp4** on any video page — the finished render, straight to
  their phone.

Same thing from the command line:

```powershell
python scripts\image_gen.py "a cracked hourglass spilling gold coins"
python scripts\image_gen.py "..." --portrait --seed 42
```

(Distinct from `thumbnail_gen.py`, which brands a frame of an *already
rendered* video.)

### Discord

```powershell
$env:RUFUS_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

Server Settings → Integrations → Webhooks → New Webhook. Unlike the phone-push
backends, Discord carries the **artifact**: a published video and every
generated thumbnail get posted into the channel, not just a link. Files over
8MB are linked instead of uploaded (Discord rejects oversized uploads only
after the whole body has been sent). `RUFUS_DISCORD_UPLOAD=0` for links only.

The daily analytics run posts a digest there too — total views, average watch
percentage, and the top five videos.

---

## YouTube API (upload + analytics)

Both the uploader and the analytics fetcher are already written — this is
purely a one-time credential setup. They share **one** token file and one
scope list (`youtube_uploader.SCOPES`), so doing this once enables both.

1. [Google Cloud Console](https://console.cloud.google.com) → create a project.
2. **APIs & Services → Library** → enable **YouTube Data API v3** *and*
   **YouTube Analytics API**.
3. **OAuth consent screen** → External → add your own Google account under
   **Test users**. (Leave it in "Testing"; publishing it triggers Google
   verification you don't need for a personal channel. Test-mode refresh
   tokens expire after 7 days — if uploads start failing weekly with an
   auth error, that's this, and moving the app to "In production" fixes it.)
4. **Credentials → Create credentials → OAuth client ID → Desktop app**.
   Download the JSON as `config/client_secrets.json`.
5. Authorize once, interactively, on the host PC:
   ```powershell
   python scripts\youtube_uploader.py media_library\output\some.mp4 "test"
   ```
   A browser opens → approve → `config/youtube_token.json` is written and
   every later run is non-interactive.

Then verify:

```powershell
python scripts\health_check.py
python scripts\analytics_fetcher.py     # should print per-video views/watch%
```

**Quota**: the default 10,000 units/day allows ~6 uploads/day (1,600 units
each); `videos.list` and analytics reads are 1 unit. A few runs a day is
comfortably inside it.

**Do the OAuth step at the keyboard, not over Tailscale** — it opens a local
browser window and needs a real display. Adding a scope later invalidates the
existing token: delete `config/youtube_token.json` and repeat step 5.

Analytics runs daily inside `run_scheduled.bat` (before the render, so the day's
script can learn from yesterday's numbers) and posts its digest to Discord.

**Two comments auto-post right after every upload** (`youtube_uploader.py`,
needs the `youtube.force-ssl` scope already in `SCOPES` above): a CTA line
from the niche's `cta_pool`, and — when the video's seed carried a real link
(Wikipedia article, Stack Exchange question; not the wisdom-pool fallback,
which has none) — a second comment citing that exact source. Both are a real
trust/differentiation lever against generic "AI slop" channels, but neither
can be **pinned** via the public YouTube Data API (no such endpoint exists,
confirmed) — pinning the source comment stays a manual 10-second step if you
want it pinned, same as the existing CTA-comment note.

---

## Security invariants

- `config/keys.json` and `config/users.json` are **gitignored** — never commit
  real keys or dashboard access tokens.
- Loopback is **not** treated as proof of identity once `config/users.json`
  exists: `tailscale serve` proxies every tailnet visitor through `127.0.0.1`,
  so authorization is by token and role, never by source address.
- Only the `owner` role can publish. A `partner` can generate videos and
  images and download them, but no role below owner can put anything on the
  channel.
- YouTube uploads default to **private**.
- The quality gate (`RUFUS_MIN_UPLOAD_SCORE`, default 8) holds weak scripts back for review.
- Every upload sets `status.containsSyntheticMedia=True` (`youtube_uploader.py`) — YouTube's altered/synthetic-content disclosure policy requires self-declaring this for realistic AI-generated video, and every Rufus video qualifies (GPT script, AI-generated imagery and motion, synthesized voice). Not optional/configurable — it's always true for this pipeline's output.
