# ComfyUI video generation for Rufus

Rufus can generate its background clips with a local ComfyUI server instead of
pulling stock footage from Pexels. Each clip is generated from the script's own
video queries, so the footage always matches the story.

## Enable it

```bash
export RUFUS_VIDEO_SOURCE=comfy          # use ComfyUI instead of Pexels
export RUFUS_GPU=1                        # Whisper→CUDA, FFmpeg→NVENC
export COMFY_URL=http://127.0.0.1:8188    # default
python scripts/main.py --skip-upload
```

If ComfyUI is unreachable or every clip fails, Rufus automatically falls back to
Pexels — it never crashes the run.

## The workflow file

`wan_t2v.json` is a ComfyUI **API-format** workflow for Wan2.1 text-to-video. Rufus
loads it and substitutes these tokens per clip:

| Token          | Filled with                          |
|----------------|--------------------------------------|
| `__PROMPT__`   | the video query + cinematic styling  |
| `__NEG__`      | negative prompt (`COMFY_NEG`)        |
| `__SEED__`     | random per clip                      |
| `__WIDTH__`    | `COMFY_WIDTH`  (default 720)         |
| `__HEIGHT__`   | `COMFY_HEIGHT` (default 1280)        |
| `__FRAMES__`   | `COMFY_FRAMES` (default 81 ≈ 5s)     |
| `__FILENAME__` | output prefix                        |

### Using your own workflow

If you build a different/better workflow in the ComfyUI canvas:

1. **Save → Save (API Format)** to export JSON.
2. Replace the prompt/seed/size fields with the tokens above.
3. Save it here as `config/comfy_workflows/<name>.json`.
4. `export COMFY_WORKFLOW=<name>`.

The output node must be `VHS_VideoCombine` (VideoHelperSuite) with
`format: video/h264-mp4` so Rufus gets a real mp4.

## Tuning

| Env var        | Default | Notes                                        |
|----------------|---------|----------------------------------------------|
| `COMFY_CLIPS`  | 5       | clips generated per video                    |
| `COMFY_FRAMES` | 81      | more = longer clips, slower                  |
| `COMFY_WIDTH`  | 720     | Rufus upscales/crops to 1080×1920 anyway     |
| `COMFY_HEIGHT` | 1280    |                                              |
| `COMFY_TIMEOUT`| 900     | per-clip ceiling in seconds                  |

For higher quality swap the 1.3B model line in `wan_t2v.json` for the 14B model
(needs ~24GB VRAM — a 3090/4090).
