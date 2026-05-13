# Rufus — Developer Guide

100% free, fully local YouTube viral automation engine.

## What it does

Rufus runs an end-to-end pipeline on your local machine:

1. Fetches trending topics (Google Trends, Reddit, YouTube)
2. Downloads royalty-free footage (Pexels, YouTube CC via yt-dlp)
3. Indexes footage with CLIP embeddings → Qdrant or FAISS
4. Generates ideas + script from media context (Ollama LLM)
5. Optimises the hook using psychology scoring engine
6. Semantically matches footage to each script scene (CLIP vectors)
7. Checks visual entropy (prevents repetitive edits)
8. Generates voiceover (Kokoro TTS, local, free)
9. Renders video (FFmpeg)
10. Uploads to YouTube (optional, private by default)
11. Syncs analytics → ML feedback loop for future improvements

## Architecture

```
main.py (Click CLI)
  └─ src/pipeline/orchestrator.py    ← central pipeline controller
       ├─ src/trends.py              ← Google Trends / Reddit / YouTube trending
       ├─ src/ideas.py               ← Ollama LLM (ideas, scripts, keywords)
       ├─ src/media_fetch/           ← Pexels + yt-dlp footage downloaders
       ├─ src/ingestion/             ← CLIP/BLIP/YOLO feature extraction + indexing
       ├─ src/database/              ← Qdrant vector store + FAISS fallback
       ├─ src/matching/semantic.py   ← CLIP text→video semantic matching
       ├─ src/entropy/engine.py      ← visual variety scoring
       ├─ src/psychology/hooks.py    ← hook scoring (open loop, loss aversion, etc.)
       ├─ src/tts/kokoro.py          ← local TTS (Kokoro, zero cost)
       ├─ src/pipeline/renderer.py   ← FFmpeg render
       ├─ src/uploader.py            ← YouTube Data API v3
       ├─ src/analytics.py           ← YouTube Analytics API sync
       └─ src/ml/                    ← feedback loop + optimizer
```

### Key design decisions

- **Media-first**: footage is downloaded and indexed *before* the script is written.
  This means the LLM generates a script from what footage actually exists, not the other way around.
- **Process-level task lock** (`src/task_lock.py`): only one heavy op (Ollama, CLIP) runs at a time.
- **Global deduplication**: assets used in the last 5 renders are excluded from matching.
- **ML feedback loop**: every upload syncs performance data back; future pipelines use this to pick better prompts and footage.

## Quick start

### Prerequisites

```bash
# System
sudo apt install ffmpeg
# Docker (for Qdrant)
docker run -d -p 6333:6333 qdrant/qdrant
# Ollama
curl https://ollama.ai/install.sh | sh
ollama pull mistral
```

### Install

```bash
git clone <repo>
cd Rufus
pip install -r requirements.txt
cp .env.example .env   # fill in PEXELS_API_KEY (free at pexels.com/api)
```

### Run

```bash
# Validate the pipeline without rendering anything
make dry-run

# Full pipeline (long form + Shorts)
python main.py pipeline --both --niche finance

# Download footage only
python main.py fetch --source both --keywords "investing stocks" --count 10

# Start API server for n8n
make run-api
```

## Testing

```bash
make test           # full suite
make test-fast      # skip integration tests
```

Tests live in `tests/`. `tests/conftest.py` mocks heavy deps (torch, faiss, google-auth)
so the suite runs on any machine without ML packages installed.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PEXELS_API_KEY` | — | Required for Pexels footage (free at pexels.com/api) |
| `RUFUS_API_KEY` | auto-generated | API key for the n8n HTTP server |
| `RUFUS_MAX_CONCURRENT_JOBS` | `2` | Max parallel pipeline jobs via API |
| `RUFUS_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `RUFUS_LOG_DIR` | `logs/` | Directory for JSON-lines log files |
| `ENTROPY_SCENE_REPEAT` | `0.97` | CLIP cosine similarity threshold for repetition |
| `ENTROPY_MIN_SCORE` | from config | Minimum entropy to pass the check |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |

## Adding a new niche

Create `config/niches/<name>.yaml` — copy `config/niches/finance.yaml` as a template.
Fields: `model`, `script_style`, `tone`, `topic_boosters`, `hook_weights`, `upload_schedule`,
`entropy_min_score`, `shorts_target_duration`.

## Multi-channel setup

Create `config/channels/<channel_id>.yaml`:
```yaml
display_name: My Finance Channel
niche: finance
model: mistral
tts_voice: af_heart
upload: true
privacy: public
```

Put each channel's OAuth token in `tokens/<channel_id>_token.json` (git-ignored).

Run: `python main.py pipeline --topic "investing" --channel my_finance_channel`

## Project layout

```
config/
  niches/       ← per-niche YAML configs
  channels/     ← per-channel YAML configs
src/
  api/          ← FastAPI server (n8n)
  channels/     ← multi-channel manager
  database/     ← vector store (Qdrant + FAISS)
  entropy/      ← visual variety engine
  ingestion/    ← CLIP/BLIP/YOLO extraction + indexer
  matching/     ← semantic scene→clip matching
  media_fetch/  ← Pexels, yt-dlp downloaders
  ml/           ← feedback store, optimizer
  pipeline/     ← orchestrator, renderer, shorts, subtitles
  psychology/   ← hook scoring engine
  thumbnail/    ← thumbnail generator
  tts/          ← Kokoro TTS
  logging_config.py
  niche_config.py
  task_lock.py
  trends.py
  uploader.py
  upload_optimizer.py
tests/
  conftest.py   ← mocks for torch/faiss/google-auth
  test_*.py
logs/           ← runtime JSON-lines logs (git-ignored)
tokens/         ← OAuth tokens (git-ignored)
output/         ← rendered videos (git-ignored)
```
