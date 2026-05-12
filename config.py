import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# --- YouTube ---
YOUTUBE_CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
YOUTUBE_TOKEN_FILE = "token.json"
DEFAULT_VIDEO_CATEGORY = os.getenv("DEFAULT_VIDEO_CATEGORY", "22")
DEFAULT_VIDEO_PRIVACY = os.getenv("DEFAULT_VIDEO_PRIVACY", "private")
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# --- Vector DB (Qdrant) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rufus_media")
CLIP_EMBEDDING_DIM = 512          # CLIP ViT-B/32 output dimension

# --- Media library ---
MEDIA_LIBRARY_PATH = Path(os.getenv("MEDIA_LIBRARY_PATH", "media_library"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "output"))
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- ML feedback ---
FEEDBACK_DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "feedback.json"))

# --- Entropy thresholds ---
ENTROPY_MIN_SCORE = float(os.getenv("ENTROPY_MIN_SCORE", "0.55"))
