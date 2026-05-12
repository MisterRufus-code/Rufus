import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
YOUTUBE_CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
YOUTUBE_TOKEN_FILE = "token.json"
DEFAULT_VIDEO_CATEGORY = os.getenv("DEFAULT_VIDEO_CATEGORY", "22")
DEFAULT_VIDEO_PRIVACY = os.getenv("DEFAULT_VIDEO_PRIVACY", "private")

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

CLAUDE_MODEL = "claude-sonnet-4-6"
