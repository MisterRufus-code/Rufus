"""
Multi-channel manager.

Supports running Rufus against multiple YouTube channels, each with its own:
  - OAuth token file
  - Niche / model config
  - Output directory
  - Upload schedule

Channel configs live in config/channels/<channel_id>.yaml.
Channel tokens live in tokens/<channel_id>_token.json  (git-ignored).

Usage:
    from src.channels.manager import get_channel, list_channels, run_channel_pipeline
    channels = list_channels()
    result = run_channel_pipeline("finance_main", topic="investing")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CHANNELS_DIR = Path(__file__).parent.parent.parent / "config" / "channels"
_TOKENS_DIR = Path(__file__).parent.parent.parent / "tokens"


@dataclass
class ChannelConfig:
    channel_id: str
    display_name: str = ""
    niche: str = "general"
    model: str = "mistral"
    tts_voice: str = "af_heart"
    output_subdir: str = ""
    upload: bool = False
    privacy: str = "private"
    token_file: str = ""          # resolved automatically if empty
    secrets_file: str = ""        # resolved automatically if empty
    auto_schedule: bool = True    # use upload_optimizer.best_upload_time()


def get_channel(channel_id: str) -> ChannelConfig:
    """Load channel config. Falls back to defaults for unknown channels."""
    path = _CHANNELS_DIR / f"{channel_id}.yaml"
    if not path.exists():
        return ChannelConfig(channel_id=channel_id)

    try:
        import yaml  # type: ignore[import]
        data: dict = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return ChannelConfig(channel_id=channel_id)

    token_file = data.get("token_file") or str(
        _TOKENS_DIR / f"{channel_id}_token.json"
    )
    secrets_file = data.get("secrets_file") or str(
        _TOKENS_DIR / f"{channel_id}_client_secrets.json"
    )

    return ChannelConfig(
        channel_id=channel_id,
        display_name=data.get("display_name", channel_id),
        niche=data.get("niche", "general"),
        model=data.get("model", "mistral"),
        tts_voice=data.get("tts_voice", "af_heart"),
        output_subdir=data.get("output_subdir", channel_id),
        upload=bool(data.get("upload", False)),
        privacy=data.get("privacy", "private"),
        token_file=token_file,
        secrets_file=secrets_file,
        auto_schedule=bool(data.get("auto_schedule", True)),
    )


def list_channels() -> list[str]:
    """Return IDs of all configured channels."""
    if not _CHANNELS_DIR.exists():
        return []
    return sorted(p.stem for p in _CHANNELS_DIR.glob("*.yaml"))


def run_channel_pipeline(
    channel_id: str,
    topic: Optional[str] = None,
    geo: str = "US",
    multi_shorts: bool = False,
) -> "PipelineResult":  # type: ignore[name-defined]  # noqa: F821
    """
    Run the full pipeline for a specific channel, applying its config and
    pointing the uploader at its own OAuth token.
    """
    import config as global_config
    from src.pipeline.orchestrator import run_pipeline

    ch = get_channel(channel_id)

    # Override token paths for this channel
    original_token = global_config.YOUTUBE_TOKEN_FILE
    original_secrets = global_config.YOUTUBE_CLIENT_SECRETS_FILE
    try:
        global_config.YOUTUBE_TOKEN_FILE = ch.token_file
        global_config.YOUTUBE_CLIENT_SECRETS_FILE = ch.secrets_file

        output_dir: Optional[Path] = None
        if ch.output_subdir:
            output_dir = Path(global_config.OUTPUT_PATH) / ch.output_subdir

        result = run_pipeline(
            topic=topic or "",
            niche=ch.niche,
            ollama_model=ch.model,
            tts_voice=ch.tts_voice,
            output_dir=output_dir,
            upload=ch.upload,
            privacy=ch.privacy,
            geo=geo,
            multi_shorts=multi_shorts,
        )
    finally:
        global_config.YOUTUBE_TOKEN_FILE = original_token
        global_config.YOUTUBE_CLIENT_SECRETS_FILE = original_secrets

    return result
