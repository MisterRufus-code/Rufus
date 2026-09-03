#!/usr/bin/env python3
"""
channel_config.py — "Channel-in-a-box": one config object per YouTube channel.

Adding channel #2 = create the YouTube account → OAuth once → one JSON block in
config/channels.json → one cron line. Zero code edits.

Resolution order for the active channel:
    explicit arg → RUFUS_CHANNEL env var → channels.json "default_channel"

LEGACY SHIM: if config/channels.json does not exist, a "main_en" channel is
synthesized from the original single-channel paths (config/youtube_token.json,
config/learnings.json, media_library/output/) — existing installs and tests
keep working with zero migration.

Composition rule: niches.json is the base; the channel's niche_overrides[niche]
shallow-merge on top; channel-level fields (voice, language, schedule) always win.

Filesystem convention (no paths inside the JSON):
    config/channels/<id>/client_secrets.json   (falls back to config/client_secrets.json)
    config/channels/<id>/youtube_token.json
    config/channels/<id>/tiktok_token.json
    config/channels/<id>/learnings.json
    media_library/output/<id>/
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import paths

ROOT          = Path(__file__).parent.parent
CONFIG_DIR    = ROOT / "config"
CHANNELS_FILE = CONFIG_DIR / "channels.json"
NICHES_FILE   = CONFIG_DIR / "niches.json"

LEGACY_ID = "main_en"

DEFAULT_UPLOAD = {
    "videos_per_day": 2,
    "min_score":      8,
    # PUBLIC, and this was `private` until the owner asked for it to change.
    # The old default was not really "keep it hidden" — an upload marked
    # private also carried a publishAt of the next peak hour, so YouTube
    # published it anyway, just later. What it actually did was make
    # publishing depend on the timezone database resolving, and on Windows
    # that database is a pip package nobody had installed: the schedule
    # silently degraded to no publishAt at all, and finished videos sat
    # private forever.
    #
    # Nothing here uploads on its own. main.py needs RUFUS_AUTO_UPLOAD=1 and a
    # score over the bar; the dashboard needs a human to press approve. So
    # "public" is what happens when somebody has already decided to publish.
    # Set it back per-channel in channels.json, or from the dashboard.
    "privacy":        "public",
    "peak_hours":     [8, 12, 17, 20],
    "timezone":       "America/New_York",
}

# The dashboard writes this; it overrides whatever the channel config says.
# One vocabulary, not two: the values are YouTube's own privacyStatus, and
# scheduling is a property of `private` because publishAt is only valid there.
PRIVACY_VALUES = ("public", "private", "unlisted")


def _privacy_override() -> str:
    # The name is written out rather than held in a constant so env_doctor's
    # scanner can see it: a setting missing from docs/ENVIRONMENT.md is a
    # setting nobody finds.
    raw = os.environ.get("RUFUS_PRIVACY", "").strip().lower()
    if raw and raw not in PRIVACY_VALUES:
        print(f"[channel] RUFUS_PRIVACY={raw!r} is not one of "
              f"{', '.join(PRIVACY_VALUES)} — ignoring it")
        return ""
    return raw


@dataclass
class Channel:
    id: str
    display_name: str = ""
    language: str = "en"
    voice: str = ""                      # "" = tts_engine default
    caption_font: str = ""               # "" = renderer default (Anton)
    niches: list = field(default_factory=list)
    schedule: list = field(default_factory=list)
    upload: dict = field(default_factory=lambda: dict(DEFAULT_UPLOAD))
    platforms: dict = field(default_factory=lambda: {"youtube": {"enabled": True}})
    niche_overrides: dict = field(default_factory=dict)
    legacy: bool = False                 # synthesized from pre-channels.json layout

    # ── Paths (convention over config) ──────────────────────────────────────────

    @property
    def dir(self) -> Path:
        return CONFIG_DIR / "channels" / self.id

    def token_path(self, platform: str = "youtube") -> Path:
        if self.legacy:
            legacy_map = {"youtube": CONFIG_DIR / "youtube_token.json",
                          "tiktok":  CONFIG_DIR / "tiktok_token.json"}
            if platform in legacy_map:
                return legacy_map[platform]
        return self.dir / f"{platform}_token.json"

    def client_secrets_path(self) -> Path:
        """Per-channel secrets if present, else the shared project secrets.
        (One Google Cloud project covers ~3 channels at 2 uploads/day — the
        videos.insert quota is 6/day/project.)"""
        per_channel = self.dir / "client_secrets.json"
        return per_channel if per_channel.exists() else CONFIG_DIR / "client_secrets.json"

    @property
    def learnings_path(self) -> Path:
        if self.legacy:
            return CONFIG_DIR / "learnings.json"
        return self.dir / "learnings.json"

    @property
    def output_dir(self) -> Path:
        base = paths.output_dir()
        return base if self.legacy else base / self.id

    # ── Niche composition ────────────────────────────────────────────────────────

    def niche_cfg(self, niche_name: str) -> dict:
        data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
        base = dict(data["niches"].get(niche_name, {}))
        base.update(self.niche_overrides.get(niche_name, {}))
        return base

    def platform_enabled(self, platform: str) -> bool:
        return bool(self.platforms.get(platform, {}).get("enabled"))


def _synthesize_legacy() -> Channel:
    """Pre-channels.json install → behave exactly like the original single channel."""
    data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    upload = dict(DEFAULT_UPLOAD)
    if _privacy_override():
        upload["privacy"] = _privacy_override()
    return Channel(
        id=LEGACY_ID,
        display_name="Main (legacy single-channel)",
        niches=list(data.get("niches", {}).keys()),
        schedule=list(data.get("schedule", []) or [data.get("active", "finance")]),
        upload=upload,
        legacy=True,
    )


def list_channels() -> list[str]:
    if not CHANNELS_FILE.exists():
        return [LEGACY_ID]
    return list(json.loads(CHANNELS_FILE.read_text(encoding="utf-8")).get("channels", {}).keys())


def load_channel(channel_id: str | None = None) -> Channel:
    """Resolve and load the active channel (see resolution order in module doc)."""
    if not CHANNELS_FILE.exists():
        return _synthesize_legacy()

    data     = json.loads(CHANNELS_FILE.read_text(encoding="utf-8"))
    channels = data.get("channels", {})
    cid      = (channel_id
                or os.environ.get("RUFUS_CHANNEL", "").strip()
                or data.get("default_channel", ""))
    if cid not in channels:
        raise KeyError(
            f"Unknown channel '{cid}'. Available: {list(channels)} "
            f"(set RUFUS_CHANNEL, pass --channel, or fix default_channel)"
        )

    raw    = channels[cid]
    upload = dict(DEFAULT_UPLOAD)
    upload.update(raw.get("upload", {}))
    # The dashboard's choice wins over the file, the same way every other
    # RUFUS_* setting does — otherwise the button appears to do nothing on a
    # box that has a channels.json.
    if _privacy_override():
        upload["privacy"] = _privacy_override()
    return Channel(
        id=cid,
        display_name=raw.get("display_name", cid),
        language=raw.get("language", "en"),
        voice=raw.get("voice", ""),
        caption_font=raw.get("caption_font", ""),
        niches=list(raw.get("niches", [])),
        schedule=list(raw.get("schedule", [])),
        upload=upload,
        platforms=raw.get("platforms", {"youtube": {"enabled": True}}),
        niche_overrides=raw.get("niche_overrides", {}),
        legacy=False,
    )


def migrate_legacy(channel_id: str = LEGACY_ID) -> None:
    """Move legacy token/learnings into config/channels/<id>/ once channels.json
    is adopted. Safe to run repeatedly; never overwrites existing files."""
    dest = CONFIG_DIR / "channels" / channel_id
    dest.mkdir(parents=True, exist_ok=True)
    for src, name in (
        (CONFIG_DIR / "youtube_token.json", "youtube_token.json"),
        (CONFIG_DIR / "tiktok_token.json",  "tiktok_token.json"),
        (CONFIG_DIR / "learnings.json",     "learnings.json"),
    ):
        target = dest / name
        if src.exists() and not target.exists():
            target.write_bytes(src.read_bytes())
            print(f"[channel] migrated {src.name} → {target}")


if __name__ == "__main__":
    import sys
    ch = load_channel(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"channel:    {ch.id}  ({'legacy shim' if ch.legacy else 'channels.json'})")
    print(f"language:   {ch.language}   voice: {ch.voice or '(default)'}")
    print(f"schedule:   {ch.schedule}")
    print(f"output:     {ch.output_dir}")
    print(f"yt token:   {ch.token_path('youtube')}")
    print(f"secrets:    {ch.client_secrets_path()}")
    print(f"learnings:  {ch.learnings_path}")
