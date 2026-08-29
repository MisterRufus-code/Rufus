#!/usr/bin/env python3
"""Refuse to start a run that cannot possibly finish, and say why.

WHAT A FRESH INSTALL ACTUALLY DID. With no config/keys.json — the state every
new machine is in — `python scripts/main.py` ran for minutes. It fetched a
dozen RSS feeds, queried Hacker News twice, tried Reddit, OpenAlex,
StackExchange, the Library of Congress and eight Wikipedia articles, rejected
four seeds through the supervisor, reported "Pre-analysis failed (non-fatal)"
about the very file it was about to die on, and then stopped at step 4 with:

    ✗ Step 4 failed: [Errno 2] No such file or directory:
      '/home/user/Rufus/config/keys.json'

Every one of those minutes was spent on work that could not be used, and the
thing that was missing was knowable before the first byte went out. What the
person is left holding is a Python errno for a file they have never heard of.

THE PRINCIPLE IS ALREADY IN THIS TREE, ONE LEVEL DOWN. main.py runs the fact
gate "BEFORE burning FLUX/SD generation time on doomed images" — check the
cheap precondition before spending the expensive thing. This is the same move
applied to the run itself: the API key, the encoder and the renderer are
checkable in milliseconds, and a run without them is doomed from the first
line.

WHAT BELONGS HERE AND WHAT DOES NOT. Only conditions under which THIS run, with
THIS configuration, cannot reach a finished video. Not "would be better with" —
health_check.py is the broad survey, and a preflight that fires on things a run
could survive is a preflight people learn to run past. Every check is therefore
conditional on the configuration: a Pexels key matters only when Pexels is the
footage source, ComfyUI only when ComfyUI is drawing, YouTube credentials only
when this run would actually upload.

Each blocker owes three things: WHAT is missing, WHY this run cannot finish
without it, and the exact command or edit that fixes it. A message that stops
at the first is the errno again in a nicer font.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
KEYS_FILE = CONFIG_DIR / "keys.json"

# Placeholder values the templates ship with. A key that is still one of these
# is not a key, and treating it as one produces a 401 forty minutes later.
PLACEHOLDER_PREFIXES = ("YOUR_", "FILL_", "sk-...", "<")


class Blocker:
    """One reason this run cannot finish, and what to do about it."""

    def __init__(self, what: str, why: str, fix: str):
        self.what = what
        self.why = why
        self.fix = fix

    def __repr__(self) -> str:      # test failures read better this way
        return f"Blocker({self.what!r})"

    def lines(self) -> list[str]:
        return [f"  ✗  {self.what}", f"     {self.why}", f"     Fix: {self.fix}"]


def _keys() -> dict | None:
    """config/keys.json as a dict, None when there is no file, {} when broken.

    THREE STATES AND NOT TWO. Absent means "you have not set this up yet" and
    has a different fix from "the file is there and the JSON is malformed" —
    which on a hand-edited file is usually a trailing comma, and is worth
    saying rather than reporting as missing.
    """
    try:
        raw = KEYS_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_set(keys: dict | None, name: str) -> bool:
    value = ((keys or {}).get(name) or "").strip()
    return bool(value) and not value.startswith(PLACEHOLDER_PREFIXES)


def _comfy_up(url: str, timeout: float = 3.0) -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def blockers(env: dict | None = None, *, skip_upload: bool = False,
             probe_network: bool = True) -> list[Blocker]:
    """Everything that would stop this run, given this configuration."""
    env = dict(os.environ if env is None else env)
    found: list[Blocker] = []
    keys = _keys()

    video_source = (env.get("RUFUS_VIDEO_SOURCE") or "").strip().lower()
    local_llm = bool((env.get("RUFUS_LLM_BASE_URL")
                      or env.get("OPENAI_BASE_URL") or "").strip())
    env_key = bool((env.get("RUFUS_LLM_KEY")
                    or env.get("OPENAI_API_KEY") or "").strip())

    # ── the writer, which every run goes through ────────────────────────────
    if not (local_llm or env_key):
        if keys is None:
            found.append(Blocker(
                "config/keys.json does not exist",
                "Every run writes its script with a language model, so without "
                "a key there is no video to make — this is the file that holds "
                "it.",
                "cp config/keys.json.template config/keys.json  (Windows: copy)"
                " then put your OpenAI key in it as \"openai\". "
                "RUFUS_LLM_BASE_URL instead points the writer at a local "
                "server and needs no key at all.",
            ))
        elif not keys:
            found.append(Blocker(
                "config/keys.json is not valid JSON",
                "The file is there but cannot be read, so nothing in it "
                "reaches the pipeline. On a hand-edited file this is almost "
                "always a trailing comma or a missing quote.",
                "Open it and check the punctuation, or start again from "
                "config/keys.json.template.",
            ))
        elif not _is_set(keys, "openai"):
            found.append(Blocker(
                "no OpenAI key in config/keys.json",
                "The file exists but its \"openai\" value is empty or still the "
                "placeholder the template ships with, which fails as a 401 "
                "partway through step 4 rather than now.",
                "Put a real key in the \"openai\" field, or set "
                "RUFUS_LLM_BASE_URL to run against a local model instead.",
            ))

    # ── the encoder, which every render goes through ────────────────────────
    if not shutil.which("ffmpeg"):
        found.append(Blocker(
            "ffmpeg is not on PATH",
            "Every video is cut, captioned and muxed by ffmpeg. The pipeline "
            "would research, write, draw and voice a whole video before "
            "discovering it cannot assemble one.",
            "Windows: winget install Gyan.FFmpeg  ·  macOS: brew install "
            "ffmpeg  ·  Debian/Ubuntu: sudo apt install ffmpeg. Then reopen "
            "the terminal so PATH is picked up.",
        ))

    # ── the picture engine this configuration actually selected ─────────────
    if video_source == "pexels" and not _is_set(keys, "pexels"):
        found.append(Blocker(
            "RUFUS_VIDEO_SOURCE=pexels with no Pexels key",
            "Pexels is the footage source for this run and its API refuses "
            "unauthenticated requests, so there would be no footage at all.",
            "Add a free key from pexels.com/api as \"pexels\" in "
            "config/keys.json, or unset RUFUS_VIDEO_SOURCE to draw the "
            "pictures locally instead.",
        ))

    if video_source == "comfy" and probe_network:
        comfy_url = (env.get("COMFY_URL") or "http://127.0.0.1:8188").strip()
        if not _comfy_up(comfy_url):
            found.append(Blocker(
                f"ComfyUI is not answering at {comfy_url}",
                "ComfyUI draws every picture in this configuration. A run "
                "started against a dead renderer gets an empty image back for "
                "every prompt — which is exactly how a gallery once stopped at "
                "9 of 38 and reported itself finished.",
                "Start ComfyUI and wait for it to finish loading, or set "
                "COMFY_URL if it listens somewhere else.",
            ))

    # ── uploading, only when this run would actually upload ─────────────────
    wants_upload = not skip_upload and (
        env.get("RUFUS_AUTO_UPLOAD", "").strip() in ("1", "true", "yes", "on"))
    if wants_upload:
        secrets = CONFIG_DIR / "client_secrets.json"
        token = CONFIG_DIR / "youtube_token.json"
        if not secrets.exists() and not token.exists():
            found.append(Blocker(
                "RUFUS_AUTO_UPLOAD is on with no YouTube credentials",
                "This run is configured to publish without a person "
                "approving it, and neither the OAuth client nor a saved token "
                "is present — so the video would be finished and then stranded.",
                "Follow the YouTube API section of the README to create "
                "config/client_secrets.json, or drop RUFUS_AUTO_UPLOAD and "
                "approve from the dashboard instead.",
            ))

    return found


def check(env: dict | None = None, *, skip_upload: bool = False,
          probe_network: bool = True) -> list[Blocker]:
    """Print the blockers, if any, and return them. Never exits by itself.

    Printing and stopping are kept apart so the dashboard can ask the same
    question without a library call being able to kill the web server.
    """
    found = blockers(env, skip_upload=skip_upload, probe_network=probe_network)
    if not found:
        return found
    print("\nThis run cannot finish as configured.\n")
    for b in found:
        for line in b.lines():
            print(line)
        print()
    print("  Nothing has been spent. Fix the above and run again;")
    print("  `python scripts/health_check.py` checks the rest of the setup.\n")
    return found


def check_or_exit(env: dict | None = None, *, skip_upload: bool = False,
                  probe_network: bool = True) -> None:
    """The form main.py uses: stop before the first byte goes out."""
    if check(env, skip_upload=skip_upload, probe_network=probe_network):
        raise SystemExit(2)


if __name__ == "__main__":
    found = check()
    if not found:
        print("\n  Nothing is missing. This configuration can produce a "
              "video.\n")
    raise SystemExit(2 if found else 0)
