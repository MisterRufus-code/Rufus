#!/usr/bin/env python3
"""
llm.py — one place that decides WHICH model answers.

WHY THIS EXISTS. Twelve call sites across ten modules each did the same two
lines: read config/keys.json, construct `OpenAI(api_key=key)`. Nothing was
wrong with any of them, and together they made one thing impossible — pointing
the pipeline at a different endpoint. A local model on the owner's own 3090 is
not a feature request, it is a base_url, and there was nowhere to put it.

So this is the client factory, and the callers ask it instead of building their
own. Cloud OpenAI stays the default and behaves exactly as before.

    RUFUS_LLM_BASE=http://localhost:11434/v1     Ollama
    RUFUS_LLM_BASE=http://localhost:8000/v1      vLLM / LM Studio / llama.cpp
    RUFUS_LLM_KEY=ollama                         most local servers ignore it

WHY AN OPENAI-COMPATIBLE ENDPOINT AND NOT A NEW CLIENT. Ollama, vLLM,
llama.cpp's server and LM Studio all speak the OpenAI chat-completions shape,
including `response_format={"type": "json_object"}` on the ones that matter
here. Reusing the same client means the gates, the retries and the JSON
parsing already written stay exactly as they are — the only thing that changes
is which machine answers.

PER-ROLE MODELS, because they are not one job. The hook factory runs eight
candidates at temperature 1.0 and wants speed; the story architect wants the
best reasoning available; the storyboard has to hold fourteen shots in its
head at once. A single RUFUS_LLM_MODEL would flatten those into one choice,
so each role can be pointed somewhere different and falls back to the model
the caller already asked for:

    RUFUS_LLM_MODEL              everything, unless overridden below
    RUFUS_LLM_MODEL_HOOK_GEN     per-role, named for config/script_standards
    RUFUS_LLM_MODEL_STORYBOARD
    RUFUS_LLM_MODEL_SUPERVISOR

CONTRACT: never raises on configuration. A missing key with a local base is
fine (local servers do not check it); a missing key with no local base is the
same FileNotFoundError-shaped failure the callers already handle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE = CONFIG_DIR / "keys.json"

# What a local server accepts as a key. Ollama and llama.cpp ignore the value
# entirely but the OpenAI client refuses to construct without one.
_LOCAL_PLACEHOLDER = "local"


def base_url() -> str:
    """The endpoint, or "" for cloud OpenAI."""
    return (os.environ.get("RUFUS_LLM_BASE") or "").strip().rstrip("/")


def is_local() -> bool:
    return bool(base_url())


def api_key() -> str:
    """RUFUS_LLM_KEY, else config/keys.json, else a placeholder when local.

    A local server needs no real key, and demanding one would mean keeping a
    fake OpenAI key in config just to run offline — a confusing thing to find
    in a file called keys.json six months later.
    """
    env = (os.environ.get("RUFUS_LLM_KEY") or "").strip()
    if env:
        return env
    key = ""
    try:
        key = json.loads(KEYS_FILE.read_text(encoding="utf-8")).get("openai", "")
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        key = ""
    key = (key or "").strip()
    if key.startswith(("YOUR_", "FILL_")):
        key = ""
    if not key and is_local():
        return _LOCAL_PLACEHOLDER
    return key


def usable() -> bool:
    """Whether a call would have somewhere to go. Callers already treat "no
    key" as "skip this stage and carry on"; this is that question, asked once.
    """
    return bool(api_key())


def client(key: str | None = None):
    """An OpenAI-compatible client for whichever endpoint is configured."""
    from openai import OpenAI
    key = (key or "").strip() or api_key()
    if is_local():
        return OpenAI(api_key=key or _LOCAL_PLACEHOLDER, base_url=base_url())
    return OpenAI(api_key=key)


def model_for(role: str, default: str) -> str:
    """The model for one role, most specific override first.

    `default` is whatever the caller was going to use — config/script_standards
    for the writers, a module constant for the rest — so a pipeline with none
    of these set behaves exactly as it did.
    """
    specific = (os.environ.get(f"RUFUS_LLM_MODEL_{role.upper()}") or "").strip()
    if specific:
        return specific
    general = (os.environ.get("RUFUS_LLM_MODEL") or "").strip()
    if general:
        return general
    return default


_said = False


def announce() -> None:
    """Say once per run which brain is answering.

    A run that came out differently because it was served by a 14B model at
    home rather than gpt-4o is a run whose log has to say so. Once, not per
    call — there are hundreds.
    """
    global _said
    if _said or not is_local():
        return
    _said = True
    model = (os.environ.get("RUFUS_LLM_MODEL") or "").strip()
    print(f"[llm] local endpoint: {base_url()}"
          + (f" · model {model}" if model else " · per-caller models"))
