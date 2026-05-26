#!/usr/bin/env python3
"""
script_writer.py
Takes a LLaVA scene description and writes a Shorts script using GPT.

Usage:
    python script_writer.py "Man running on mountain at sunrise, athletic"
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

CONFIG_DIR  = Path(__file__).parent.parent / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
KEYS_FILE   = CONFIG_DIR / "keys.json"


def load_niche():
    niches = json.loads(NICHES_FILE.read_text())
    active = niches["active"]
    return niches["niches"][active], active


def load_openai_key() -> str:
    keys = json.loads(KEYS_FILE.read_text())
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("OpenAI key not set in config/keys.json")
    return key


def write_script(scene_description: str) -> str:
    niche, active = load_niche()
    client        = OpenAI(api_key=load_openai_key())

    system_prompt = niche["gpt_system"]
    cta           = niche["cta"]

    user_prompt = f"""Scene in the video: {scene_description}

Write a 30-40 second Shorts script based on what you see in this scene.
End with this call to action: "{cta}"
Output ONLY the script. No labels, no quotes, no explanation."""

    print(f"[gpt] writing script for niche: {active}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=400,
    )

    script = response.choices[0].message.content.strip()
    print(f"[gpt] script written ({len(script.split())} words)")
    return script


def check_blacklist(script: str) -> bool:
    """Returns True if topic seems already used."""
    blacklist_path = CONFIG_DIR / "blacklist.json"
    if not blacklist_path.exists():
        return False
    items  = json.loads(blacklist_path.read_text())
    # Simple check: first 6 words of script
    key    = " ".join(script.lower().split()[:6])
    return any(key in item.lower() for item in items)


def add_to_blacklist(script: str) -> None:
    blacklist_path = CONFIG_DIR / "blacklist.json"
    items = json.loads(blacklist_path.read_text()) if blacklist_path.exists() else []
    key   = " ".join(script.lower().split()[:6])
    items.append(key)
    blacklist_path.write_text(json.dumps(items[-500:], indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_writer.py '<scene description>'")
        sys.exit(1)

    description = " ".join(sys.argv[1:])
    script      = write_script(description)

    if check_blacklist(script):
        print("[blacklist] WARNING: similar script already made")
    else:
        add_to_blacklist(script)

    print(f"\nSCRIPT={script}")
