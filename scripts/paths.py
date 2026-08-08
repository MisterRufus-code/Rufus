#!/usr/bin/env python3
"""
paths.py — one place that decides WHERE Rufus writes.

Why this exists: the debug/media/log roots were hardcoded to the repo folder in
six different modules, so a machine whose repo sits on a small SSD had no way
to send the bulky, ever-growing output (keyframes, voiceovers, rendered mp4s,
the permanent quality-review record) to a roomier drive. These resolvers are
the single source of truth so the modules can't drift apart.

Every path is overridable, and all of them accept an absolute path on any
drive (e.g. W:\\Rufus on Windows, /mnt/data on Linux):

  RUFUS_MEDIA_DIR   whole media_library tree  (default <repo>/media_library)
  RUFUS_DEBUG_DIR   per-run review record     (default <media>/debug)
  RUFUS_OUTPUT_DIR  finished mp4s             (default <media>/output)
  RUFUS_LOG_DIR     run logs                  (default <repo>/logs)

Set RUFUS_MEDIA_DIR alone to move everything media-related at once; the more
specific vars win when you want to split them across drives.
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _env_path(var: str, default: Path) -> Path:
    raw = (os.environ.get(var) or "").strip()
    return Path(raw).expanduser() if raw else default


def media_root() -> Path:
    """The media_library tree (cache/temp/music/output/debug live under here)."""
    return _env_path("RUFUS_MEDIA_DIR", ROOT / "media_library")


def debug_root() -> Path:
    """Per-run review record: script, voiceover, keyframes + their prompts.
    Kept permanently (never swept by housekeeping) — it's the audit trail."""
    return _env_path("RUFUS_DEBUG_DIR", media_root() / "debug")


def output_dir() -> Path:
    """Finished 1080x1920 mp4s."""
    return _env_path("RUFUS_OUTPUT_DIR", media_root() / "output")


def log_dir() -> Path:
    """Per-run text logs."""
    return _env_path("RUFUS_LOG_DIR", ROOT / "logs")


def thumbnails_dir() -> Path:
    """Standalone thumbnails generated outside a video run (thumbnail_gen.py).

    Kept out of debug/ and output/ on purpose: those are per-run artifacts
    swept and reasoned about as a unit, while a thumbnail is a loose asset
    someone asked for by hand and will download to a phone."""
    return _env_path("RUFUS_THUMBNAIL_DIR", media_root() / "thumbnails")


def character_dataset_dir() -> Path:
    """LoRA training sets for the recurring characters (character_dataset.py).

    Separate from thumbnails/ and debug/ because this is a training corpus, not
    a run artifact: it is curated by hand after generation, fed to an external
    trainer, and must survive the debug sweep that clears per-run folders."""
    return _env_path("RUFUS_CHARACTER_DATASET_DIR",
                     media_root() / "character_datasets")


def write_run_report(run_id: str, *, script: str = "", prompts: list[str] | None = None,
                     meta: dict | None = None,
                     motion: list[dict] | None = None) -> "Path | None":
    """Write ONE human-readable markdown file per run collecting everything a
    reviewer needs: the final script, every beat's image prompt, and the run's
    scores/costs.

    Why: the review record was already complete but SCATTERED — script.txt plus
    NN.txt per beat plus the terminal scroll. Reviewing a run (or pasting it to
    someone for a second opinion) meant opening a dozen files. This is the one
    file to open. Appends rather than overwrites so the script section (written
    at Step 4) and the image prompts (Step 2.5) both survive.

    Non-fatal by contract: a reporting failure must never break a render."""
    try:
        d = debug_root() / run_id
        d.mkdir(parents=True, exist_ok=True)
        out = d / "run_report.md"
        parts = []
        if not out.exists():
            parts.append(f"# Rufus run `{run_id}`\n")
        if meta:
            parts.append("## Run\n")
            parts += [f"- **{k}**: {v}\n" for k, v in meta.items()]
            parts.append("\n")
        if script:
            parts.append("## Final script\n\n```\n" + script.strip() + "\n```\n\n")
        if prompts:
            parts.append(f"## Image prompts ({len(prompts)} beats)\n\n")
            for i, p in enumerate(prompts, 1):
                parts.append(f"**Beat {i}**\n\n```\n{p}\n```\n\n")
        if motion:
            # Per-beat motion record: which engine handled it, how long it
            # took, and the exact prompt + sampler settings behind it. This is
            # the raw material for "why did clip 4 look wrong" — without the
            # settings a bad clip is unreproducible, and without the timing
            # there's no way to see an engine quietly getting slower.
            parts.append("## Motion\n\n")
            shown_settings = False
            for r in motion:
                head = (f"- **beat {r.get('beat')}** — {r.get('engine')}: "
                        f"{'ok' if r.get('ok') else 'failed'}")
                if r.get("seconds") is not None:
                    head += f" ({r['seconds']}s)"
                parts.append(head + "\n")
                if r.get("note"):
                    parts.append(f"    - {r['note']}\n")
                if r.get("motion_prompt"):
                    parts.append(f"    - prompt: `{r['motion_prompt']}`\n")
                if not shown_settings:
                    cfg = {k: v for k, v in r.items()
                           if k not in ("beat", "engine", "ok", "seconds",
                                        "note", "motion_prompt")}
                    if cfg:
                        parts.append("\n### Motion engine settings\n\n")
                        parts += [f"- **{k}**: {v}\n" for k, v in sorted(cfg.items())]
                        parts.append("\n")
                        shown_settings = True
            parts.append("\n")

        with out.open("a", encoding="utf-8") as fh:
            fh.write("".join(parts))
        return out
    except Exception as e:
        print(f"[report] run-report write failed (non-fatal): {e}")
        return None
