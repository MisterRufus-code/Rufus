"""
Pre-flight system check — run before any pipeline execution.

Validates all hard dependencies so failures are caught in under 2 seconds
rather than mid-pipeline after expensive downloads or model loads.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()

# Minimum free disk space required before pipeline starts (bytes)
MIN_FREE_DISK_GB = float(
    __import__("os").getenv("RUFUS_MIN_FREE_DISK_GB", "5")
)


def check_ffmpeg() -> Optional[str]:
    if not shutil.which("ffmpeg"):
        return "ffmpeg not found — install with: sudo apt install ffmpeg"
    if not shutil.which("ffprobe"):
        return "ffprobe not found — install with: sudo apt install ffmpeg"
    return None


def check_ollama() -> Optional[str]:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            return "Ollama is running but no models are installed. Run: ollama pull mistral"
        return None
    except Exception as e:
        return f"Ollama not reachable ({e}) — start with: ollama serve"


def check_disk_space(output_dir: Optional[Path] = None) -> Optional[str]:
    import shutil as _shutil
    path = str(output_dir or Path("."))
    try:
        _, _, free = _shutil.disk_usage(path)
        free_gb = free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            return (
                f"Low disk space: {free_gb:.1f} GB free, "
                f"need {MIN_FREE_DISK_GB} GB. Free up space or set RUFUS_MIN_FREE_DISK_GB."
            )
    except Exception:
        pass
    return None


def check_qdrant() -> Optional[str]:
    """Non-blocking — Qdrant failure falls back to FAISS, so just warn."""
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333).get_collections()
        return None
    except Exception:
        return None  # FAISS fallback handles this — not a blocker


def run_preflight(
    output_dir: Optional[Path] = None,
    abort_on_failure: bool = True,
) -> list[str]:
    """
    Run all pre-flight checks. Returns list of error messages.
    If abort_on_failure=True and any check fails, raises SystemExit.
    """
    checks = {
        "FFmpeg":     check_ffmpeg(),
        "Ollama":     check_ollama(),
        "Disk space": check_disk_space(output_dir),
    }

    errors = [(name, msg) for name, msg in checks.items() if msg]
    warnings = []

    table = Table(title="Pre-flight Check", show_header=True, header_style="bold")
    table.add_column("Component", style="cyan", width=14)
    table.add_column("Status")

    for name in checks:
        msg = checks[name]
        if msg:
            table.add_row(name, f"[bold red]FAIL[/bold red]  {msg}")
        else:
            table.add_row(name, "[bold green]OK[/bold green]")

    console.print(table)

    if errors and abort_on_failure:
        console.print("[bold red]Pre-flight failed — fix the issues above before running the pipeline.[/bold red]")
        raise SystemExit(1)

    return [msg for _, msg in errors]
