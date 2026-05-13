"""Scan library → extract features → upsert into Qdrant."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn

from src.ingestion.scanner import scan_library, RawAsset
from src.ingestion.extractor import extract_features
from src.database.vector_store import get_vector_store

console = Console()


def index_library(
    library_path: Path | None = None,
    force_reindex: bool = False,
    low_power: bool = False,
) -> dict:
    """
    Full ingestion pipeline: scan → extract → upsert.
    Returns a summary dict.
    """
    import time

    if low_power:
        from src.ingestion.extractor import set_low_power
        set_low_power(True)

    store = get_vector_store()
    existing_ids = store.get_all_asset_ids() if not force_reindex else set()

    assets = list(scan_library(library_path))
    if not assets:
        console.print("[yellow]No media assets found in library.[/yellow]")
        return {"indexed": 0, "skipped": 0, "errors": 0}

    indexed = skipped = errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing media assets...", total=len(assets))

        for raw in assets:
            progress.advance(task)
            try:
                features = extract_features(raw.path, raw.asset_type)
                if features.asset_id in existing_ids:
                    skipped += 1
                    continue
                store.upsert(features)
                indexed += 1
                if low_power:
                    time.sleep(0.8)  # breathe between files
            except Exception as exc:
                console.print(f"[red]Error indexing {raw.path.name}: {exc}[/red]")
                errors += 1

    return {"indexed": indexed, "skipped": skipped, "errors": errors}
