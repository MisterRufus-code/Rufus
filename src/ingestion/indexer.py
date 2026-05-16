"""Scan library → extract features → upsert into Qdrant."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn

from src.ingestion.scanner import scan_library, RawAsset
from src.ingestion.extractor import extract_features, clip_embed_batch
from src.database.vector_store import get_vector_store
from src.logging_config import get_logger

console = Console()
log = get_logger("ingestion.indexer")


_CLIP_BATCH_SIZE = 16


def index_library(
    library_path: Path | None = None,
    force_reindex: bool = False,
    low_power: bool = False,
    clip_batch_size: int = _CLIP_BATCH_SIZE,
) -> dict:
    """
    Full ingestion pipeline: scan → extract → upsert.

    CLIP embeddings are computed in batches of *clip_batch_size* (default 16)
    for GPU efficiency.  All other per-asset work (BLIP caption, YOLO objects,
    colour extraction) still runs one asset at a time.

    Returns a summary dict.
    """
    import time
    from PIL import Image as _PILImage

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

        from src.task_lock import task_lock

        # ------------------------------------------------------------------ #
        # Process assets in batches so CLIP runs a single GPU forward pass    #
        # per batch rather than one per asset.                                 #
        # ------------------------------------------------------------------ #
        for batch_start in range(0, len(assets), clip_batch_size):
            batch = assets[batch_start : batch_start + clip_batch_size]

            # Step 1: extract everything except CLIP for each asset in the batch
            batch_features = []
            batch_images: list[_PILImage.Image] = []
            batch_valid_indices: list[int] = []  # tracks position in batch_features

            for raw in batch:
                progress.advance(task)
                # Skip already-indexed assets before expensive extraction
                raw_id = hashlib.sha256(str(raw.path.resolve()).encode()).hexdigest()[:16]
                if raw_id in existing_ids and not force_reindex:
                    skipped += 1
                    continue
                try:
                    with task_lock("clip_extract"):
                        features = extract_features(raw.path, raw.asset_type)
                    batch_features.append(features)
                    batch_images.append(_PILImage.open(raw.path).convert("RGB")
                                        if raw.asset_type == "image"
                                        else _get_thumbnail_image(raw.path))
                    batch_valid_indices.append(len(batch_features) - 1)
                except FileNotFoundError:
                    console.print(f"[yellow]Skipping missing asset: {raw.path.name}[/yellow]")
                    skipped += 1
                except Exception as exc:
                    console.print(f"[red]Error indexing {raw.path.name}: {exc}[/red]")
                    errors += 1

            if not batch_images:
                continue

            # Step 2: re-embed the whole batch with a single CLIP forward pass
            try:
                with task_lock("clip_extract"):
                    embeddings = clip_embed_batch(batch_images)
                for feat_idx, embedding in zip(batch_valid_indices, embeddings):
                    batch_features[feat_idx].clip_embedding = embedding
            except Exception as exc:
                console.print(f"[red]Batch CLIP embed failed, falling back to per-asset: {exc}[/red]")
                # embeddings from extract_features() are already set; nothing to do

            # Step 3: upsert each asset that isn't already indexed
            for features in batch_features:
                if features.asset_id in existing_ids:
                    skipped += 1
                else:
                    store.upsert(features)
                    indexed += 1

    summary = {"indexed": indexed, "skipped": skipped, "errors": errors}
    log.info("indexing complete", **summary)
    return summary


def _get_thumbnail_image(path: Path):
    """Return the representative thumbnail PIL Image for a video asset."""
    from src.ingestion.extractor import _video_thumbnail
    image, _duration, _motion = _video_thumbnail(path)
    return image
