"""Storage manager — dedup footage, hot/cold tiering, auto-cleanup.

Manages three top-level directories:
    media_library/  — downloaded footage (hot)
    output/         — rendered videos (hot → archive after N days)
    output/_tmp_*   — temp files from renders (cleaned automatically)
    archive/        — cold outputs moved here by archive_old_outputs()
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from src.logging_config import get_logger

log = get_logger(__name__)

# How many bytes to read when fingerprinting a file for dedup
_DEDUP_SAMPLE_BYTES = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StorageReport:
    total_gb: float
    used_gb: float
    free_gb: float
    footage_gb: float
    output_gb: float
    temp_gb: float
    archive_gb: float
    duplicate_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under *path*. Returns 0 if path does not exist."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _bytes_to_gb(n: int) -> float:
    return round(n / 1024 ** 3, 3)


def _sha256_head(path: Path, nbytes: int = _DEDUP_SAMPLE_BYTES) -> str:
    """SHA256 of the first *nbytes* of *path*."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            h.update(f.read(nbytes))
    except OSError:
        return ""
    return h.hexdigest()


def _disk_usage(path: Path) -> tuple[int, int]:
    """Return (total_bytes, free_bytes) for the partition containing *path*."""
    try:
        stat = shutil.disk_usage(path)
        return stat.total, stat.free
    except OSError:
        return 0, 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class StorageManager:
    """
    Central storage manager for Rufus media assets and rendered outputs.

    All paths default to the values in *config*, but can be overridden by
    passing *base_path* (useful for testing).
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        root = base_path or Path(os.getenv("RUFUS_BASE_PATH", "."))
        self._footage_dir: Path = Path(
            os.getenv("MEDIA_LIBRARY_PATH", str(root / config.MEDIA_LIBRARY_PATH))
        )
        self._output_dir: Path = Path(
            os.getenv("OUTPUT_PATH", str(root / config.OUTPUT_PATH))
        )
        self._archive_dir: Path = self._output_dir.parent / "archive"
        self._temp_pattern = "_tmp_"

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> StorageReport:
        """Scan all managed directories and return a StorageReport."""
        footage_bytes = _dir_size_bytes(self._footage_dir)
        output_bytes = _dir_size_bytes(self._output_dir)
        archive_bytes = _dir_size_bytes(self._archive_dir)

        # Temp bytes live inside output — count them separately
        temp_bytes = sum(
            p.stat().st_size
            for p in self._output_dir.rglob(f"*{self._temp_pattern}*")
            if p.is_file()
        ) if self._output_dir.exists() else 0

        # Disk-level totals
        reference = self._footage_dir if self._footage_dir.exists() else Path(".")
        total_bytes, free_bytes = _disk_usage(reference)
        used_bytes = total_bytes - free_bytes

        # Count duplicates without deleting them
        dups = self.find_duplicate_footage()
        dup_count = sum(len(paths) - 1 for paths in dups.values())

        return StorageReport(
            total_gb=_bytes_to_gb(total_bytes),
            used_gb=_bytes_to_gb(used_bytes),
            free_gb=_bytes_to_gb(free_bytes),
            footage_gb=_bytes_to_gb(footage_bytes),
            output_gb=_bytes_to_gb(output_bytes),
            temp_gb=_bytes_to_gb(temp_bytes),
            archive_gb=_bytes_to_gb(archive_bytes),
            duplicate_count=dup_count,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def find_duplicate_footage(self) -> dict[str, list[Path]]:
        """
        Hash the first 64 KB of every video file in the footage directory.

        Returns a mapping of {hash: [path1, path2, ...]} where the list
        has **at least two entries** — i.e. only actual duplicates.
        """
        if not self._footage_dir.exists():
            return {}

        hash_map: dict[str, list[Path]] = {}
        for path in self._footage_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in config.SUPPORTED_VIDEO_EXTS:
                continue
            digest = _sha256_head(path)
            if digest:
                hash_map.setdefault(digest, []).append(path)

        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    def deduplicate_footage(self, dry_run: bool = True) -> int:
        """
        Delete all but the first occurrence of each duplicate group.

        Args:
            dry_run: When True, log what *would* be deleted but do nothing.

        Returns:
            Number of files deleted (0 when dry_run=True).
        """
        dups = self.find_duplicate_footage()
        deleted = 0

        for digest, paths in dups.items():
            # Keep the lexicographically first path for determinism
            keep = sorted(paths)[0]
            for path in paths:
                if path == keep:
                    continue
                if dry_run:
                    log.info(
                        "deduplicate_footage: would delete",
                        path=str(path),
                        kept=str(keep),
                        dry_run=True,
                    )
                else:
                    try:
                        path.unlink()
                        deleted += 1
                        log.info(
                            "deduplicate_footage: deleted duplicate",
                            path=str(path),
                            kept=str(keep),
                        )
                    except OSError as exc:
                        log.warning(
                            "deduplicate_footage: could not delete file",
                            path=str(path),
                            error=str(exc),
                        )

        if not dry_run:
            log.info("deduplicate_footage: complete", deleted=deleted)
        return deleted

    # ------------------------------------------------------------------
    # Output archiving
    # ------------------------------------------------------------------

    def archive_old_outputs(
        self, older_than_days: int = 30, dry_run: bool = True
    ) -> int:
        """
        Move output directories that are older than *older_than_days* to archive/.

        Only top-level subdirectories inside *output/* are considered (one
        directory per render job).

        Args:
            older_than_days: Age threshold.
            dry_run:         Log only — do not move anything.

        Returns:
            Number of directories moved (0 when dry_run=True).
        """
        if not self._output_dir.exists():
            return 0

        cutoff = time.time() - older_than_days * 86400
        moved = 0

        for entry in self._output_dir.iterdir():
            if not entry.is_dir():
                continue
            # Skip the archive dir itself if it happens to be a sibling
            if entry.resolve() == self._archive_dir.resolve():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime > cutoff:
                continue

            dest = self._archive_dir / entry.name
            if dry_run:
                log.info(
                    "archive_old_outputs: would move",
                    src=str(entry),
                    dest=str(dest),
                    dry_run=True,
                )
            else:
                try:
                    self._archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(entry), str(dest))
                    moved += 1
                    log.info(
                        "archive_old_outputs: moved",
                        src=str(entry),
                        dest=str(dest),
                    )
                except (OSError, shutil.Error) as exc:
                    log.warning(
                        "archive_old_outputs: could not move directory",
                        src=str(entry),
                        error=str(exc),
                    )

        if not dry_run:
            log.info("archive_old_outputs: complete", moved=moved)
        return moved

    # ------------------------------------------------------------------
    # Temp cleanup
    # ------------------------------------------------------------------

    def cleanup_temp_files(self) -> int:
        """
        Delete ``_tmp_*`` files and empty directories inside *output/*.

        Returns:
            Total bytes freed.
        """
        if not self._output_dir.exists():
            return 0

        freed = 0

        # Remove _tmp_ files
        for path in self._output_dir.rglob(f"*{self._temp_pattern}*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
                log.debug("cleanup_temp_files: removed temp file", path=str(path), bytes=size)
            except OSError as exc:
                log.warning("cleanup_temp_files: could not remove file", path=str(path), error=str(exc))

        # Prune empty directories (deepest first)
        for dirpath in sorted(self._output_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if dirpath.is_dir() and dirpath != self._output_dir:
                try:
                    dirpath.rmdir()  # only succeeds if truly empty
                    log.debug("cleanup_temp_files: removed empty dir", path=str(dirpath))
                except OSError:
                    pass  # not empty — that is fine

        log.info("cleanup_temp_files: complete", freed_bytes=freed)
        return freed

    # ------------------------------------------------------------------
    # Auto-cleanup
    # ------------------------------------------------------------------

    def auto_cleanup(self, target_free_gb: float = 5.0) -> dict:
        """
        Run a tiered cleanup sequence until *target_free_gb* is free.

        Order:
            1. Temp files (fastest, lowest risk)
            2. Duplicate footage deduplication
            3. Archive outputs older than 30 days

        Stops as soon as the target is met.

        Returns:
            Summary dict with keys: freed_bytes, duplicates_deleted,
            outputs_archived, target_met.
        """
        target_bytes = int(target_free_gb * 1024 ** 3)

        def _free_bytes() -> int:
            _, free = _disk_usage(
                self._footage_dir if self._footage_dir.exists() else Path(".")
            )
            return free

        summary: dict = {
            "freed_bytes": 0,
            "duplicates_deleted": 0,
            "outputs_archived": 0,
            "target_met": False,
        }

        if _free_bytes() >= target_bytes:
            summary["target_met"] = True
            return summary

        # Step 1 — temp files
        freed = self.cleanup_temp_files()
        summary["freed_bytes"] += freed
        log.info("auto_cleanup: step 1 (temp) complete", freed_bytes=freed)

        if _free_bytes() >= target_bytes:
            summary["target_met"] = True
            return summary

        # Step 2 — deduplicate footage
        deleted = self.deduplicate_footage(dry_run=False)
        summary["duplicates_deleted"] = deleted
        log.info("auto_cleanup: step 2 (dedup) complete", deleted=deleted)

        if _free_bytes() >= target_bytes:
            summary["target_met"] = True
            return summary

        # Step 3 — archive old outputs
        archived = self.archive_old_outputs(older_than_days=30, dry_run=False)
        summary["outputs_archived"] = archived
        log.info("auto_cleanup: step 3 (archive) complete", archived=archived)

        summary["target_met"] = _free_bytes() >= target_bytes
        return summary


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Return the process-wide StorageManager singleton."""
    global _manager
    if _manager is None:
        _manager = StorageManager()
    return _manager
