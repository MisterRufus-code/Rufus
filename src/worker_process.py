"""Run pipeline in an isolated subprocess — crash isolation + memory cleanup.

Each call to run_pipeline_isolated() spawns a fresh Python interpreter so that:
  - A crash in one run cannot corrupt another run's state.
  - PyTorch / CLIP GPU memory is fully released when the child exits.
  - The parent process stays lean regardless of how many runs have completed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_args_file(args: dict) -> Path:
    """Serialise *args* to a temporary JSON file and return its path.

    Using a file (rather than argv or stdin) avoids shell-injection risks and
    sidesteps OS limits on argument length.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        prefix="rufus_pipeline_args_",
    )
    json.dump(args, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline_isolated(
    topic: str,
    niche: str = "general",
    model: str = "mistral",
    voice: str = "af_heart",
    upload: bool = False,
    privacy: str = "private",
    shorts: bool = False,
    geo: str = "US",
    dry_run: bool = False,
    timeout: int = 3600,
) -> dict:
    """Spawn a new Python subprocess to run the pipeline.

    Parameters match the public surface of ``src.pipeline.orchestrator.run_pipeline``
    (using the CLI-friendly aliases: *model* instead of *ollama_model*, *voice*
    instead of *tts_voice*).

    Returns
    -------
    dict with keys:
        success     – bool
        stdout      – captured stdout of the subprocess
        stderr      – captured stderr of the subprocess
        returncode  – integer exit code (0 = clean)
        result      – parsed pipeline result dict, or None on failure
    """
    args: dict[str, Any] = {
        "topic": topic,
        "niche": niche,
        "model": model,
        "voice": voice,
        "upload": upload,
        "privacy": privacy,
        "shorts": shorts,
        "geo": geo,
        "dry_run": dry_run,
    }

    args_file = _write_args_file(args)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "src._pipeline_runner", str(args_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Make sure the temp file is cleaned up even on timeout.
        args_file.unlink(missing_ok=True)
        return {
            "success": False,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": -1,
            "result": None,
            "error": "timeout",
        }
    except Exception as exc:  # pragma: no cover
        args_file.unlink(missing_ok=True)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "result": None,
            "error": str(exc),
        }

    # args_file is removed by the child process on success; guard anyway.
    args_file.unlink(missing_ok=True)

    if completed.returncode != 0:
        return {
            "success": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "result": None,
            "error": f"subprocess exited with code {completed.returncode}",
        }

    # The child writes one JSON line to stdout as its last output.
    result_dict: Optional[dict] = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result_dict = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    return {
        "success": result_dict.get("success", False) if result_dict else False,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "result": result_dict,
    }
