"""Entry point for isolated pipeline subprocess.

Invoked by src.worker_process.run_pipeline_isolated() as:

    python -m src._pipeline_runner <args_file>

The args file is a JSON document written by _write_args_file(); it is deleted
here immediately after reading so no sensitive data lingers on disk.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "errors": ["no args file provided"],
                          "youtube_video_id": None, "entropy_score": 0.0}))
        sys.exit(1)

    args_file = Path(sys.argv[1])
    try:
        args = json.loads(args_file.read_text())
    except Exception as exc:
        print(json.dumps({"success": False, "errors": [f"could not read args file: {exc}"],
                          "youtube_video_id": None, "entropy_score": 0.0}))
        sys.exit(1)
    finally:
        args_file.unlink(missing_ok=True)  # clean up regardless

    # Map CLI-friendly key names to orchestrator parameter names.
    orchestrator_args = {
        "topic":        args["topic"],
        "niche":        args.get("niche", "general"),
        "ollama_model": args.get("model", "mistral"),
        "tts_voice":    args.get("voice", "af_heart"),
        "upload":       args.get("upload", False),
        "privacy":      args.get("privacy", "private"),
        "shorts":       args.get("shorts", False),
        "geo":          args.get("geo", "US"),
        "dry_run":      args.get("dry_run", False),
    }

    from src.pipeline.orchestrator import run_pipeline
    result = run_pipeline(**orchestrator_args)

    import dataclasses  # noqa: F401 — kept for reference; result is a dataclass

    output = {
        "success": result.success,
        "errors": result.errors,
        "youtube_video_id": result.youtube_video_id,
        "entropy_score": result.entropy_score,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
