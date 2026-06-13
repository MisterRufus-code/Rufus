#!/usr/bin/env python3
"""
script_logger.py
Append-only JSONL log of every script-writer attempt: hook generation,
hook scoring, body generation, body scoring. One file per UTC day under
logs/scripts/YYYYMMDD.jsonl.

Source of truth for script quality analysis. The DB mirrors aggregates;
this captures the full reasoning trace.
"""

import json
import time
import uuid
from pathlib import Path

ROOT    = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs" / "scripts"

# OpenAI pricing per 1k tokens (input, output). Used for cost estimates.
# Updated 2025-11. If prices change, update here.
PRICING = {
    "gpt-4o":      (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
}


def new_run_id() -> str:
    """Generate a unique run_id for one full write_script() invocation."""
    return time.strftime("%Y%m%d-") + uuid.uuid4().hex[:10]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost estimate for a single OpenAI call."""
    p_in, p_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1000.0) * p_in + (completion_tokens / 1000.0) * p_out


def log_attempt(record: dict) -> None:
    """Append one JSONL record to today's script log.

    Caller fills in any subset of the documented schema; we add ts if missing.
    Never raises — logging failure must not abort a render.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        path = LOG_DIR / f"{time.strftime('%Y%m%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[script_logger] log failed (non-fatal): {e}")
