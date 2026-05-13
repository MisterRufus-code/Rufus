"""
FastAPI server — exposes Rufus pipeline as HTTP endpoints for n8n.

Endpoints:
  POST /run/pipeline    — full long-form pipeline
  POST /run/shorts      — Shorts-only pipeline
  POST /run/both        — long form + Shorts in one run
  GET  /status          — system health check
  GET  /metrics         — observability: job stats, error rate, uptime
  GET  /results         — recent pipeline outputs
  GET  /topics/trending — get trending topic for a niche
"""

from __future__ import annotations

import os
import secrets
import threading
import time as _time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_SERVER_START = _time.monotonic()

import signal as _signal
import sys as _sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Shutdown event — set on SIGTERM/SIGINT or app shutdown
_shutdown_event = threading.Event()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    _shutdown_event.set()


app = FastAPI(title="Rufus API", version="1.0.0", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# API Key authentication — set RUFUS_API_KEY in .env (auto-generated if unset)
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _get_configured_key() -> str:
    key = os.getenv("RUFUS_API_KEY", "")
    if not key:
        # Auto-generate and print on first startup — user must save it
        key = secrets.token_urlsafe(32)
        from rich.console import Console
        Console().print(
            f"\n[bold yellow]No RUFUS_API_KEY set. Using auto-generated key for this session:[/bold yellow]\n"
            f"[bold cyan]{key}[/bold cyan]\n"
            f"Add  RUFUS_API_KEY={key}  to your .env to make it permanent.\n"
        )
        os.environ["RUFUS_API_KEY"] = key
    return key

def _require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    if api_key != _get_configured_key():
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

# ---------------------------------------------------------------------------
# Rate limiting — max N concurrent pipeline jobs
# ---------------------------------------------------------------------------
_MAX_CONCURRENT_JOBS = int(os.getenv("RUFUS_MAX_CONCURRENT_JOBS", "2"))
_job_semaphore = threading.Semaphore(_MAX_CONCURRENT_JOBS)

# ---------------------------------------------------------------------------
# Bounded in-memory job store — capped at 200 entries, TTL 24 h
# ---------------------------------------------------------------------------
_JOB_MAX = 200
_JOB_TTL_HOURS = 24
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _prune_jobs() -> None:
    """Remove jobs older than TTL and keep store under _JOB_MAX."""
    cutoff = datetime.utcnow() - timedelta(hours=_JOB_TTL_HOURS)
    to_delete = [
        k for k, v in _jobs.items()
        if v.get("finished_at") and datetime.fromisoformat(v["finished_at"]) < cutoff
    ]
    for k in to_delete:
        del _jobs[k]
    # Hard cap: evict oldest finished jobs first
    if len(_jobs) >= _JOB_MAX:
        def _finished_at_dt(item):
            try:
                return datetime.fromisoformat(item[1].get("finished_at", "2000-01-01"))
            except ValueError:
                return datetime.min

        finished = sorted(
            [(k, v) for k, v in _jobs.items() if v.get("status") in ("done", "failed")],
            key=_finished_at_dt,
        )
        for k, _ in finished[:len(_jobs) - _JOB_MAX + 1]:
            del _jobs[k]


def _new_job(data: dict) -> str:
    """Create a new job entry, prune old ones, return full UUID job_id."""
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _prune_jobs()
        if len(_jobs) >= _JOB_MAX:
            raise HTTPException(status_code=429, detail="Job queue full — try again later")
        _jobs[job_id] = data
    return job_id


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    niche: str = "general"
    topic: Optional[str] = None          # auto-selected if not provided
    model: str = "mistral"
    voice: str = "af_heart"
    upload: bool = False
    privacy: str = "private"
    low_power: bool = True
    geo: str = "US"


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _run_job(job_id: str, topic: str, niche: str, model: str,
             voice: str, upload: bool, privacy: str, low_power: bool,
             shorts: bool, geo: str) -> None:
    if not _job_semaphore.acquire(blocking=True, timeout=5):
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "Server busy — max concurrent jobs reached"
            _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()
        return
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        from src.pipeline.orchestrator import run_pipeline
        result = run_pipeline(
            topic=topic, niche=niche, ollama_model=model,
            tts_voice=voice, upload=upload, privacy=privacy,
            low_power=low_power, shorts=shorts, geo=geo,
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "done" if result.success else "failed"
            _jobs[job_id]["result"] = {
                "topic": result.topic,
                "success": result.success,
                "video_path": str(result.video_path) if result.video_path else None,
                "youtube_video_id": result.youtube_video_id,
                "entropy_score": result.entropy_score,
                "errors": result.errors,
            }
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
    finally:
        _job_semaphore.release()
    with _jobs_lock:
        _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()


def _resolve_topic(topic: Optional[str], niche: str, model: str, geo: str) -> str:
    if topic:
        return topic
    from src.trends import get_trending_topic
    return get_trending_topic(niche=niche, geo=geo, model=model)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def status(_: None = Security(_require_api_key)):
    """System health check — verifies Ollama, Qdrant, FFmpeg."""
    checks = {}

    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        checks["ollama"] = {"ok": True, "models": models}
    except Exception as e:
        checks["ollama"] = {"ok": False, "error": str(e)}

    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333).get_collections()
        checks["qdrant"] = {"ok": True}
    except Exception as e:
        checks["qdrant"] = {"ok": False, "error": str(e)}

    import shutil
    checks["ffmpeg"] = {"ok": bool(shutil.which("ffmpeg"))}

    all_ok = all(v["ok"] for v in checks.values())
    return {"healthy": all_ok, "checks": checks, "timestamp": datetime.utcnow().isoformat()}


@app.post("/run/pipeline", response_model=JobResponse)
def run_pipeline_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks, _: None = Security(_require_api_key)):
    """Start a long-form pipeline job. Returns job_id immediately."""
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    job_id = _new_job({"status": "queued", "topic": topic, "mode": "longform"})
    background_tasks.add_task(
        _run_job, job_id, topic, req.niche, req.model,
        req.voice, req.upload, req.privacy, req.low_power,
        False, req.geo,
    )
    return JobResponse(job_id=job_id, status="queued", message=f"Long-form pipeline started for: {topic}")


@app.post("/run/shorts", response_model=JobResponse)
def run_shorts_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks, _: None = Security(_require_api_key)):
    """Start a Shorts-only pipeline job."""
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    job_id = _new_job({"status": "queued", "topic": topic, "mode": "shorts"})
    background_tasks.add_task(
        _run_job, job_id, topic, req.niche, req.model,
        req.voice, req.upload, req.privacy, req.low_power,
        True, req.geo,
    )
    return JobResponse(job_id=job_id, status="queued", message=f"Shorts pipeline started for: {topic}")


@app.post("/run/both", response_model=JobResponse)
def run_both_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks, _: None = Security(_require_api_key)):
    """Start long-form + Shorts pipeline. One topic, two videos."""
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    job_id = _new_job({"status": "queued", "topic": topic, "mode": "both"})

    def _run_both():
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        results = {}
        sub_ids: list[str] = []
        try:
            for mode, shorts_flag in [("longform", False), ("shorts", True)]:
                sub_id = _new_job({"status": "running", "topic": topic, "mode": mode})
                sub_ids.append(sub_id)
                _run_job(sub_id, topic, req.niche, req.model,
                         req.voice, req.upload, req.privacy, req.low_power,
                         shorts_flag, req.geo)
                with _jobs_lock:
                    results[mode] = _jobs[sub_id].get("result", {})
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["results"] = results
                _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()
        finally:
            # Clean up sub-jobs to avoid leaking memory
            with _jobs_lock:
                for sid in sub_ids:
                    _jobs.pop(sid, None)

    background_tasks.add_task(_run_both)
    return JobResponse(job_id=job_id, status="queued",
                       message=f"Long-form + Shorts pipeline started for: {topic}")


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: None = Security(_require_api_key)):
    """Poll job status. n8n calls this until status == 'done' or 'failed'."""
    # Path traversal guard: job IDs are UUIDs only
    if not all(c in "0123456789abcdef-" for c in job_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job


@app.get("/results")
def get_results(limit: int = 10, _: None = Security(_require_api_key)):
    """Return recent completed jobs."""
    with _jobs_lock:
        done = [
            {"job_id": k, **v}
            for k, v in _jobs.items()
            if v.get("status") in ("done", "failed")
        ]
    return {"results": done[-limit:], "total": len(done)}


@app.get("/metrics")
def metrics(_: None = Security(_require_api_key)):
    """Observability endpoint — job stats, error rate, uptime, system resources."""
    import shutil

    with _jobs_lock:
        all_jobs = list(_jobs.values())

    total = len(all_jobs)
    done = sum(1 for j in all_jobs if j.get("status") == "done")
    failed = sum(1 for j in all_jobs if j.get("status") == "failed")
    running = sum(1 for j in all_jobs if j.get("status") == "running")
    queued = sum(1 for j in all_jobs if j.get("status") == "queued")
    error_rate = round(failed / total, 4) if total else 0.0

    uptime_s = _time.monotonic() - _SERVER_START
    uptime = {"hours": int(uptime_s // 3600), "minutes": int((uptime_s % 3600) // 60)}

    # Disk space for output dir
    disk: dict = {}
    try:
        usage = shutil.disk_usage(os.getenv("OUTPUT_PATH", "output"))
        disk = {
            "total_gb": round(usage.total / 1e9, 2),
            "used_gb": round(usage.used / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
        }
    except Exception:
        pass

    # Semaphore slots available (approximation)
    semaphore_free = max(0, _MAX_CONCURRENT_JOBS - running)

    return {
        "uptime": uptime,
        "jobs": {
            "total": total,
            "done": done,
            "failed": failed,
            "running": running,
            "queued": queued,
            "error_rate": error_rate,
        },
        "capacity": {
            "max_concurrent": _MAX_CONCURRENT_JOBS,
            "slots_free": semaphore_free,
        },
        "disk": disk,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/topics/trending")
def trending_topic(niche: str = "general", geo: str = "US", model: str = "mistral", _: None = Security(_require_api_key)):
    """Get the best trending topic for a niche — called by n8n before triggering pipeline."""
    try:
        from src.trends import get_trending_topic
        topic = get_trending_topic(niche=niche, geo=geo, model=model)
        return {"topic": topic, "niche": niche, "timestamp": datetime.utcnow().isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dashboard", include_in_schema=False)
def dashboard(request: Request):
    """Live HTML dashboard — no auth required for browser convenience."""
    import shutil

    uptime_s = _time.monotonic() - _SERVER_START
    uptime_h = int(uptime_s // 3600)
    uptime_m = int((uptime_s % 3600) // 60)
    uptime_str = f"{uptime_h}h {uptime_m}m"

    # System health checks
    ollama_ok = False
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    ffmpeg_ok = bool(shutil.which("ffmpeg"))

    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333).get_collections()
        qdrant_ok = True
    except Exception:
        pass

    def badge(ok: bool) -> str:
        return '<span class="ok">OK</span>' if ok else '<span class="fail">FAIL</span>'

    # Jobs table — sorted by started_at desc, limit 50
    with _jobs_lock:
        jobs_snapshot = list(_jobs.items())

    def _started_key(item):
        try:
            return item[1].get("started_at") or "0000"
        except Exception:
            return "0000"

    jobs_sorted = sorted(jobs_snapshot, key=_started_key, reverse=True)[:50]

    total = len(jobs_snapshot)
    done = sum(1 for _, v in jobs_snapshot if v.get("status") == "done")
    running = sum(1 for _, v in jobs_snapshot if v.get("status") == "running")
    queued = sum(1 for _, v in jobs_snapshot if v.get("status") == "queued")
    success_rate = f"{round(done / total * 100)}%" if total else "—"

    def status_cell(s: str) -> str:
        colours = {"done": "#00ff88", "failed": "#ff4444", "running": "#ffcc00", "queued": "#aaaaaa"}
        c = colours.get(s, "#ffffff")
        return f'<span style="color:{c};font-weight:bold">{s}</span>'

    rows = ""
    for jid, jdata in jobs_sorted:
        started = jdata.get("started_at", "—")
        finished = jdata.get("finished_at")
        if started != "—" and finished:
            try:
                dur_s = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
                duration = f"{int(dur_s)}s"
            except Exception:
                duration = "—"
        elif started != "—" and jdata.get("status") == "running":
            duration = "running…"
        else:
            duration = "—"
        rows += (
            f"<tr>"
            f"<td>{jid[:8]}</td>"
            f"<td>{jdata.get('mode', '—')}</td>"
            f"<td>{status_cell(jdata.get('status', '?'))}</td>"
            f"<td>{jdata.get('topic', '—')[:40]}</td>"
            f"<td>{started[:19] if started != '—' else '—'}</td>"
            f"<td>{duration}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>Rufus Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f0f0f;color:#e0e0e0;font-family:monospace;font-size:14px;padding:24px}}
  h1{{color:#00ff88;font-size:1.6rem;margin-bottom:4px}}
  .sub{{color:#666;margin-bottom:24px;font-size:12px}}
  .section{{margin-bottom:28px}}
  .section h2{{color:#00ff88;font-size:1rem;border-bottom:1px solid #222;padding-bottom:6px;margin-bottom:12px}}
  .health-row{{display:flex;gap:24px;flex-wrap:wrap}}
  .health-item{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;padding:12px 20px;min-width:140px}}
  .health-item .label{{color:#888;font-size:11px;margin-bottom:4px}}
  .ok{{color:#00ff88;font-weight:bold}}
  .fail{{color:#ff4444;font-weight:bold}}
  .stats-row{{display:flex;gap:24px;flex-wrap:wrap}}
  .stat{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;padding:12px 20px;min-width:120px}}
  .stat .val{{color:#00ff88;font-size:1.4rem;font-weight:bold}}
  .stat .label{{color:#888;font-size:11px;margin-top:2px}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;color:#888;font-size:11px;padding:6px 10px;border-bottom:1px solid #222}}
  td{{padding:6px 10px;border-bottom:1px solid #1a1a1a;font-size:13px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  tr:hover td{{background:#141414}}
</style>
</head>
<body>
<h1>Rufus Dashboard</h1>
<p class="sub">Uptime: {uptime_str} &nbsp;|&nbsp; Auto-refreshes every 10s</p>

<div class="section">
  <h2>System Health</h2>
  <div class="health-row">
    <div class="health-item"><div class="label">Ollama</div>{badge(ollama_ok)}</div>
    <div class="health-item"><div class="label">FFmpeg</div>{badge(ffmpeg_ok)}</div>
    <div class="health-item"><div class="label">Qdrant</div>{badge(qdrant_ok)}</div>
  </div>
</div>

<div class="section">
  <h2>Stats</h2>
  <div class="stats-row">
    <div class="stat"><div class="val">{total}</div><div class="label">Total Jobs</div></div>
    <div class="stat"><div class="val">{success_rate}</div><div class="label">Success Rate</div></div>
    <div class="stat"><div class="val">{running}</div><div class="label">Running</div></div>
    <div class="stat"><div class="val">{queued}</div><div class="label">Queued</div></div>
  </div>
</div>

<div class="section">
  <h2>Jobs (last 50, newest first)</h2>
  <table>
    <thead><tr><th>Job ID</th><th>Mode</th><th>Status</th><th>Topic</th><th>Started</th><th>Duration</th></tr></thead>
    <tbody>{rows if rows else '<tr><td colspan="6" style="color:#555;text-align:center;padding:20px">No jobs yet</td></tr>'}</tbody>
  </table>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _handle_shutdown(signum: int, frame: object) -> None:
    from rich.console import Console
    Console().print("\n[yellow]Rufus API: shutdown signal received — waiting for running jobs...[/yellow]")
    _shutdown_event.set()
    # Give background threads up to 30s to finish, then exit
    _shutdown_event.wait(timeout=30)
    _sys.exit(0)


for _sig in (getattr(_signal, "SIGTERM", None), getattr(_signal, "SIGINT", None)):
    if _sig is not None:
        try:
            _signal.signal(_sig, _handle_shutdown)
        except (OSError, ValueError):
            pass  # not in main thread — uvicorn handles it


