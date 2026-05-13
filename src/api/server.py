"""
FastAPI server — exposes Rufus pipeline as HTTP endpoints for n8n.

Endpoints:
  POST /run/pipeline   — full long-form pipeline
  POST /run/shorts     — Shorts-only pipeline
  POST /run/both       — long form + Shorts in one run
  GET  /status         — system health check
  GET  /results        — recent pipeline outputs
  GET  /topics/trending — get trending topic for a niche
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Rufus API", version="1.0.0")

# In-memory job store (good enough for single-machine use)
_jobs: dict[str, dict] = {}


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
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
    try:
        from src.pipeline.orchestrator import run_pipeline
        result = run_pipeline(
            topic=topic, niche=niche, ollama_model=model,
            tts_voice=voice, upload=upload, privacy=privacy,
            low_power=low_power, shorts=shorts, geo=geo,
        )
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
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
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
def status():
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
def run_pipeline_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a long-form pipeline job. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    _jobs[job_id] = {"status": "queued", "topic": topic, "mode": "longform"}
    background_tasks.add_task(
        _run_job, job_id, topic, req.niche, req.model,
        req.voice, req.upload, req.privacy, req.low_power,
        False, req.geo,
    )
    return JobResponse(job_id=job_id, status="queued", message=f"Long-form pipeline started for: {topic}")


@app.post("/run/shorts", response_model=JobResponse)
def run_shorts_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a Shorts-only pipeline job."""
    job_id = str(uuid.uuid4())[:8]
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    _jobs[job_id] = {"status": "queued", "topic": topic, "mode": "shorts"}
    background_tasks.add_task(
        _run_job, job_id, topic, req.niche, req.model,
        req.voice, req.upload, req.privacy, req.low_power,
        True, req.geo,
    )
    return JobResponse(job_id=job_id, status="queued", message=f"Shorts pipeline started for: {topic}")


@app.post("/run/both", response_model=JobResponse)
def run_both_endpoint(req: PipelineRequest, background_tasks: BackgroundTasks):
    """Start long-form + Shorts pipeline. One topic, two videos."""
    job_id = str(uuid.uuid4())[:8]
    topic = _resolve_topic(req.topic, req.niche, req.model, req.geo)
    _jobs[job_id] = {"status": "queued", "topic": topic, "mode": "both"}

    def _run_both():
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        results = {}
        for mode, shorts_flag in [("longform", False), ("shorts", True)]:
            sub_id = f"{job_id}_{mode}"
            _jobs[sub_id] = {"status": "running"}
            _run_job(sub_id, topic, req.niche, req.model,
                     req.voice, req.upload, req.privacy, req.low_power,
                     shorts_flag, req.geo)
            results[mode] = _jobs[sub_id].get("result", {})
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["results"] = results
        _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()

    background_tasks.add_task(_run_both)
    return JobResponse(job_id=job_id, status="queued",
                       message=f"Long-form + Shorts pipeline started for: {topic}")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Poll job status. n8n calls this until status == 'done' or 'failed'."""
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job


@app.get("/results")
def get_results(limit: int = 10):
    """Return recent completed jobs."""
    done = [
        {"job_id": k, **v}
        for k, v in _jobs.items()
        if v.get("status") in ("done", "failed")
        and "_" not in k  # skip sub-jobs
    ]
    return {"results": done[-limit:], "total": len(done)}


@app.get("/topics/trending")
def trending_topic(niche: str = "general", geo: str = "US", model: str = "mistral"):
    """Get the best trending topic for a niche — called by n8n before triggering pipeline."""
    from src.trends import get_trending_topic
    topic = get_trending_topic(niche=niche, geo=geo, model=model)
    return {"topic": topic, "niche": niche, "timestamp": datetime.utcnow().isoformat()}
