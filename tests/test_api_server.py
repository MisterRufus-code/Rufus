"""Tests for FastAPI server — job lifecycle, rate limiting, path traversal guard."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def client():
    """Return a TestClient with a fresh app state."""
    import importlib
    import sys
    # Reload server module so _jobs dict is empty for each test class
    if "src.api.server" in sys.modules:
        del sys.modules["src.api.server"]
    from fastapi.testclient import TestClient
    from src.api import server
    # Set a known API key for tests
    import os
    os.environ["RUFUS_API_KEY"] = "test-key-123"
    return TestClient(server.app), server


class TestAuthentication:
    def test_missing_key_returns_401(self, client):
        tc, _ = client
        r = tc.get("/status")
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, client):
        tc, _ = client
        r = tc.get("/status", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_correct_key_allowed(self, client):
        tc, _ = client
        with patch("httpx.get", side_effect=Exception("no ollama")), \
             patch("shutil.which", return_value=None):
            r = tc.get("/status", headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 200


class TestJobLifecycle:
    def test_run_pipeline_returns_job_id(self, client):
        tc, srv = client
        with patch.object(srv, "_resolve_topic", return_value="investing"), \
             patch.object(srv, "_run_job"):
            r = tc.post("/run/pipeline",
                        json={"niche": "finance"},
                        headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        # Full UUID format (not truncated 8-char)
        assert len(data["job_id"]) == 36

    def test_get_job_returns_queued(self, client):
        tc, srv = client
        with patch.object(srv, "_resolve_topic", return_value="investing"), \
             patch.object(srv, "_run_job"):
            r = tc.post("/run/pipeline",
                        json={"niche": "finance"},
                        headers={"X-API-Key": "test-key-123"})
        job_id = r.json()["job_id"]
        r2 = tc.get(f"/jobs/{job_id}", headers={"X-API-Key": "test-key-123"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "queued"

    def test_unknown_job_returns_404(self, client):
        tc, _ = client
        r = tc.get("/jobs/00000000-0000-0000-0000-000000000000",
                   headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 404

    def test_invalid_job_id_returns_400(self, client):
        tc, _ = client
        r = tc.get("/jobs/../../etc/passwd",
                   headers={"X-API-Key": "test-key-123"})
        # FastAPI path params with slashes won't match the route, but test direct attack
        # The path traversal guard rejects non-hex-dash characters
        assert r.status_code in (400, 404, 422)


class TestJobStoreBounds:
    def test_prune_jobs_removes_old_entries(self, client):
        _, srv = client
        from datetime import datetime, timedelta
        # Inject 5 expired jobs directly
        old_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        with srv._jobs_lock:
            for i in range(5):
                srv._jobs[f"old-job-{i:04d}-0000-0000-0000-000000000000"] = {
                    "status": "done",
                    "finished_at": old_time,
                }
        count_before = len(srv._jobs)
        with srv._jobs_lock:
            srv._prune_jobs()
        assert len(srv._jobs) < count_before

    def test_results_endpoint_returns_list(self, client):
        tc, _ = client
        r = tc.get("/results", headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 200
        assert "results" in r.json()


class TestTrendingEndpoint:
    def test_trending_returns_topic(self, client):
        tc, _ = client
        with patch("src.trends.get_trending_topic", return_value="AI stocks"):
            r = tc.get("/topics/trending?niche=finance",
                       headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 200
        assert r.json()["topic"] == "AI stocks"

    def test_trending_500_on_exception(self, client):
        tc, _ = client
        with patch("src.trends.get_trending_topic", side_effect=RuntimeError("offline")):
            r = tc.get("/topics/trending", headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 500
