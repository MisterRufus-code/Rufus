"""Cancelling a motion job we've stopped waiting for.

Measured from a live 10-clip run that took 12,952s (3h36m). Giving up on a clip
used to mean only that RUFUS stopped waiting — the job kept running on ComfyUI,
which has ONE serial queue, so the next clip was submitted behind a job still
holding the GPU. The ComfyUI-side log shows the whole mechanism:

    Prompt executed in 00:13:36     <- clip 1, normal
    Prompt executed in 00:10:46     <- clip 2, normal
    got prompt                      <- clip 3 starts
    12/12 [04:18]                   <- its SAMPLER finishes in 4 minutes
    got prompt                      <- clip 4  ) submitted after RUFUS gave
    got prompt                      <- clip 5  ) up on clip 3, and now
    got prompt                      <- clip 6  ) queued behind it
    Prompt executed in 01:31:04     <- clip 3 finally ends: 91 MINUTES

Clip 3 sampled normally; the other 87 minutes were VAE decode thrashing on a
16GB-RAM box. Clips 4 and 5 then reported timeouts of their own — but they had
never started. Three "failures" and ~90 minutes from one actual cause.

Cancelling makes that one lost clip instead of three.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import svd_client  # noqa: E402


class _Recorder:
    """Stands in for requests, recording what the client would send."""

    def __init__(self, running_ids=(), fail_on=None):
        self.running_ids = list(running_ids)
        self.fail_on = fail_on
        self.posts = []
        self.gets = []

    def get(self, url, **kw):
        self.gets.append(url)
        if self.fail_on == "get":
            raise RuntimeError("comfy unreachable")
        rec = self

        class R:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return {"queue_running": [[0, i] for i in rec.running_ids],
                        "queue_pending": []}
        return R()

    def post(self, url, **kw):
        self.posts.append((url, kw.get("json")))
        if self.fail_on == "post":
            raise RuntimeError("comfy unreachable")

        class R:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass
        return R()


def _abandon(monkeypatch, **kw):
    rec = _Recorder(**kw)
    monkeypatch.setattr(svd_client, "requests", rec)
    svd_client._abandon("abc123def456")
    return rec


def test_a_running_job_is_interrupted(monkeypatch):
    """The 91-minute case: the job holds the GPU, so only /interrupt frees it."""
    rec = _abandon(monkeypatch, running_ids=["abc123def456"])
    assert any(u.endswith("/interrupt") for u, _ in rec.posts)


def test_a_pending_job_is_deleted_from_the_queue(monkeypatch):
    rec = _abandon(monkeypatch, running_ids=["something-else"])
    deletes = [body for u, body in rec.posts
               if u.endswith("/queue") and body and "delete" in body]
    assert deletes and deletes[0]["delete"] == ["abc123def456"]


def test_a_job_that_is_not_running_is_not_interrupted(monkeypatch):
    """/interrupt takes no id and kills whatever is executing. Sending it for a
    merely-QUEUED job would kill an unrelated clip that had already started."""
    rec = _abandon(monkeypatch, running_ids=["another-job"])
    assert not any(u.endswith("/interrupt") for u, _ in rec.posts)


def test_an_unreadable_queue_still_interrupts(monkeypatch):
    """If we can't tell whether it's running, assume it is — leaving the GPU
    held is what cost 90 minutes, and the queue is serial and single-writer."""
    rec = _abandon(monkeypatch, fail_on="get")
    assert any(u.endswith("/interrupt") for u, _ in rec.posts)


def test_cancel_failure_never_raises(monkeypatch):
    """Fail-open, like every other optional step: a clip we already gave up on
    must not turn into a crashed run."""
    _abandon(monkeypatch, running_ids=["abc123def456"], fail_on="post")


def test_await_frames_cancels_on_timeout(monkeypatch):
    """The wiring: a timeout must actually reach _abandon. Without this the
    next clip is submitted behind a job still on the GPU."""
    # A negative timeout puts the deadline in the past, so the poll loop is
    # skipped and the timeout branch runs at once. Stubbing the clock is the
    # obvious alternative and the wrong one: svd_client.time IS the stdlib time
    # module, so patching it reaches pytest's own timing and hangs the run
    # instead of the code under test.
    called = {}
    monkeypatch.setattr(svd_client, "_abandon",
                        lambda pid: called.setdefault("pid", pid))
    assert svd_client._await_frames("job-42", timeout=-1) == []
    assert called["pid"] == "job-42"


def test_successful_frames_are_not_cancelled(monkeypatch):
    """Only abandonment cancels. A job that delivered must never be touched."""
    class R:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"job-1": {"outputs": {"9": {"images": [
                {"filename": "f.png", "subfolder": "", "type": "output"}]}}}}
        content = b"PNGDATA"

    monkeypatch.setattr(svd_client.requests, "get", lambda *a, **kw: R())
    monkeypatch.setattr(svd_client, "_abandon",
                        lambda pid: (_ for _ in ()).throw(
                            AssertionError("cancelled a job that succeeded")))
    assert svd_client._await_frames("job-1", timeout=30) == [b"PNGDATA"]
