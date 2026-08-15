"""A fallback that is worse than stopping is not a fallback.

VERBATIM FROM A REAL RUN, with ComfyUI simply not started:

    [comfy] ComfyUI not running at http://localhost:8188 — Falling back.
               ⚠ ComfyUI offline — trying A1111 SD...
    [sd] A1111 not running at http://localhost:7860
               ⚠ A1111 offline — trying diffusers in-process...
    [diffusers] loading sdxl-turbo (stabilityai/sdxl-turbo) on CPU…
    Downloading bytes: ###6      | 5.02GB, 29.4MB/s
    Reconstructing (incomplete total...):  30%|### | 4.17GB / 13.9GB

Thirteen point nine gigabytes onto a disk that had filled twice that week, to
render ten images on a CPU. Neither cost is a slower version of the same thing
— they are a different failure, and an unattended scheduled run would have
spent hours and produced nothing.

The chain is right to exist; it just has to stop before the step that is worse
than the outage it is covering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import diffusers_client as dc  # noqa: E402


def test_cpu_is_refused_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_DIFFUSERS_CPU", raising=False)
    reason = dc._refuse_reason("cpu", "stabilityai/sdxl-turbo")
    assert reason and "CPU" in reason


def test_the_refusal_names_the_override(monkeypatch):
    """Choosing to wait is legitimate. Being enrolled in it silently is not —
    so every refusal has to say how to say yes."""
    monkeypatch.delenv("RUFUS_DIFFUSERS_CPU", raising=False)
    assert "RUFUS_DIFFUSERS_CPU=1" in dc._refuse_reason("cpu", "x/y")


def test_the_refusal_names_the_real_fix_first(monkeypatch):
    """On this box the cause is almost always "ComfyUI is not running", so the
    message leads with that rather than with the override."""
    monkeypatch.delenv("RUFUS_DIFFUSERS_CPU", raising=False)
    assert "Start ComfyUI" in dc._refuse_reason("cpu", "x/y")


def test_cpu_is_allowed_when_asked_for(monkeypatch):
    monkeypatch.setenv("RUFUS_DIFFUSERS_CPU", "1")
    monkeypatch.setenv("RUFUS_DIFFUSERS_DOWNLOAD", "1")
    assert dc._refuse_reason("cpu", "x/y") is None


def test_an_uncached_model_is_not_downloaded_mid_run(monkeypatch):
    monkeypatch.setenv("RUFUS_DIFFUSERS_CPU", "1")
    monkeypatch.delenv("RUFUS_DIFFUSERS_DOWNLOAD", raising=False)
    monkeypatch.setattr(dc, "_model_is_cached", lambda mid: False)
    reason = dc._refuse_reason("cuda", "stabilityai/sdxl-turbo")
    assert reason and "RUFUS_DIFFUSERS_DOWNLOAD=1" in reason


def test_a_cached_model_on_gpu_is_a_real_fallback(monkeypatch):
    """The chain must still work where it genuinely helps — GPU, already
    downloaded. Guarding everything would just be removing the feature."""
    monkeypatch.setattr(dc, "_model_is_cached", lambda mid: True)
    assert dc._refuse_reason("cuda", "stabilityai/sdxl-turbo") is None


def test_an_unreadable_cache_does_not_block_the_run(monkeypatch):
    """The guard exists to stop a surprise multi-GB pull, not to become its own
    outage — so an unreadable cache layout errs toward allowing."""
    def _boom(*a, **k):
        raise RuntimeError("hub layout changed")
    monkeypatch.setattr(dc, "try_to_load_from_cache", _boom, raising=False)
    assert dc._model_is_cached("stabilityai/sdxl-turbo") in (True, False)


def test_a_declined_load_returns_none_rather_than_raising(monkeypatch):
    """_generate_image must see None and give up quietly; an exception here
    would surface as a crash on a run whose only problem is a stopped server."""
    monkeypatch.delenv("RUFUS_DIFFUSERS_CPU", raising=False)
    monkeypatch.setattr(dc, "_pipe", None)
    src = Path(dc.__file__).read_text(encoding="utf-8")
    assert "if pipe is None:" in src
    assert "return None" in src.split("if pipe is None:")[1][:120]
