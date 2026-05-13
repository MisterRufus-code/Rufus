"""
pytest conftest — mock unavailable heavy dependencies at the sys.modules level
so tests can run without torch, faiss, qdrant, kokoro, cv2, etc.

This conftest runs before any test collection, so all module-level imports
in src/ that reference these packages will see the mock instead of failing.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _mock_module(name: str, **attrs) -> MagicMock:
    # No spec — we need arbitrary attribute access for `from module import X`
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── Heavy ML / vision packages ────────────────────────────────────────────────
for _pkg in (
    "torch", "torchvision", "torchvision.transforms",
    "clip", "transformers",
    "ultralytics",
    "qdrant_client", "qdrant_client.models",
    "cv2",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
    "sklearn", "sklearn.cluster", "sklearn.preprocessing",
    "kokoro_onnx",
    "imageio",
):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _mock_module(_pkg)

# soundfile — needs real write/read for TTS tests; mock only if not installed
try:
    import soundfile  # noqa: F401
except (ImportError, OSError):
    _sf_mock = _mock_module("soundfile")
    _sf_mock.write = MagicMock()
    _sf_mock.read = MagicMock(return_value=(__import__("numpy").zeros(100), 24000))
    sys.modules["soundfile"] = _sf_mock

# numpy must be real (installed); only mock if missing
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = _mock_module("numpy")

# ── Google auth (installed but broken cryptography in this env) ──────────────
# Force-override regardless of whether the package is installed — the
# system cryptography library has a missing _cffi_backend which causes
# pyo3_runtime.PanicException when google-auth tries to load it.
_google_pkgs = [
    "google", "google.oauth2", "google.oauth2.credentials",
    "google.auth", "google.auth.transport", "google.auth.transport.requests",
    "google_auth_oauthlib", "google_auth_oauthlib.flow",
    "googleapiclient", "googleapiclient.discovery",
    "googleapiclient.errors", "googleapiclient.http",
]
for _pkg in _google_pkgs:
    sys.modules[_pkg] = _mock_module(_pkg)

# Provide HttpError as a real subclass of Exception so it can be raised
class _FakeHttpError(Exception):
    def __init__(self, resp=None, content=b""):
        self.resp = resp or MagicMock()
        self.resp.status = getattr(self.resp, "status", 500)
        self.content = content
        super().__init__(str(content))

sys.modules["googleapiclient.errors"].HttpError = _FakeHttpError

# Provide Credentials mock
_creds_mock = MagicMock()
_creds_mock.from_authorized_user_file = MagicMock(return_value=MagicMock())
sys.modules["google.oauth2.credentials"].Credentials = _creds_mock

# ── yaml (optional) — mock if missing ────────────────────────────────────────
try:
    import yaml  # noqa: F401
except ImportError:
    _yaml_mod = _mock_module("yaml")
    _yaml_mod.safe_load = lambda s: {}
    sys.modules["yaml"] = _yaml_mod
