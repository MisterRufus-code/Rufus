"""
conftest.py – pytest fixtures and path setup.
Makes scripts/ importable as a flat package so tests can `from script_writer import …`.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ── the suite must not read the developer's own config ──────────────────────
#
# THE SECOND TIME THIS EXACT SHAPE HAS BLOCKED THE OWNER FROM RUNNING TESTS.
# The first was encoding: tests read repo sources with the ANSI code page and
# collection died on their box while every CI runner passed. This is the same
# error one layer up — the suite reads config/users.json from the real repo,
# so a machine with actual dashboard users configured fails 176 tests that
# pass on CI purely because CI has no users file.
#
# The tests are not wrong and neither is the auth code. What is wrong is a
# suite whose result depends on files the developer legitimately owns. So the
# user store is pointed at a temp path for the whole session: tests that want
# legacy loopback mode get it, tests that want users create their own, and
# neither can be broken by what is on disk.
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _isolate_the_user_store(tmp_path_factory):
    try:
        import auth
    except Exception:
        yield
        return
    real = auth.USERS_FILE
    auth.USERS_FILE = tmp_path_factory.mktemp("auth") / "users.json"
    try:
        yield
    finally:
        auth.USERS_FILE = real


def pytest_configure(config):
    """Registered so an unknown-marker warning never masks a real one."""
    config.addinivalue_line(
        "markers",
        "real_preflight: exercise the real run preflight instead of the stub "
        "that keeps wizard tests independent of this box's setup")
