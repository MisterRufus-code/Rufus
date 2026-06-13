"""
conftest.py – pytest fixtures and path setup.
Makes scripts/ importable as a flat package so tests can `from script_writer import …`.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
