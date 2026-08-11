"""Every file this project reads or writes is UTF-8, stated explicitly.

THE LIVE BUG. On the owner's Hebrew-locale Windows 11 box, Path.read_text()
with no encoding decodes using the ANSI code page — cp1255, not UTF-8. Every
config file in this repo is UTF-8 and several contain em-dashes.

    "—".encode("utf-8").decode("cp1255")  ==  "ג€”"

That is not a hypothetical. run.bat carries the string "ג€”" in a comment
describing it as console mojibake that `chcp 65001` fixes, and a live run
printed a CTA from config/niches.json as

    [gpt] cta: Save this ╫עΓג¼Γא¥ money was never what you think.

where niches.json actually holds "Save this — money was never what you think."
chcp cannot fix it: the text is already corrupted before it reaches stdout,
because the corruption happens at read_text(). PYTHONIOENCODING sets stdio
only; the switch that changes open()/read_text() is PYTHONUTF8, which this
repo never set.

The CTA is not a debug string — it is written into the YouTube description and
the pinned comment. So this shipped corrupted text to the channel.
"""

import ast
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _mode_of(node: ast.Call) -> str:
    """The mode string of an open()/Path.open() call, "" if not given."""
    is_method = isinstance(node.func, ast.Attribute)
    idx = 0 if is_method else 1          # p.open(mode) vs open(path, mode)
    if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
        return str(node.args[idx].value)
    for k in node.keywords:
        if k.arg == "mode" and isinstance(k.value, ast.Constant):
            return str(k.value.value)
    return ""


def _receiver(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id
    return ""


# Image.open / webbrowser.open / io.open-on-bytes are not text file reads.
_NOT_A_TEXT_FILE = {"Image", "webbrowser", "_io", "io", "BytesIO"}


def _unencoded(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in ("read_text", "write_text", "open"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if name == "open":
            if _receiver(node) in _NOT_A_TEXT_FILE or "b" in _mode_of(node):
                continue
        out.append(f"{path.name}:{node.lineno} {name}()")
    return out


def test_no_script_reads_or_writes_a_file_without_saying_utf8():
    offenders = [o for p in sorted(SCRIPTS.glob("*.py")) for o in _unencoded(p)]
    assert not offenders, (
        "these fall back to the locale code page (cp1255 on the owner's box, "
        "which turns an em-dash into 'ג€”'):\n  " + "\n  ".join(offenders))


def test_the_cta_pool_really_does_contain_non_ascii():
    """If this ever becomes ASCII-only the bug above stops being observable,
    and the guard above stops being obviously worth keeping. It is not
    hypothetical today."""
    import json
    niches = json.loads(
        (SCRIPTS.parent / "config" / "niches.json").read_text(encoding="utf-8"))
    ctas = niches["niches"]["money_history"]["cta_pool"]
    assert any(not c.isascii() for c in ctas), ctas


def test_round_tripping_a_cta_through_cp1255_is_the_reported_mojibake():
    """Pins the diagnosis itself, so nobody re-litigates it as a console issue."""
    assert "—".encode("utf-8").decode("cp1255") == "ג€”"


def test_the_launchers_force_utf8_mode():
    """PYTHONIOENCODING is stdio only. PYTHONUTF8 is what changes open()/
    read_text(), and belt-and-braces matters here because a manual PowerShell
    run bypasses the .bat files entirely — which is how every run in this
    session was actually launched."""
    root = SCRIPTS.parent
    for bat in ("run.bat", "run_scheduled.bat", "run_dashboard.bat"):
        f = root / bat
        if f.exists():
            assert "PYTHONUTF8=1" in f.read_text(encoding="utf-8"), bat
