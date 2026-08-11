"""Node-backed engines must spawn npx by absolute path, not by bare name.

On Windows npx is `npx.cmd`. `shutil.which("npx")` finds it because which
consults PATHEXT; CreateProcess does not, so `subprocess.run(["npx", ...])`
raises `[WinError 2] The system cannot find the file specified` on a machine
where npx is installed and on PATH.

That is what every run on the owner's box hit:

    [4/4] Remotion render: 9 clip(s) → 42.8s + music…
    ⚠ Remotion failed ([WinError 2] The system cannot find the file specified)
      → falling back to FFmpeg

The readiness check passed (which found npx), the spawn failed, and the
fallback chain absorbed it — so the cinematic renderer was never once used, and
the log blamed Remotion. Same defect disabled HyperFrames via is_available().
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ------------------------------------------------------------------ remotion

def test_remotion_spawns_the_resolved_absolute_path(monkeypatch, tmp_path):
    import remotion_renderer

    fake = tmp_path / "npx.cmd"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(remotion_renderer.shutil, "which",
                        lambda name: str(fake) if name == "npx" else None)

    assert remotion_renderer._npx() == str(fake)
    assert remotion_renderer._npx() != "npx", (
        "a bare 'npx' is what CreateProcess cannot resolve on Windows"
    )


def test_remotion_says_node_is_missing_when_it_really_is(monkeypatch):
    import remotion_renderer

    monkeypatch.setattr(remotion_renderer.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="npx not found"):
        remotion_renderer._npx()


def test_remotion_render_command_starts_with_the_resolved_path(monkeypatch, tmp_path):
    """The argv actually handed to subprocess is the one that matters — the
    readiness check passing is precisely what made this bug invisible."""
    import remotion_renderer

    fake = tmp_path / "npx.cmd"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(remotion_renderer.shutil, "which",
                        lambda name: str(fake) if name == "npx" else None)

    source = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    assert '"npx", "remotion", "render"' not in source, (
        "the render argv must use _npx(), not the bare name"
    )
    assert "_npx(), \"remotion\", \"render\"" in source


# --------------------------------------------------------------- hyperframes

def test_hyperframes_launcher_resolves_its_executable(monkeypatch, tmp_path):
    import hyperframes_client

    fake = tmp_path / "npx.cmd"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(hyperframes_client.shutil, "which",
                        lambda name: str(fake) if name == "npx" else None)

    argv = hyperframes_client._launcher()
    assert argv[0] == str(fake)
    assert argv[1:] == ["--yes", "hyperframes"]


def test_hyperframes_launcher_honours_the_env_override(monkeypatch, tmp_path):
    import hyperframes_client

    fake = tmp_path / "hyperframes.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("HYPERFRAMES_CMD", "hyperframes --quiet")
    monkeypatch.setattr(hyperframes_client.shutil, "which",
                        lambda name: str(fake) if name == "hyperframes" else None)

    assert hyperframes_client._launcher() == [str(fake), "--quiet"]


def test_hyperframes_launcher_falls_back_to_the_bare_name(monkeypatch):
    """An unresolvable name still spawns — it fails with a clear error from
    subprocess rather than an empty argv or a crash inside the launcher."""
    import hyperframes_client

    monkeypatch.delenv("HYPERFRAMES_CMD", raising=False)
    monkeypatch.setattr(hyperframes_client.shutil, "which", lambda name: None)
    assert hyperframes_client._launcher() == ["npx", "--yes", "hyperframes"]


# ------------------------------------------------------------------ the class

def test_no_engine_spawns_a_bare_npx():
    """A regression net over the whole package: any new Node-backed engine has
    to resolve its executable too.

    Parsed rather than grepped, so prose about the bug in a docstring is not
    mistaken for the bug. The only legitimate bare "npx" is the one handed to
    shutil.which — that call is the resolution.
    """
    import ast

    scripts = Path(__file__).parent.parent / "scripts"
    offenders = []
    for py in sorted(scripts.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))

        resolved = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "which":
                resolved.update(id(a) for a in node.args)

        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and node.value == "npx"
                    and id(node) not in resolved):
                offenders.append(f"{py.name}:{node.lineno}")

    assert not offenders, (
        "spawn npx by the path shutil.which returns — a bare 'npx' is "
        "unresolvable by CreateProcess on Windows: " + ", ".join(offenders)
    )
