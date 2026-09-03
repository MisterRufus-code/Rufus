"""Registering the pip-installed CUDA runtime so GPU whisper can find it.

THE LIVE FAILURE. pip reported nvidia-cublas-cu12 12.9.2.10 and
nvidia-cudnn-cu12 9.24.0.43 "already satisfied", and whisper still printed
"Library cublas64_12.dll is not found or cannot be loaded" on every run and
transcribed on CPU. The DLLs were on disk the whole time; Windows just doesn't
search site-packages/nvidia/<lib>/bin unless the directory is registered.

_add_nvidia_dll_dirs existed to do exactly that, and did nothing: `nvidia` is a
NAMESPACE package, so nvidia.__file__ is None, Path(None) raises TypeError, and
a bare `except Exception: pass` ate it. Silence is what made this expensive —
the symptom pointed at a missing package, so the advice was to install what was
already installed.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import audio_gen  # noqa: E402


def _fake_nvidia(monkeypatch, tmp_path, layout=("cublas/bin", "cudnn/bin")):
    """A namespace package, like the real one: __path__ set, __file__ None."""
    for rel in layout:
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "cublas64_12.dll").write_bytes(b"\0")
    mod = types.ModuleType("nvidia")
    mod.__path__ = [str(tmp_path)]        # namespace packages have this
    mod.__file__ = None                   # ...and this is None, which broke it
    monkeypatch.setitem(sys.modules, "nvidia", mod)

    registered = []
    monkeypatch.setattr(audio_gen, "_is_windows", lambda: True)
    monkeypatch.setattr(audio_gen, "_register_dll_dir", registered.append)
    return registered


def test_a_namespace_package_still_gets_registered(monkeypatch, tmp_path):
    """The whole bug in one assertion: __file__ is None, __path__ is not."""
    registered = _fake_nvidia(monkeypatch, tmp_path)
    added = audio_gen._add_nvidia_dll_dirs()
    assert len(added) == 2
    assert any("cublas" in p for p in registered)
    assert any("cudnn" in p for p in registered)


def test_a_nested_arch_directory_is_found(monkeypatch, tmp_path):
    """Some Windows wheels ship <lib>/bin/x64 instead of <lib>/bin."""
    registered = _fake_nvidia(monkeypatch, tmp_path, layout=("cublas/bin/x64",))
    audio_gen._add_nvidia_dll_dirs()
    assert registered and registered[0].endswith("x64")


def test_a_bin_directory_with_no_dlls_is_skipped(monkeypatch, tmp_path):
    (tmp_path / "cublas" / "bin").mkdir(parents=True)      # empty
    mod = types.ModuleType("nvidia")
    mod.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "nvidia", mod)
    monkeypatch.setattr(audio_gen, "_is_windows", lambda: True)
    monkeypatch.setattr(audio_gen, "_register_dll_dir", lambda p: None)
    assert audio_gen._add_nvidia_dll_dirs() == []


def test_no_nvidia_package_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setattr(audio_gen, "_is_windows", lambda: True)
    monkeypatch.setitem(sys.modules, "nvidia", None)   # import raises
    assert audio_gen._add_nvidia_dll_dirs() == []
    assert "nvidia-cublas-cu12" in capsys.readouterr().out


def test_the_platform_check_is_injectable():
    """These tests flip _is_windows() rather than os.name, and that is not a
    style choice: os.name is the STDLIB os, so patching it also changes what
    pathlib.Path() constructs, and pytest's own tmp-dir cleanup then dies on
    "cannot instantiate 'WindowsPath' on your system" — which is exactly what
    happened on the first attempt at this file."""
    assert callable(audio_gen._is_windows)
    assert callable(audio_gen._register_dll_dir)


def test_linux_is_a_no_op(monkeypatch):
    monkeypatch.setattr(audio_gen, "_is_windows", lambda: False)
    assert audio_gen._add_nvidia_dll_dirs() == []


def test_the_failure_is_never_silent_again():
    """The old code was `except Exception: pass` around the whole body, so a
    box with the DLLs on disk looked identical to one without them."""
    import inspect
    src = inspect.getsource(audio_gen._add_nvidia_dll_dirs)
    assert "except Exception:\n        pass" not in src
    assert "__path__" in src
    assert "__file__" not in src.split('"""')[2], \
        "__file__ is None on a namespace package — never read it here"


# ── Why the previous fix registered correctly and still failed ───────────────

def test_registration_also_updates_PATH(monkeypatch, tmp_path):
    """os.add_dll_directory alone does not help here.

    It only affects DLLs loaded via LoadLibraryEx with LOAD_LIBRARY_SEARCH_*
    flags. cublas64_12.dll is an IMPLICIT dependency of ctranslate2's own DLL,
    and Windows resolves those through the standard search order — which
    consults PATH and never consults the add_dll_directory list.

    The live signature: every diagnostic stayed silent (directories registered
    fine), WhisperModel(device="cuda") constructed and printed "CUDA / float16
    (GPU mode)", then the first transcribe failed on the missing DLL.
    """
    import audio_gen

    calls = []
    monkeypatch.setattr(audio_gen.os, "add_dll_directory",
                        lambda p: calls.append(p), raising=False)
    monkeypatch.setenv("PATH", "C:\\existing")

    audio_gen._register_dll_dir(str(tmp_path))

    assert calls == [str(tmp_path)], "add_dll_directory must still be called"
    assert audio_gen.os.environ["PATH"].startswith(str(tmp_path))
    assert "C:\\existing" in audio_gen.os.environ["PATH"]


def test_PATH_is_not_duplicated_on_repeat_registration(monkeypatch, tmp_path):
    """_whisper() calls this on every model construction; PATH must not grow
    without bound across a long-running dashboard process."""
    import audio_gen

    monkeypatch.setattr(audio_gen.os, "add_dll_directory", lambda p: None,
                        raising=False)
    monkeypatch.setenv("PATH", "C:\\existing")

    for _ in range(5):
        audio_gen._register_dll_dir(str(tmp_path))

    assert audio_gen.os.environ["PATH"].count(str(tmp_path)) == 1


def test_a_missing_dll_is_named_rather_than_implied(monkeypatch, tmp_path, capsys):
    """Registering a directory and failing anyway is what cost weeks here: the
    fix reported success while the loader still could not find the file, and
    nothing distinguished 'wrong directory' from 'not installed'."""
    import audio_gen

    fake_lib = tmp_path / "torch" / "lib"
    fake_lib.mkdir(parents=True)
    (fake_lib / "something_else.dll").write_bytes(b"")

    monkeypatch.setattr(audio_gen, "_is_windows", lambda: True)
    monkeypatch.setattr(audio_gen, "_register_dll_dir", lambda p: None)

    class _FakeTorch:
        __path__ = [str(tmp_path / "torch")]

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(sys.modules, "nvidia", None)

    audio_gen._add_nvidia_dll_dirs()
    out = capsys.readouterr().out
    assert "cublas64_12.dll" in out
    assert "not present in any registered directory" in out


def test_a_present_dll_produces_no_warning(monkeypatch, tmp_path, capsys):
    import audio_gen

    fake_lib = tmp_path / "torch" / "lib"
    fake_lib.mkdir(parents=True)
    for dll in audio_gen._REQUIRED_CUDA_DLLS:
        (fake_lib / dll).write_bytes(b"")

    monkeypatch.setattr(audio_gen, "_is_windows", lambda: True)
    monkeypatch.setattr(audio_gen, "_register_dll_dir", lambda p: None)

    class _FakeTorch:
        __path__ = [str(tmp_path / "torch")]

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    added = audio_gen._add_nvidia_dll_dirs()
    assert str(fake_lib) in added
    assert "not present" not in capsys.readouterr().out


def test_torch_is_searched_as_a_second_source():
    """torch ships its own copy of the same CUDA runtime and is already
    installed here for Kokoro — a real second source, not a suggestion to go
    install something."""
    src = Path(audio_gen_path()).read_text(encoding="utf-8")
    assert "import torch" in src
    assert "torch/lib" in src or '"lib"' in src


def audio_gen_path():
    import audio_gen
    return audio_gen.__file__
