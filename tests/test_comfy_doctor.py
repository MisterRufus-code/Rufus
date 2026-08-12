"""The doctor answers "I downloaded the models" — did they land where ComfyUI looks?

Every ComfyUI-backed engine here is inert until its API export exists, which is
right (comfy_template.py: the graph is never hand-wired). The gap it leaves is
that "off" reads identically for four different causes — server down, nodes
missing, weights in the wrong folder, export not done — and telling them apart
used to mean starting a full run and reading the log.

Tests run with no ComfyUI and no network, per AGENTS.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_doctor  # noqa: E402


def test_an_unreachable_server_stops_before_guessing(monkeypatch, capsys):
    """Every later answer is derived from /object_info. Without it the report
    would be confident and wrong, which is worse than no report."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: False)
    assert comfy_doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "not reachable" in out
    assert "Start ComfyUI first" in out


def test_the_wrong_folder_is_named_as_the_cause(monkeypatch, capsys):
    """The single most common reason a downloaded model "doesn't work" is that
    it is not under ComfyUI's own models/ tree. Say that, don't say "missing"."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files",
                        lambda host: {"UNETLoader": {"flux1-dev.safetensors"}})
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "wrong folder" in out
    assert "loader dropdown" in out


def test_a_present_model_is_listed_with_its_loader(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"},
    })
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "wan2.2_t2v_high_noise_14B_fp8.safetensors" in out
    assert "[UNETLoader]" in out
    assert "umt5_xxl_fp8.safetensors" in out


def test_a_missing_export_gets_the_exact_click_path(monkeypatch, capsys):
    """The export is the one step that is never automatic, so the instruction
    has to be complete enough to follow without leaving the terminal."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files",
                        lambda host: {"UNETLoader": {"wan2.2_t2v.safetensors"}})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "no export at config/wan_t2v_api.json" in out
    assert "RUFUS_PROMPT" in out
    assert "Export (API)" in out


def test_an_unknown_engine_name_is_reported_not_ignored(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    comfy_doctor.main(["wan_t2v", "nonsense"])
    out = capsys.readouterr().out
    assert "unknown engine(s): nonsense" in out


def test_no_argument_reports_every_engine(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    comfy_doctor.main([])
    out = capsys.readouterr().out
    for name in comfy_doctor.ENGINES:
        assert name in out


def test_every_template_path_it_names_matches_the_engine_that_reads_it():
    """A doctor that points at the wrong filename sends the owner to export a
    file nothing will ever load."""
    import wan_t2v_client
    assert comfy_doctor.ENGINES["wan_t2v"][1] in str(wan_t2v_client._template_path())


def test_it_is_read_only():
    """This runs when something is already broken. It must not write, delete or
    start anything."""
    src = Path(comfy_doctor.__file__).read_text(encoding="utf-8")
    for forbidden in ("write_text(", "unlink(", "mkdir(", "requests.post",
                      "subprocess"):
        assert forbidden not in src, forbidden
