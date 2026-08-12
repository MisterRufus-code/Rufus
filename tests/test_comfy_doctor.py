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


# ── the Wan inventory: which variant, and what it costs per clip ─────────────

def test_ti2v_5b_is_not_filed_as_an_i2v_model():
    """"ti2v" CONTAINS "i2v". Without an exclusion the one variant that solves
    the 16GB-RAM problem gets reported as a 14B image-to-video model and the
    advice inverts — it would say "you have i2v but no t2v, go download more"
    at someone who already has the right file."""
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_ti2v_5B_fp16.safetensors"}})
    assert found["ti2v_5b"] == ["wan2.2_ti2v_5B_fp16.safetensors"]
    assert found["i2v_model"] == []


def test_t2v_and_i2v_models_are_told_apart():
    found = comfy_doctor._classify_wan({"UNETLoader": {
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}})
    assert found["t2v_model"] == ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"]
    assert found["i2v_model"] == ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"]


def test_a_lora_is_not_counted_as_a_model():
    """Otherwise "you have T2V models" is true while nothing can generate."""
    found = comfy_doctor._classify_wan({"LoraLoaderModelOnly": {
        "wan2.2_t2v_lightx2v_4steps_lora_v1_high_noise.safetensors"}})
    assert found["t2v_model"] == []
    assert found["lora_t2v"]


def test_having_i2v_but_not_t2v_says_they_are_separate_downloads():
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "different downloads" in tips


def test_the_missing_4_step_lora_leads_the_advice():
    """It is a 5x saving against maybe one minute from fixing the expert swap,
    so it must come before any RAM talk."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = comfy_doctor._wan_advice(found, have_export=False)
    assert "BIGGEST WIN" in tips[0]
    assert "4-step" in tips[0] or "4 steps" in tips[0]


def test_a_present_lora_is_pointed_at_export_time_not_an_env_var():
    """prepare() substitutes prompt/image/seed/dims only — steps and the LoRA
    toggle are frozen into the export. Telling someone to set an env var for
    them would send them chasing a variable that does not exist."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "LoraLoaderModelOnly": {"wan2.2_t2v_lightx2v_4steps_lora_v1_high.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "Export (API)" in tips
    assert "no env var" in tips.lower()


def test_ti2v_without_its_own_vae_is_flagged_before_the_run_not_after():
    """TI2V-5B does not use wan_2.1_vae. The wrong VAE is only rejected at
    submit time — after the whole stills phase has already run."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_ti2v_5B_fp16.safetensors"},
        "VAELoader": {"wan_2.1_vae.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "Wan 2.2 VAE" in tips
    assert "value_not_in_list" in tips


def test_a_missing_text_encoder_is_reported(monkeypatch):
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_t2v_high_noise_14B.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "umt5" in tips


def test_an_empty_comfyui_does_not_raise():
    found = comfy_doctor._classify_wan({})
    assert all(v == [] for v in found.values())
    assert comfy_doctor._wan_advice(found, have_export=False)
