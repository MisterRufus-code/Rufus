"""Finding the right Wan weights among dozens of near-identical names.

These files get republished under new names constantly, come in halves that
must match, and land in different ComfyUI folders depending on what they are.
Getting any of that wrong produces a file on disk that ComfyUI cannot see —
which looks exactly like never having downloaded it.

No network: the repo listing is the boundary, so tests pass it directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import fetch_wan_assets as fw  # noqa: E402


_LISTING = [
    "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "split_files/vae/wan2.2_vae.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
    "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    "README.md",
]


# ── the speed LoRAs ──────────────────────────────────────────────────────────

def test_the_t2v_pair_is_found_whole():
    pair = fw.find_pair(_LISTING, i2v=False)
    assert set(pair) == {"high", "low"}
    assert "t2v" in pair["high"] and "high_noise" in pair["high"]
    assert "t2v" in pair["low"] and "low_noise" in pair["low"]


def test_the_i2v_pair_is_a_different_pair():
    pair = fw.find_pair(_LISTING, i2v=True)
    assert "i2v" in pair["high"]
    assert "t2v" not in pair["high"]


def test_a_plain_model_is_never_mistaken_for_a_lora():
    """Without the distillation marker check, the 14B model files would be
    picked as the "pair" and 28GB would land in models/loras."""
    pair = fw.find_pair(
        ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
         "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"], i2v=False)
    assert pair == {}


def test_a_lone_half_is_not_a_pair():
    """The two experts split the sampling schedule, so one alone does nothing.
    Reporting it as found would send someone to export a broken workflow."""
    pair = fw.find_pair(
        ["wan2.2_t2v_lightx2v_4steps_high_noise.safetensors"], i2v=False)
    assert len(pair) == 1


def test_ti2v_files_are_never_counted_as_the_i2v_pair():
    """"ti2v" CONTAINS "i2v" — the same trap comfy_doctor hit."""
    assert fw.find_pair(
        ["wan2.2_ti2v_5B_lightx2v_4steps_high_noise.safetensors"],
        i2v=True) == {}


# ── TI2V-5B and the VAE it cannot run without ────────────────────────────────

def test_ti2v_takes_the_2_2_vae_and_not_the_2_1():
    """TI2V-5B does not use wan_2.1_vae. Both are in the listing and only one
    is correct; the wrong one is rejected at submit time, after the whole
    stills phase has been paid for."""
    found = fw.find_ti2v(_LISTING)
    assert found["model"].endswith("wan2.2_ti2v_5B_fp16.safetensors")
    assert found["vae"].endswith("wan2.2_vae.safetensors")
    assert "2.1" not in found["vae"]


def test_the_ti2v_model_is_not_confused_with_a_vae():
    found = fw.find_ti2v(["split_files/vae/wan2.2_ti2v_vae.safetensors"])
    assert "model" not in found


def test_a_repo_with_the_model_but_no_vae_reports_only_the_model():
    """The caller refuses a partial fetch — half of this pair is not usable,
    and delivering it looks like success."""
    found = fw.find_ti2v(["diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"])
    assert "model" in found and "vae" not in found


def test_non_safetensors_files_are_ignored():
    assert fw.find_ti2v(["README.md", "wan2.2_ti2v_5B.txt"]) == {}


# ── argument handling ────────────────────────────────────────────────────────

def test_ti2v_and_i2v_together_are_refused(capsys):
    assert fw.main(["--ti2v", "--i2v"]) == 2
    assert "pick one" in capsys.readouterr().out


def test_a_missing_destination_names_the_right_folder_per_mode(capsys):
    """models/loras and models/diffusion_models are different places, and a
    file in the wrong one is invisible to ComfyUI."""
    fw.main(["--ti2v"])
    assert "models/diffusion_models" in capsys.readouterr().out
    fw.main([])
    assert "models/loras" in capsys.readouterr().out


def test_it_refuses_to_create_the_destination(tmp_path):
    """A folder ComfyUI has never heard of is not a destination — writing
    600MB into one produces the exact symptom the download was meant to fix."""
    missing = tmp_path / "nope"
    assert fw._resolve_dest(str(missing)) is None
    assert not missing.exists()


def test_an_existing_destination_resolves(tmp_path):
    assert fw._resolve_dest(str(tmp_path)) == tmp_path


def test_the_env_var_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_COMFY_LORAS", str(tmp_path))
    assert fw._resolve_dest(None) == tmp_path
