"""Tests for character_dataset.py — stills-only LoRA training-set generation
for a niche's recurring character."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import character_dataset as cd
import character_engine as ce


_CHAR = {
    "enabled": True,
    "name": "the Chronicler",
    "description": "a hooded figure in a weathered sepia cloak carrying a bronze lantern",
    "short_description": "a hooded figure with a bronze lantern",
}


@pytest.fixture
def niches(tmp_path, monkeypatch):
    cfg = {"niches": {"money_history": {
        "style_suffix": "flat 2D vector illustration, sepia palette",
        "character": dict(_CHAR),
    }}}
    p = tmp_path / "niches.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(ce, "NICHES_FILE", p)
    return p


# ── Variation matrix ─────────────────────────────────────────────────────────
# A LoRA learns what stays CONSTANT and treats what varies as incidental, so
# the set must not repeat the same setup — an earlier version had every list at
# length 4, which advanced them in lockstep and collapsed 24 images into 8
# distinct setups.

def test_variation_is_deterministic():
    assert cd._variation(7) == cd._variation(7)


def test_variation_gives_a_distinct_setup_per_image_at_realistic_sizes():
    for size in (24, 40, 60):
        seen = {cd._variation(i)[0] for i in range(size)}
        assert len(seen) == size, f"only {len(seen)} distinct setups across {size} images"


def test_variation_period_matches_the_declared_constant():
    """Guards the list-length choice: if someone makes the lists equal-length
    again, the period collapses and this fails."""
    seen = {cd._variation(i)[0] for i in range(cd._VARIATION_PERIOD)}
    assert len(seen) == cd._VARIATION_PERIOD
    assert cd._variation(0) == cd._variation(cd._VARIATION_PERIOD)


# ── Trigger word ─────────────────────────────────────────────────────────────

def test_default_trigger_slugifies_and_drops_the_article():
    assert cd._default_trigger({"name": "the Chronicler"}) == "chronicler_v1"
    assert cd._default_trigger({"name": "The Grinder"}) == "grinder_v1"


def test_default_trigger_survives_a_nameless_character():
    assert cd._default_trigger({}) == "character_v1"


def test_configured_lora_trigger_wins(niches, monkeypatch):
    cfg = dict(_CHAR, lora_trigger="my_custom_tok")
    niches.write_text(json.dumps({"niches": {"money_history": {"character": cfg}}}))
    _, caption = cd.build_prompt("money_history", 0)
    assert caption.startswith("my_custom_tok,")


# ── Prompt / caption split ───────────────────────────────────────────────────

def test_build_prompt_none_without_character(niches):
    niches.write_text(json.dumps({"niches": {"money_history": {}}}))
    assert cd.build_prompt("money_history", 0) is None


def test_render_prompt_uses_the_full_description_not_the_short_form(niches):
    """The dataset is the one place the long description belongs — unlike the
    per-beat prompts, there is no 2-4 sentence budget competing with it."""
    prompt, _ = cd.build_prompt("money_history", 0)
    assert "weathered sepia cloak carrying a bronze lantern" in prompt
    assert "flat 2D vector illustration" in prompt   # niche style applied


def test_caption_does_not_describe_the_character(niches):
    """The core LoRA captioning rule: what you caption is treated as variable,
    what you omit is absorbed into the trigger. Captioning the cloak would
    teach the model the cloak is interchangeable — the opposite of the point."""
    _, caption = cd.build_prompt("money_history", 0)
    low = caption.lower()
    for word in ("hooded", "cloak", "lantern", "sepia"):
        assert word not in low, f"caption leaks character detail: '{word}'"


def test_caption_starts_with_the_trigger_and_describes_the_variation(niches):
    _, caption = cd.build_prompt("money_history", 0)
    assert caption.startswith("chronicler_v1, ")
    assert cd._variation(0)[1] in caption


# ── build_dataset ────────────────────────────────────────────────────────────

def _fake_generate(prompt, out_path, *, width, height, seed):
    """Stand-in for image_gen.generate_image — writes the PNG AND the
    PROMPT:/SEED: sidecar the real one writes, so the caption-overwrite is
    genuinely exercised."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"PNG")
    out_path.with_suffix(".txt").write_text(f"PROMPT: {prompt}\nSEED: {seed}\n")
    return out_path


def test_build_dataset_writes_images_and_captions(niches, tmp_path, monkeypatch):
    import image_gen
    monkeypatch.setattr(image_gen, "generate_image", _fake_generate)

    out = tmp_path / "ds"
    written = cd.build_dataset("money_history", count=3, out_dir=out, seed_base=1)

    assert len(written) == 3
    assert [p.name for p in written] == [
        "chronicler_000.png", "chronicler_001.png", "chronicler_002.png"]
    for p in written:
        assert p.exists()
        assert p.with_suffix(".txt").exists()


def test_build_dataset_overwrites_the_metadata_sidecar_with_a_real_caption(
        niches, tmp_path, monkeypatch):
    """image_gen writes 'PROMPT: …/SEED: …' at exactly the path a LoRA trainer
    reads as the caption. Left alone, every training caption would be that
    metadata blob."""
    import image_gen
    monkeypatch.setattr(image_gen, "generate_image", _fake_generate)

    out = tmp_path / "ds"
    written = cd.build_dataset("money_history", count=1, out_dir=out, seed_base=1)

    caption = written[0].with_suffix(".txt").read_text()
    assert "PROMPT:" not in caption
    assert "SEED:" not in caption
    assert caption.startswith("chronicler_v1,")


def test_build_dataset_always_passes_an_explicit_seed(niches, tmp_path, monkeypatch):
    """Load-bearing: image_gen skips its perceptual-dedup pass only when given
    an explicit seed. Without one, a set of near-identical character images
    would be fought by the dedup meant for repeated video frames, and would
    also write those hashes into the shared freshness pool the video pipeline
    reads."""
    import image_gen
    seeds = []

    def capture(prompt, out_path, *, width, height, seed):
        seeds.append(seed)
        return _fake_generate(prompt, out_path, width=width, height=height, seed=seed)

    monkeypatch.setattr(image_gen, "generate_image", capture)
    cd.build_dataset("money_history", count=4, out_dir=tmp_path / "ds", seed_base=1)

    assert len(seeds) == 4
    assert all(s is not None for s in seeds)
    assert len(set(seeds)) == 4, "seeds must differ per image"


def test_build_dataset_is_reproducible_from_a_seed_base(niches, tmp_path, monkeypatch):
    import image_gen
    runs = []

    def capture(prompt, out_path, *, width, height, seed):
        runs.append(seed)
        return _fake_generate(prompt, out_path, width=width, height=height, seed=seed)

    monkeypatch.setattr(image_gen, "generate_image", capture)
    cd.build_dataset("money_history", count=3, out_dir=tmp_path / "a", seed_base=42)
    first = list(runs)
    runs.clear()
    cd.build_dataset("money_history", count=3, out_dir=tmp_path / "b", seed_base=42)
    assert runs == first


def test_build_dataset_continues_after_a_failed_image(niches, tmp_path, monkeypatch):
    """A ComfyUI hiccup partway through must not discard the images that
    already succeeded."""
    import image_gen
    calls = {"n": 0}

    def flaky(prompt, out_path, *, width, height, seed):
        calls["n"] += 1
        if calls["n"] == 2:
            return None
        return _fake_generate(prompt, out_path, width=width, height=height, seed=seed)

    monkeypatch.setattr(image_gen, "generate_image", flaky)
    written = cd.build_dataset("money_history", count=4, out_dir=tmp_path / "ds", seed_base=1)
    assert len(written) == 3          # 4 attempted, 1 failed, 3 kept


def test_build_dataset_start_index_continues_an_existing_set(niches, tmp_path, monkeypatch):
    import image_gen
    monkeypatch.setattr(image_gen, "generate_image", _fake_generate)
    written = cd.build_dataset("money_history", count=2, out_dir=tmp_path / "ds",
                               start_index=24, seed_base=1)
    assert [p.name for p in written] == ["chronicler_024.png", "chronicler_025.png"]


def test_build_dataset_writes_a_manifest(niches, tmp_path, monkeypatch):
    import image_gen
    monkeypatch.setattr(image_gen, "generate_image", _fake_generate)
    out = tmp_path / "ds"
    cd.build_dataset("money_history", count=2, out_dir=out, seed_base=1)

    lines = [json.loads(l) for l in (out / "_manifest.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert {"file", "index", "seed", "caption", "prompt"} <= set(lines[0])


def test_build_dataset_empty_for_niche_without_character(niches, tmp_path):
    niches.write_text(json.dumps({"niches": {"money_history": {}}}))
    with patch("image_gen.generate_image") as gen:
        assert cd.build_dataset("money_history", count=3, out_dir=tmp_path / "ds") == []
    gen.assert_not_called()
