"""Tests for the HyperFrames motion-graphic video source. Pure-function +
mocked-subprocess level — no Node/HyperFrames needed to run these."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import hyperframes_client as hf


# ── Scene prompt ─────────────────────────────────────────────────────────────────

def test_scene_prompt_encodes_resolution_duration_accent_and_no_text_rule():
    p = hf._scene_prompt("rising market chart", {"accent_color": "#FFC53D",
                                                 "style_suffix": "muted teal"}, 8.0)
    assert 'data-width="1080"' in p and 'data-height="1920"' in p
    assert 'data-duration="8.0"' in p
    assert "#FFC53D" in p
    assert "NO large text" in p or "NO words" in p
    assert "NO external" in p          # offline/deterministic rule


# ── HTML extraction ──────────────────────────────────────────────────────────────

def test_extract_html_strips_code_fences():
    raw = "```html\n<!DOCTYPE html><html><body></body></html>\n```"
    out = hf._extract_html(raw)
    assert out.startswith("<!DOCTYPE html>") and out.endswith("</html>")


def test_extract_html_rejects_non_html():
    assert hf._extract_html("sorry, I can't do that") is None
    assert hf._extract_html("") is None


# ── Availability ─────────────────────────────────────────────────────────────────

def test_is_available_returns_bool_without_crashing():
    hf.is_available.cache_clear()
    with patch("hyperframes_client.subprocess.run", side_effect=FileNotFoundError("no npx")):
        assert hf.is_available() is False


def test_is_available_true_on_zero_exit():
    hf.is_available.cache_clear()
    class _R: returncode = 0
    with patch("hyperframes_client.subprocess.run", return_value=_R()):
        assert hf.is_available() is True
    hf.is_available.cache_clear()


# ── generate_clips ───────────────────────────────────────────────────────────────

def test_generate_clips_empty_when_unavailable():
    with patch("hyperframes_client.is_available", return_value=True), \
         patch("hyperframes_client._scene_to_html", return_value=None):
        # No HTML produced for any scene → no clips, no crash
        assert hf.generate_clips(["a", "b"], n=2) == []


def test_generate_clips_returns_empty_when_backend_missing():
    with patch("hyperframes_client.is_available", return_value=False):
        assert hf.generate_clips(["a"], n=1) == []


def test_generate_clips_skips_failed_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "TMP_DIR", tmp_path)
    calls = {"n": 0}

    def fake_render(html, work_dir, out_mp4):
        calls["n"] += 1
        return False  # every render fails

    with patch("hyperframes_client.is_available", return_value=True), \
         patch("hyperframes_client._scene_to_html", return_value="<html></html>"), \
         patch("hyperframes_client._render_html", side_effect=fake_render):
        result = hf.generate_clips(["q1", "q2"], n=2)

    assert result == []
    assert calls["n"] == 2          # tried both, skipped both — no exception


def test_generate_clips_collects_successful(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "TMP_DIR", tmp_path)

    def fake_render(html, work_dir, out_mp4):
        out_mp4.write_bytes(b"\x00" * (hf.MIN_BYTES + 1))
        return True

    with patch("hyperframes_client.is_available", return_value=True), \
         patch("hyperframes_client._scene_to_html", return_value="<html></html>"), \
         patch("hyperframes_client._render_html", side_effect=fake_render):
        result = hf.generate_clips(["q1", "q2"], n=2)

    assert len(result) == 2
    assert all(p.exists() for p in result)


# ── Niche-specific visual guide ───────────────────────────────────────────────────

def test_niche_visual_guide_finance_mentions_chart():
    guide = hf._niche_visual_guide({"name": "finance", "style_suffix": "data visualization"})
    assert "chart" in guide.lower() or "counter" in guide.lower() or "bar" in guide.lower()


def test_niche_visual_guide_motivation_mentions_particles():
    guide = hf._niche_visual_guide({"name": "motivation", "style_suffix": "sunrise"})
    assert "particle" in guide.lower() or "ray" in guide.lower() or "ring" in guide.lower()


def test_niche_visual_guide_generic_fallback():
    guide = hf._niche_visual_guide({})
    assert len(guide) > 20  # returns something, doesn't crash


# ── Hybrid prompt ─────────────────────────────────────────────────────────────────

def test_scene_prompt_hybrid_contains_placeholder_and_rules():
    p = hf._scene_prompt_hybrid("market rally", {"accent_color": "#FFC53D",
                                                  "name": "finance"}, 8.0)
    assert "__IMAGE_DATA_URI__" in p
    assert 'data-duration="8.0"' in p
    assert "kenburns" in p.lower()
    assert "NO external" in p or "NO CDN" in p


# ── generate_clips with image_paths (hybrid mode) ────────────────────────────────

def test_generate_clips_hybrid_mode_uses_image(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "TMP_DIR", tmp_path)
    fake_img = tmp_path / "scene_0.png"
    fake_img.write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)  # minimal fake PNG

    html_used = []

    def fake_to_html_hybrid(desc, cfg, dur, img_b64):
        html_used.append("hybrid")
        return "<html><body>hybrid</body></html>"

    def fake_render(html, work_dir, out_mp4):
        out_mp4.write_bytes(b"\x00" * (hf.MIN_BYTES + 1))
        return True

    with patch("hyperframes_client.is_available", return_value=True), \
         patch("hyperframes_client._scene_to_html_hybrid", side_effect=fake_to_html_hybrid), \
         patch("hyperframes_client._render_html", side_effect=fake_render):
        result = hf.generate_clips(["desc"], n=1, image_paths=[fake_img])

    assert len(result) == 1
    assert html_used == ["hybrid"]


def test_generate_clips_falls_back_to_css_when_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "TMP_DIR", tmp_path)
    html_used = []

    def fake_to_html(desc, cfg, dur):
        html_used.append("css")
        return "<html><body>css</body></html>"

    def fake_render(html, work_dir, out_mp4):
        out_mp4.write_bytes(b"\x00" * (hf.MIN_BYTES + 1))
        return True

    with patch("hyperframes_client.is_available", return_value=True), \
         patch("hyperframes_client._scene_to_html", side_effect=fake_to_html), \
         patch("hyperframes_client._render_html", side_effect=fake_render):
        result = hf.generate_clips(["desc"], n=1, image_paths=None)

    assert len(result) == 1
    assert html_used == ["css"]


# ── main.py per-niche source resolution ──────────────────────────────────────────

def test_video_source_resolution_precedence(monkeypatch):
    """env > niche video_source > default 'sd' — mirrors main.py's resolution."""
    def resolve(env, niche_cfg):
        return (env or niche_cfg.get("video_source") or "sd").strip().lower()

    # env wins over everything
    assert resolve("pexels", {"video_source": "hybrid"}) == "pexels"
    # niche config wins when no env
    assert resolve(None, {"video_source": "hybrid"}) == "hybrid"
    assert resolve(None, {"video_source": "hyperframes"}) == "hyperframes"
    # default when neither set
    assert resolve(None, {}) == "sd"
