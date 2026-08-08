"""Tests for video_critique.py — free, local (Ollama) video quality critique.
Pure-function + mocked-HTTP level, no real Ollama server or ffmpeg needed."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import video_critique as vc


# ── Availability / model-readiness preflight ─────────────────────────────────

def test_is_available_false_on_connection_error():
    with patch("video_critique.requests.get", side_effect=OSError("refused")):
        assert vc.is_available() is False


def test_is_available_true_on_200():
    resp = MagicMock(status_code=200)
    with patch("video_critique.requests.get", return_value=resp):
        assert vc.is_available() is True


def test_pulled_models_empty_on_error():
    with patch("video_critique.requests.get", side_effect=OSError("refused")):
        assert vc._pulled_models() == []


def test_model_ready_matches_exact_name():
    with patch.object(vc, "_pulled_models", return_value=["llama3.2-vision:latest"]):
        assert vc._model_ready("llama3.2-vision") is True
        assert vc._model_ready("llava") is False


def test_model_ready_false_when_nothing_pulled():
    with patch.object(vc, "_pulled_models", return_value=[]):
        assert vc._model_ready("llama3.2-vision") is False


# ── critique_video: graceful degradation ─────────────────────────────────────

def test_critique_video_none_when_file_missing(tmp_path):
    assert vc.critique_video(tmp_path / "nope.mp4") is None


def test_critique_video_none_when_ollama_unreachable(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with patch.object(vc, "is_available", return_value=False):
        assert vc.critique_video(video) is None


def test_critique_video_none_when_model_not_pulled(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=False):
        assert vc.critique_video(video) is None


def test_critique_video_none_when_no_frames_extracted(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[]):
        assert vc.critique_video(video) is None


def test_critique_video_none_on_request_failure(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"fake jpg")
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[frame]), \
         patch("video_critique.requests.post", side_effect=OSError("timeout")):
        assert vc.critique_video(video) is None


def test_critique_video_none_on_empty_response(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"fake jpg")
    resp = MagicMock()
    resp.json.return_value = {"response": "   "}
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[frame]), \
         patch("video_critique.requests.post", return_value=resp):
        assert vc.critique_video(video) is None


def test_critique_video_returns_report_on_success(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"fake jpg")
    resp = MagicMock()
    resp.json.return_value = {"response": "HOOK (first frame): strong. ..."}
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[frame]), \
         patch("video_critique.requests.post", return_value=resp) as post:
        result = vc.critique_video(video, script="the script", model="llava")

    assert result == {"report": "HOOK (first frame): strong. ...",
                      "model": "llava", "frame_count": 1}
    sent = post.call_args.kwargs["json"]
    assert sent["model"] == "llava"
    assert sent["stream"] is False
    assert len(sent["images"]) == 1


def test_critique_video_cleans_up_temp_frames(tmp_path):
    """Frames are temp files extracted just for this call — they must not
    accumulate on disk run after run."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"fake jpg")
    resp = MagicMock()
    resp.json.return_value = {"response": "some report"}
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[frame]), \
         patch("video_critique.requests.post", return_value=resp):
        vc.critique_video(video)
    assert not frame.exists()


def test_critique_video_cleans_up_temp_frames_even_on_failure(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"fake jpg")
    with patch.object(vc, "is_available", return_value=True), \
         patch.object(vc, "_model_ready", return_value=True), \
         patch.object(vc, "extract_frames", return_value=[frame]), \
         patch("video_critique.requests.post", side_effect=OSError("boom")):
        vc.critique_video(video)
    assert not frame.exists()


# ── Prompt building ───────────────────────────────────────────────────────────

def test_build_prompt_includes_all_report_sections():
    prompt = vc._build_prompt("")
    for section in vc.REPORT_SECTIONS:
        assert section in prompt


def test_build_prompt_includes_script_when_given():
    prompt = vc._build_prompt("Rome debased the denarius.")
    assert "Rome debased the denarius." in prompt


def test_build_prompt_omits_script_block_when_empty():
    prompt = vc._build_prompt("")
    assert "NARRATION SCRIPT" not in prompt
