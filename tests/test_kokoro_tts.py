"""Tests for Kokoro TTS — voice validation, synthesize, merge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestListVoices:
    def test_returns_list_of_strings(self):
        from src.tts.kokoro import list_voices
        voices = list_voices()
        assert isinstance(voices, list)
        assert len(voices) >= 1
        assert all(isinstance(v, str) for v in voices)

    def test_default_voice_in_list(self):
        from src.tts.kokoro import list_voices, DEFAULT_VOICE
        assert DEFAULT_VOICE in list_voices()

    def test_af_heart_in_list(self):
        from src.tts.kokoro import list_voices
        assert "af_heart" in list_voices()

    def test_am_adam_in_list(self):
        from src.tts.kokoro import list_voices
        assert "am_adam" in list_voices()


class TestVoiceValidation:
    def test_invalid_voice_falls_back_to_default(self, tmp_path):
        from src.tts import kokoro
        import src.tts.kokoro as kokoro_mod
        called_with_voice = []

        def fake_kokoro_create(text, voice, speed, lang):
            called_with_voice.append(voice)
            import numpy as np
            return np.zeros(100, dtype=np.float32), 24000

        mock_kokoro_instance = MagicMock()
        mock_kokoro_instance.create.side_effect = fake_kokoro_create
        mock_kokoro_class = MagicMock(return_value=mock_kokoro_instance)

        import soundfile as sf_mod
        with patch.object(kokoro_mod, "_get_kokoro", return_value=mock_kokoro_class), \
             patch.object(sf_mod, "write"):
            kokoro.synthesize("hello", tmp_path / "out.wav", voice="invalid_voice_xyz")

        assert called_with_voice[0] == kokoro.DEFAULT_VOICE

    def test_valid_voice_used_as_is(self, tmp_path):
        from src.tts import kokoro
        import src.tts.kokoro as kokoro_mod
        called_with_voice = []

        def fake_create(text, voice, speed, lang):
            called_with_voice.append(voice)
            import numpy as np
            return np.zeros(100, dtype=np.float32), 24000

        mock_instance = MagicMock()
        mock_instance.create.side_effect = fake_create
        mock_class = MagicMock(return_value=mock_instance)

        import soundfile as sf_mod
        with patch.object(kokoro_mod, "_get_kokoro", return_value=mock_class), \
             patch.object(sf_mod, "write"):
            kokoro.synthesize("hello", tmp_path / "out.wav", voice="af_bella")

        assert called_with_voice[0] == "af_bella"


class TestSynthesizeSections:
    def test_empty_sections_returns_empty_list(self, tmp_path):
        from src.tts.kokoro import synthesize_sections
        result = synthesize_sections([], tmp_path)
        assert result == []

    def test_blank_section_skipped(self, tmp_path):
        from src.tts import kokoro
        with patch.object(kokoro, "synthesize") as mock_synth:
            result = kokoro.synthesize_sections(
                [{"script": ""}, {"script": "   "}], tmp_path
            )
        assert mock_synth.call_count == 0
        assert result == []

    def test_sections_synthesized_in_order(self, tmp_path):
        from src.tts import kokoro
        calls = []
        def fake_synth(text, path, voice, speed):
            calls.append(text)
            path.write_bytes(b"x")
            return path

        with patch.object(kokoro, "synthesize", side_effect=fake_synth):
            kokoro.synthesize_sections(
                [{"script": "first"}, {"script": "second"}], tmp_path
            )
        assert calls == ["first", "second"]


class TestMergeAudioFiles:
    def test_merge_concatenates_arrays(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from src.tts.kokoro import merge_audio_files, SAMPLE_RATE

        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "merged.wav"
        sf.write(str(a), np.ones(100, dtype=np.float32), SAMPLE_RATE)
        sf.write(str(b), np.ones(100, dtype=np.float32), SAMPLE_RATE)

        merge_audio_files([a, b], out, gap_seconds=0.0)
        data, sr = sf.read(str(out))
        assert sr == SAMPLE_RATE
        assert len(data) == 200
