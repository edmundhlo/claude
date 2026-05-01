"""End-to-end tests for the analyzer orchestrator.

Mocks download, audio loading, classification, and meter estimation so the
test exercises the control flow without touching the network or any models.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from time_signature import analyzer
from time_signature.classifier import ClassificationResult
from time_signature.meter import MeterResult


@pytest.fixture
def fake_audio():
    return np.zeros(22050 * 4, dtype=np.float32), 22050


def _patch_pipeline(monkeypatch, *, fake_audio, classification, meter):
    monkeypatch.setattr(
        analyzer, "download_audio",
        lambda url, out_dir: Path(out_dir) / "abc.wav",
    )
    monkeypatch.setattr(
        analyzer, "_load_audio",
        lambda path, *, start_time, end_time, max_duration: fake_audio,
    )
    monkeypatch.setattr(analyzer, "classify", mock.Mock(return_value=classification))
    monkeypatch.setattr(analyzer, "estimate_meter", mock.Mock(return_value=meter))


def test_music_url_returns_time_signature(monkeypatch, fake_audio):
    classification = ClassificationResult(
        is_music=True, music_confidence=0.85, top_label="Music"
    )
    meter = MeterResult(
        beats_per_bar=4, time_signature="4/4", tempo_bpm=120.0,
        backend="librosa", candidate_scores={2: 0.4, 3: 0.2, 4: 0.9},
    )
    _patch_pipeline(monkeypatch, fake_audio=fake_audio,
                    classification=classification, meter=meter)

    result = analyzer.analyze_url("https://www.youtube.com/watch?v=abc")

    assert result.is_music is True
    assert result.time_signature == "4/4"
    assert result.beats_per_bar == 4
    assert result.tempo_bpm == 120.0
    assert result.backend == "librosa"
    analyzer.estimate_meter.assert_called_once()


def test_non_music_skips_meter_detection(monkeypatch, fake_audio):
    classification = ClassificationResult(
        is_music=False, music_confidence=0.05, top_label="Speech"
    )
    # Meter mock would explode if called.
    meter_mock = mock.Mock(side_effect=AssertionError("must not run"))

    monkeypatch.setattr(
        analyzer, "download_audio",
        lambda url, out_dir: Path(out_dir) / "abc.wav",
    )
    monkeypatch.setattr(
        analyzer, "_load_audio",
        lambda path, *, start_time, end_time, max_duration: fake_audio,
    )
    monkeypatch.setattr(analyzer, "classify", mock.Mock(return_value=classification))
    monkeypatch.setattr(analyzer, "estimate_meter", meter_mock)

    result = analyzer.analyze_url("https://www.youtube.com/watch?v=xyz")

    assert result.is_music is False
    assert result.time_signature is None
    assert result.beats_per_bar is None
    assert result.tempo_bpm is None
    assert result.top_label == "Speech"
    meter_mock.assert_not_called()


def test_threshold_propagates_to_classifier(monkeypatch, fake_audio):
    classification = ClassificationResult(
        is_music=False, music_confidence=0.4, top_label="Speech"
    )
    classify_mock = mock.Mock(return_value=classification)

    monkeypatch.setattr(
        analyzer, "download_audio",
        lambda url, out_dir: Path(out_dir) / "abc.wav",
    )
    monkeypatch.setattr(
        analyzer, "_load_audio",
        lambda path, *, start_time, end_time, max_duration: fake_audio,
    )
    monkeypatch.setattr(analyzer, "classify", classify_mock)

    analyzer.analyze_url("https://example.com/v", music_threshold=0.7)

    _args, kwargs = classify_mock.call_args
    assert kwargs["threshold"] == 0.7


def test_analyze_file_does_not_download(monkeypatch, tmp_path, fake_audio):
    download_mock = mock.Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(analyzer, "download_audio", download_mock)
    monkeypatch.setattr(
        analyzer, "_load_audio",
        lambda path, *, start_time, end_time, max_duration: fake_audio,
    )
    classification = ClassificationResult(
        is_music=True, music_confidence=0.9, top_label="Music"
    )
    meter = MeterResult(
        beats_per_bar=3, time_signature="3/4", tempo_bpm=90.0,
        backend="librosa", candidate_scores={2: 0.2, 3: 0.9, 4: 0.4},
    )
    monkeypatch.setattr(analyzer, "classify", mock.Mock(return_value=classification))
    monkeypatch.setattr(analyzer, "estimate_meter", mock.Mock(return_value=meter))

    fake_file = tmp_path / "song.wav"
    fake_file.write_bytes(b"")
    result = analyzer.analyze_file(fake_file)

    download_mock.assert_not_called()
    assert result.time_signature == "3/4"
    assert result.url == str(fake_file)


def test_start_and_end_time_propagate_to_loader(monkeypatch, fake_audio, tmp_path):
    """start_time + end_time should be forwarded as offset/duration to librosa.load."""
    captured = {}

    def fake_librosa_load(path, *, sr, mono, offset, duration):
        captured["offset"] = offset
        captured["duration"] = duration
        return fake_audio

    fake_librosa = mock.MagicMock()
    fake_librosa.load = fake_librosa_load
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)

    monkeypatch.setattr(
        analyzer, "classify",
        mock.Mock(return_value=ClassificationResult(False, 0.0, "Speech")),
    )

    fake_file = tmp_path / "song.wav"
    fake_file.write_bytes(b"")
    analyzer.analyze_file(fake_file, start_time=30.0, end_time=90.0)

    assert captured["offset"] == 30.0
    assert captured["duration"] == 60.0


def test_end_time_none_uses_max_duration(monkeypatch, fake_audio, tmp_path):
    captured = {}

    def fake_librosa_load(path, *, sr, mono, offset, duration):
        captured["offset"] = offset
        captured["duration"] = duration
        return fake_audio

    fake_librosa = mock.MagicMock()
    fake_librosa.load = fake_librosa_load
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)
    monkeypatch.setattr(
        analyzer, "classify",
        mock.Mock(return_value=ClassificationResult(False, 0.0, "Speech")),
    )

    fake_file = tmp_path / "song.wav"
    fake_file.write_bytes(b"")
    analyzer.analyze_file(fake_file, start_time=10.0, max_duration=45.0)

    assert captured["offset"] == 10.0
    assert captured["duration"] == 45.0


def test_negative_start_time_rejected():
    with pytest.raises(ValueError, match="start_time"):
        analyzer.analyze_file("nope.wav", start_time=-1.0)


def test_end_before_start_rejected():
    with pytest.raises(ValueError, match="end_time"):
        analyzer.analyze_file("nope.wav", start_time=30.0, end_time=10.0)


def test_to_dict_serialises_result():
    result = analyzer.AnalysisResult(
        url="https://x", is_music=True, music_confidence=0.9,
        top_label="Music", time_signature="4/4", beats_per_bar=4,
        tempo_bpm=120.0, backend="librosa",
    )
    assert result.to_dict() == {
        "url": "https://x",
        "is_music": True,
        "music_confidence": 0.9,
        "top_label": "Music",
        "time_signature": "4/4",
        "beats_per_bar": 4,
        "tempo_bpm": 120.0,
        "backend": "librosa",
    }
