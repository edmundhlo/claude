"""Tests for the YAMNet music classifier.

The TF Hub model is mocked so these tests run offline and don't need TF
installed. We verify the classifier correctly aggregates per-frame scores,
applies the threshold, and resamples non-16kHz input.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from time_signature import classifier
from time_signature.exceptions import ClassificationError


def _fake_yamnet(music_score: float, top_index: int = 0):
    """Build a (model, class_names, music_idx) triple matching _load_yamnet."""
    n_classes = 6
    class_names = [
        "Speech",
        "Music",
        "Singing",
        "Wind",
        "Rock music",
        "Silence",
    ]
    music_idx = np.array([1, 2, 4], dtype=np.int64)

    # Mean across frames should give music_score on the music classes combined.
    per_class_mean = np.full(n_classes, 0.01, dtype=np.float32)
    per_class_mean[music_idx] = music_score / len(music_idx)
    per_class_mean[top_index] = max(per_class_mean[top_index], 0.5)

    # 10 frames, all equal to per_class_mean, so the column-wise mean equals it.
    scores = np.tile(per_class_mean, (10, 1))

    fake_scores = mock.Mock()
    fake_scores.numpy.return_value = scores

    fake_model = mock.Mock(return_value=(fake_scores, mock.Mock(), mock.Mock()))
    return fake_model, class_names, music_idx


@pytest.fixture(autouse=True)
def _clear_yamnet_cache():
    classifier._load_yamnet.cache_clear()
    yield
    classifier._load_yamnet.cache_clear()


def test_high_music_confidence_is_classified_as_music():
    audio = np.random.randn(classifier.YAMNET_SAMPLE_RATE * 2).astype(np.float32)

    with mock.patch.object(classifier, "_load_yamnet", return_value=_fake_yamnet(0.9, top_index=1)):
        result = classifier.classify(audio, classifier.YAMNET_SAMPLE_RATE, threshold=0.5)

    assert result.is_music is True
    assert result.music_confidence >= 0.5
    assert result.top_label == "Music"


def test_low_music_confidence_is_not_music():
    audio = np.random.randn(classifier.YAMNET_SAMPLE_RATE * 2).astype(np.float32)

    with mock.patch.object(classifier, "_load_yamnet", return_value=_fake_yamnet(0.05, top_index=0)):
        result = classifier.classify(audio, classifier.YAMNET_SAMPLE_RATE, threshold=0.5)

    assert result.is_music is False
    assert result.top_label == "Speech"


def test_resamples_non_16k_input():
    sr_in = 44100
    audio = np.random.randn(sr_in * 2).astype(np.float32)

    captured = {}

    def fake_model(arr):
        captured["len"] = arr.shape[0]
        fake_scores = mock.Mock()
        fake_scores.numpy.return_value = np.full((5, 6), 0.01, dtype=np.float32)
        return fake_scores, mock.Mock(), mock.Mock()

    fake_loader = (fake_model, ["Speech", "Music", "Singing", "Wind", "Rock music", "Silence"], np.array([1, 2, 4]))

    with mock.patch.object(classifier, "_load_yamnet", return_value=fake_loader):
        classifier.classify(audio, sr_in)

    # The resampled audio should be roughly 2 seconds * 16k samples (within tolerance).
    assert abs(captured["len"] - 16000 * 2) < 200


def test_rejects_stereo_audio():
    audio = np.zeros((2, 16000), dtype=np.float32)
    with pytest.raises(ClassificationError):
        classifier.classify(audio, 16000)


def test_rejects_empty_audio():
    audio = np.zeros((0,), dtype=np.float32)
    with pytest.raises(ClassificationError):
        classifier.classify(audio, 16000)
