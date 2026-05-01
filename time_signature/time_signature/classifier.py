"""Music-vs-non-music classification using Google's YAMNet (AudioSet)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .exceptions import ClassificationError

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000

# AudioSet class display names that indicate the audio is music.
# Anything else (Speech, environmental sounds, silence) is non-music.
_MUSIC_KEYWORDS = ("music", "singing", "song", "instrument")


@dataclass(frozen=True)
class ClassificationResult:
    is_music: bool
    music_confidence: float
    top_label: str


@lru_cache(maxsize=1)
def _load_yamnet():
    try:
        import tensorflow_hub as hub
    except ImportError as e:
        raise ClassificationError(
            "tensorflow + tensorflow-hub are required for YAMNet classification"
        ) from e

    model = hub.load(YAMNET_HANDLE)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]

    music_idx = np.array(
        [
            i
            for i, name in enumerate(class_names)
            if any(kw in name.lower() for kw in _MUSIC_KEYWORDS)
        ],
        dtype=np.int64,
    )
    return model, class_names, music_idx


def classify(audio: np.ndarray, sr: int, threshold: float = 0.5) -> ClassificationResult:
    """Classify whether ``audio`` is music using YAMNet.

    ``audio`` must be a 1-D float array. It is resampled to 16 kHz internally
    if needed. ``threshold`` is applied to the summed score of music-related
    AudioSet classes averaged across frames.
    """
    if audio.ndim != 1:
        raise ClassificationError(f"expected mono audio, got shape {audio.shape}")
    if audio.size == 0:
        raise ClassificationError("audio is empty")

    if sr != YAMNET_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=YAMNET_SAMPLE_RATE)
    else:
        audio = audio.astype(np.float32)

    model, class_names, music_idx = _load_yamnet()
    scores, _embeddings, _spectrogram = model(audio)
    mean_scores = scores.numpy().mean(axis=0)  # shape: (521,)

    music_confidence = float(mean_scores[music_idx].sum())
    music_confidence = min(music_confidence, 1.0)
    top_label = class_names[int(np.argmax(mean_scores))]

    return ClassificationResult(
        is_music=music_confidence >= threshold,
        music_confidence=music_confidence,
        top_label=top_label,
    )
