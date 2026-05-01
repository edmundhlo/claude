"""Estimate the meter (beats per bar) of a music recording.

Two backends are exposed:

* ``"librosa"`` (default): tracks beats with librosa, then for each candidate
  meter k scans every possible downbeat phase and picks the one that maximises
  the gap between presumed downbeats and offbeats. The k whose best phase has
  the strongest gap wins. Lightweight, no model downloads, and robust to small
  beat-tracking phase errors. Cannot resolve compound meters such as 6/8 —
  they typically read as 3/4 or 4/4.

* ``"madmom"``: stub for a future implementation backed by madmom's RNN/DBN
  downbeat tracker. Raises ``BackendNotAvailableError`` for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exceptions import BackendNotAvailableError, MeterDetectionError

Backend = Literal["librosa", "madmom"]

# Candidate beats-per-bar values we score. 2 is included so a clear duple meter
# does not get incorrectly forced into 4.
_METER_CANDIDATES = (2, 3, 4)


@dataclass(frozen=True)
class MeterResult:
    beats_per_bar: int
    time_signature: str
    tempo_bpm: float
    backend: str
    candidate_scores: dict[int, float]


def estimate_meter(
    audio: np.ndarray,
    sr: int,
    backend: Backend = "librosa",
) -> MeterResult:
    if backend == "librosa":
        return _estimate_librosa(audio, sr)
    if backend == "madmom":
        raise BackendNotAvailableError(
            "the madmom backend is not implemented yet; use backend='librosa'"
        )
    raise ValueError(f"unknown backend: {backend!r}")


def _estimate_librosa(audio: np.ndarray, sr: int) -> MeterResult:
    try:
        import librosa
    except ImportError as e:
        raise MeterDetectionError("librosa is required for the librosa backend") from e

    if audio.ndim != 1:
        raise MeterDetectionError(f"expected mono audio, got shape {audio.shape}")
    if audio.size < sr:
        raise MeterDetectionError("audio is too short to estimate a meter (need >= 1s)")

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    if len(beats) < max(_METER_CANDIDATES) * 2:
        raise MeterDetectionError(
            f"only {len(beats)} beats detected; need at least "
            f"{max(_METER_CANDIDATES) * 2} to estimate a meter"
        )

    # Peak onset value within each beat window. ``max`` is robust to short
    # transients that ``mean`` would average away across a long beat.
    beat_strengths = librosa.util.sync(
        onset_env, beats, aggregate=np.max
    ).flatten().astype(np.float64)

    candidate_scores = _score_meters(beat_strengths)
    if not candidate_scores:
        raise MeterDetectionError("no meter candidates could be scored")

    best = max(candidate_scores, key=candidate_scores.get)

    return MeterResult(
        beats_per_bar=best,
        time_signature=f"{best}/4",
        tempo_bpm=tempo,
        backend="librosa",
        candidate_scores=candidate_scores,
    )


def _score_meters(beat_strengths: np.ndarray) -> dict[int, float]:
    """For each candidate k, find the downbeat phase that best separates
    'downbeat' beats (every k-th) from offbeats, and score it by the
    normalised mean gap. Phase-scanning makes the score insensitive to where
    the recording starts relative to the first downbeat."""
    n = len(beat_strengths)
    overall = beat_strengths.mean()
    spread = beat_strengths.std() + 1e-9

    scores: dict[int, float] = {}
    for k in _METER_CANDIDATES:
        if n < k * 2:
            continue
        best_phase_gap = -np.inf
        for phase in range(k):
            mask = np.zeros(n, dtype=bool)
            mask[phase::k] = True
            downbeats = beat_strengths[mask]
            offbeats = beat_strengths[~mask]
            if downbeats.size == 0 or offbeats.size == 0:
                continue
            gap = (downbeats.mean() - offbeats.mean()) / spread
            if gap > best_phase_gap:
                best_phase_gap = gap
        scores[k] = float(best_phase_gap)
    return scores
