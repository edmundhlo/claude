"""Estimate the meter (beats per bar) of a music recording.

Two backends are exposed:

* ``"librosa"`` (default): tracks beats with librosa, then for each candidate
  meter k scans every possible downbeat phase and picks the one that maximises
  the gap between presumed downbeats and offbeats. The k whose best phase has
  the strongest gap wins. Lightweight, no model downloads, and robust to small
  beat-tracking phase errors. Cannot resolve compound meters such as 6/8 —
  they typically read as 3/4 or 4/4.

* ``"madmom"``: madmom's pre-trained RNN downbeat activation network feeding
  a Dynamic Bayesian Network downbeat tracker. More accurate on real music
  than the librosa heuristic. Requires the ``madmom`` package; raises
  :class:`BackendNotAvailableError` if not importable. madmom 0.16.1 has
  several Python 3.10+ / NumPy 2.x incompatibilities that are worked around
  at runtime — see :func:`_apply_madmom_compat_shims`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

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
        return _estimate_madmom(audio, sr)
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


# --------------------------------------------------------------- madmom backend
#
# madmom 0.16.1 (2018) is the most recent release and is unmaintained. It does
# not officially support Python 3.10+ or NumPy 1.24+. Two issues bite us:
#
#   1. ``from collections import MutableSequence`` (and friends) — these moved
#      to ``collections.abc`` in Python 3.3 and were removed from
#      ``collections`` in Python 3.10.
#
#   2. Use of removed NumPy aliases ``np.float`` / ``np.int`` / ``np.bool`` /
#      ``np.object``, plus a ``np.asarray(...)[:, 1]`` pattern in the DBN
#      downbeat tracker that NumPy 2.x rejects when the inner sequences have
#      mismatched shapes.
#
# Rather than patch madmom on disk (brittle — undone by ``pip install --upgrade``)
# we apply the fixes at runtime, scoped to the call. The shim functions below
# are idempotent and only run when the madmom backend is invoked.

# Compatibility tables — single source of truth so the shim can also report
# what it patched if someone needs to debug.
_COLLECTIONS_ABC_NAMES = (
    "MutableSequence",
    "MutableMapping",
    "Iterable",
    "Callable",
    "Mapping",
    "Hashable",
    "Sequence",
)
_NUMPY_ALIASES: tuple[tuple[str, type], ...] = (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("object", object),
    ("complex", complex),
)


def _apply_madmom_compat_shims() -> None:
    """Re-inject names that Python 3.10+ / NumPy 2.x removed but madmom still uses."""
    import collections
    import collections.abc

    for name in _COLLECTIONS_ABC_NAMES:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    for alias, target in _NUMPY_ALIASES:
        if not hasattr(np, alias):
            setattr(np, alias, target)


@contextmanager
def _allow_inhomogeneous_asarray() -> Iterator[None]:
    """Patch :func:`numpy.asarray` to fall back to ``dtype=object`` for sequences
    whose inner shapes don't match. Required for madmom's DBN tracker, which
    builds an array of ``(path, log_likelihood)`` tuples where ``path`` has
    variable length per HMM. NumPy 2.x rejects this without an explicit dtype.
    """
    original = np.asarray

    def _patched(a, *args, **kwargs):
        try:
            return original(a, *args, **kwargs)
        except ValueError as e:
            if "inhomogeneous" in str(e):
                return original(a, dtype=object)
            raise

    np.asarray = _patched
    try:
        yield
    finally:
        np.asarray = original


# madmom's pre-trained downbeat RNN was trained on this rate.
_MADMOM_SAMPLE_RATE = 44100
_MADMOM_FPS = 100  # frames per second the RNN/DBN operate on


def _estimate_madmom(audio: np.ndarray, sr: int) -> MeterResult:
    """Estimate meter via madmom's RNN downbeat activations + DBN tracker."""
    if audio.ndim != 1:
        raise MeterDetectionError(f"expected mono audio, got shape {audio.shape}")
    if audio.size < sr:
        raise MeterDetectionError("audio is too short to estimate a meter (need >= 1s)")

    _apply_madmom_compat_shims()

    try:
        from madmom.audio.signal import Signal
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )
    except ImportError as e:
        raise BackendNotAvailableError(
            "the madmom backend requires `pip install madmom` (and a build env "
            "with Cython available)."
        ) from e

    # madmom's RNN is hard-wired to 44.1kHz; resample if needed.
    if sr != _MADMOM_SAMPLE_RATE:
        try:
            import librosa
        except ImportError as e:
            raise MeterDetectionError(
                "librosa is required to resample input for the madmom backend"
            ) from e
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=_MADMOM_SAMPLE_RATE,
        )

    sig = Signal(audio, sample_rate=_MADMOM_SAMPLE_RATE)
    activations = RNNDownBeatProcessor()(sig)
    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=list(_METER_CANDIDATES), fps=_MADMOM_FPS
    )

    with _allow_inhomogeneous_asarray():
        beats = tracker(activations)

    if len(beats) == 0:
        raise MeterDetectionError("madmom found no beats in the input")

    positions = beats[:, 1].astype(int)
    bpb = int(positions.max())
    if bpb < 2:
        raise MeterDetectionError(
            f"madmom returned implausible beats-per-bar: {bpb}"
        )

    times = beats[:, 0]
    if len(times) >= 2:
        beat_intervals = np.diff(times)
        median_interval = float(np.median(beat_intervals))
        tempo = 60.0 / median_interval if median_interval > 0 else float("nan")
    else:
        tempo = float("nan")

    return MeterResult(
        beats_per_bar=bpb,
        time_signature=f"{bpb}/4",
        tempo_bpm=tempo,
        backend="madmom",
        # madmom's DBN tracker doesn't expose per-candidate likelihoods through
        # its public API, so we leave this empty rather than fabricating one.
        candidate_scores={},
    )


# ----------------------------------------------------------- librosa internals


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
