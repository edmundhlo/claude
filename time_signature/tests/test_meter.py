"""Tests for the meter estimator.

Synthesises click tracks with known meter and verifies the librosa backend
recovers the correct beats-per-bar. No network or model downloads needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from time_signature.exceptions import (
    BackendNotAvailableError,
    MeterDetectionError,
)
from time_signature.meter import estimate_meter


SR = 22050


def _click_track(beats_per_bar: int, bpm: float = 120.0, bars: int = 16) -> np.ndarray:
    """Build a click track at ``bpm`` where beat 1 of every bar is louder."""
    beat_samples = int(SR * 60.0 / bpm)
    total = beat_samples * beats_per_bar * bars
    audio = np.zeros(total, dtype=np.float32)

    click_len = int(0.01 * SR)
    decay = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)

    n_beats = beats_per_bar * bars
    for i in range(n_beats):
        start = i * beat_samples
        amp = 1.0 if i % beats_per_bar == 0 else 0.35
        audio[start : start + click_len] += amp * decay
    return audio


@pytest.mark.parametrize("meter", [3, 4])
def test_recovers_meter_from_click_track(meter):
    audio = _click_track(meter, bpm=120.0)
    result = estimate_meter(audio, SR, backend="librosa")

    assert result.beats_per_bar == meter
    assert result.time_signature == f"{meter}/4"
    assert 100.0 < result.tempo_bpm < 140.0
    assert result.backend == "librosa"
    # The chosen meter should win against the other candidates.
    assert result.candidate_scores[meter] == max(result.candidate_scores.values())


def test_rejects_short_audio():
    audio = np.zeros(int(SR * 0.5), dtype=np.float32)
    with pytest.raises(MeterDetectionError):
        estimate_meter(audio, SR)


def test_rejects_non_mono_audio():
    audio = np.zeros((2, SR * 4), dtype=np.float32)
    with pytest.raises(MeterDetectionError):
        estimate_meter(audio, SR)


def test_too_few_beats_raises():
    # Silent audio yields no beats; should fail cleanly rather than crash.
    audio = np.zeros(SR * 4, dtype=np.float32)
    with pytest.raises(MeterDetectionError):
        estimate_meter(audio, SR)


def test_unknown_backend_rejected():
    audio = _click_track(4)
    with pytest.raises(ValueError):
        estimate_meter(audio, SR, backend="acme")  # type: ignore[arg-type]


def test_madmom_backend_raises_if_not_installed(monkeypatch):
    """If the madmom import fails, the backend surfaces BackendNotAvailableError."""
    import sys

    # Hide every madmom submodule so the import inside _estimate_madmom fails.
    for mod_name in list(sys.modules):
        if mod_name == "madmom" or mod_name.startswith("madmom."):
            monkeypatch.setitem(sys.modules, mod_name, None)

    audio = _click_track(4)
    with pytest.raises(BackendNotAvailableError):
        estimate_meter(audio, SR, backend="madmom")


# Try a lightweight import probe; if madmom can't be imported (even with shims),
# skip the smoke test rather than fail the whole suite.
try:
    from time_signature.meter import _apply_madmom_compat_shims as _shim
    _shim()
    import madmom as _madmom  # noqa: F401
    _MADMOM_AVAILABLE = True
except Exception:
    _MADMOM_AVAILABLE = False


@pytest.mark.skipif(not _MADMOM_AVAILABLE, reason="madmom not importable")
def test_madmom_backend_runs_on_click_track():
    """Smoke test: madmom backend produces a valid MeterResult on synthetic
    audio. We don't assert on the meter value because madmom's RNN was trained
    on real music and is not reliable on bare clicks — just check the plumbing.
    """
    audio = _click_track(4, bars=8)
    # madmom expects 44.1kHz; the backend resamples internally.
    result = estimate_meter(audio, SR, backend="madmom")

    assert result.backend == "madmom"
    assert result.beats_per_bar in (2, 3, 4)
    assert result.time_signature == f"{result.beats_per_bar}/4"
    # tempo should be in a plausible range for a 120 bpm input
    assert 60.0 < result.tempo_bpm < 240.0
