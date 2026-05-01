"""High-level orchestration: URL → audio → classification → meter."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .classifier import ClassificationResult, classify
from .download import download_audio
from .exceptions import AudioLoadError
from .meter import Backend, MeterResult, estimate_meter

# Cap analysis at the first N seconds when no explicit window is given. Most
# meter cues are stable across a track, and capping limits memory + runtime
# for long videos.
DEFAULT_MAX_DURATION_SECONDS = 120.0
DEFAULT_ANALYSIS_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class AnalysisResult:
    url: str
    is_music: bool
    music_confidence: float
    top_label: str
    time_signature: Optional[str]
    beats_per_bar: Optional[int]
    tempo_bpm: Optional[float]
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_url(
    url: str,
    *,
    backend: Backend = "librosa",
    music_threshold: float = 0.5,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_duration: float = DEFAULT_MAX_DURATION_SECONDS,
    keep_audio: bool = False,
    work_dir: Optional[str | os.PathLike] = None,
) -> AnalysisResult:
    """Download a YouTube video's audio and analyse its time signature.

    Parameters
    ----------
    url:
        Any URL yt-dlp accepts (YouTube, etc).
    backend:
        ``"librosa"`` (default) or ``"madmom"`` (not yet implemented).
    music_threshold:
        Minimum YAMNet music-class confidence required to classify as music.
    start_time:
        Seconds into the audio at which to begin sampling. Defaults to 0.
    end_time:
        Seconds into the audio at which to stop sampling. If None, the window
        ends at ``start_time + max_duration``.
    max_duration:
        Fallback window length used when ``end_time`` is None. Ignored if
        ``end_time`` is given.
    keep_audio:
        If True, keep the downloaded WAV file; otherwise it is deleted on exit.
    work_dir:
        Where to place the WAV. If None, a system temp dir is used.
    """
    _validate_window(start_time, end_time)

    if work_dir is None and not keep_audio:
        with tempfile.TemporaryDirectory(prefix="time_signature_") as tmp:
            wav = download_audio(url, tmp)
            return _analyze_file(
                wav, url=url, backend=backend,
                music_threshold=music_threshold,
                start_time=start_time, end_time=end_time,
                max_duration=max_duration,
            )

    target = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="time_signature_"))
    wav = download_audio(url, target)
    return _analyze_file(
        wav, url=url, backend=backend,
        music_threshold=music_threshold,
        start_time=start_time, end_time=end_time,
        max_duration=max_duration,
    )


def analyze_file(
    path: str | os.PathLike,
    *,
    backend: Backend = "librosa",
    music_threshold: float = 0.5,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_duration: float = DEFAULT_MAX_DURATION_SECONDS,
) -> AnalysisResult:
    """Analyse a local audio file (any format librosa/soundfile can read).

    See :func:`analyze_url` for the meaning of ``start_time``, ``end_time``,
    and ``max_duration``.
    """
    _validate_window(start_time, end_time)
    return _analyze_file(
        Path(path), url=str(path), backend=backend,
        music_threshold=music_threshold,
        start_time=start_time, end_time=end_time,
        max_duration=max_duration,
    )


def _validate_window(start_time: float, end_time: Optional[float]) -> None:
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if end_time is not None and end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )


def _analyze_file(
    path: Path,
    *,
    url: str,
    backend: Backend,
    music_threshold: float,
    start_time: float,
    end_time: Optional[float],
    max_duration: float,
) -> AnalysisResult:
    audio, sr = _load_audio(
        path,
        start_time=start_time,
        end_time=end_time,
        max_duration=max_duration,
    )

    classification: ClassificationResult = classify(audio, sr, threshold=music_threshold)

    if not classification.is_music:
        return AnalysisResult(
            url=url,
            is_music=False,
            music_confidence=classification.music_confidence,
            top_label=classification.top_label,
            time_signature=None,
            beats_per_bar=None,
            tempo_bpm=None,
            backend=backend,
        )

    meter: MeterResult = estimate_meter(audio, sr, backend=backend)
    return AnalysisResult(
        url=url,
        is_music=True,
        music_confidence=classification.music_confidence,
        top_label=classification.top_label,
        time_signature=meter.time_signature,
        beats_per_bar=meter.beats_per_bar,
        tempo_bpm=meter.tempo_bpm,
        backend=meter.backend,
    )


def _load_audio(
    path: Path,
    *,
    start_time: float,
    end_time: Optional[float],
    max_duration: float,
) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError as e:
        raise AudioLoadError("librosa is required to load audio") from e

    duration = (end_time - start_time) if end_time is not None else max_duration

    try:
        audio, sr = librosa.load(
            str(path),
            sr=DEFAULT_ANALYSIS_SAMPLE_RATE,
            mono=True,
            offset=start_time,
            duration=duration,
        )
    except Exception as e:
        raise AudioLoadError(f"failed to load audio from {path}: {e}") from e

    if audio.size == 0:
        raise AudioLoadError(
            f"audio window [{start_time}, "
            f"{end_time if end_time is not None else start_time + duration}] "
            f"of {path} is empty (start_time past end of file?)"
        )
    return audio, sr
