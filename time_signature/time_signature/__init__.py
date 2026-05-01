"""Detect the time signature of music in YouTube (and other) audio sources."""

from .analyzer import AnalysisResult, analyze_file, analyze_url
from .classifier import ClassificationResult, classify
from .download import download_audio
from .exceptions import (
    AudioLoadError,
    BackendNotAvailableError,
    ClassificationError,
    DownloadError,
    MeterDetectionError,
    TimeSignatureError,
)
from .meter import MeterResult, estimate_meter

__all__ = [
    "AnalysisResult",
    "analyze_url",
    "analyze_file",
    "ClassificationResult",
    "classify",
    "download_audio",
    "MeterResult",
    "estimate_meter",
    "TimeSignatureError",
    "DownloadError",
    "AudioLoadError",
    "ClassificationError",
    "MeterDetectionError",
    "BackendNotAvailableError",
]

__version__ = "0.1.0"
