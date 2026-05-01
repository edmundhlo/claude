class TimeSignatureError(Exception):
    """Base exception for the time_signature package."""


class DownloadError(TimeSignatureError):
    """Failed to download or extract audio from the URL."""


class AudioLoadError(TimeSignatureError):
    """Failed to decode the downloaded audio file."""


class ClassificationError(TimeSignatureError):
    """The music/speech classifier failed to produce a result."""


class MeterDetectionError(TimeSignatureError):
    """Could not estimate a meter from the audio."""


class BackendNotAvailableError(TimeSignatureError):
    """Requested analysis backend is not installed."""
