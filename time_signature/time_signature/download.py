"""Download YouTube audio to a local WAV file via yt-dlp + ffmpeg."""

from __future__ import annotations

import os
from pathlib import Path

from .exceptions import DownloadError


def download_audio(url: str, out_dir: str | os.PathLike) -> Path:
    """Download the audio track from a YouTube URL as a 16-bit PCM WAV file.

    Requires ``ffmpeg`` on PATH (yt-dlp shells out to it for extraction).
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError("yt-dlp is not installed") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"yt-dlp failed for {url}: {e}") from e
    except Exception as e:
        raise DownloadError(f"unexpected error downloading {url}: {e}") from e

    video_id = info.get("id")
    if not video_id:
        raise DownloadError("yt-dlp returned no video id")

    wav_path = out_dir / f"{video_id}.wav"
    if not wav_path.exists():
        raise DownloadError(
            f"expected {wav_path} after extraction; is ffmpeg installed and on PATH?"
        )
    return wav_path
