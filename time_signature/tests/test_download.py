"""Tests for the YouTube audio download wrapper.

yt-dlp is mocked; we don't hit YouTube during tests.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from time_signature.download import download_audio
from time_signature.exceptions import DownloadError


def _install_fake_ytdlp(monkeypatch, *, info=None, raise_dl_error=False, raise_other=None):
    """Inject a fake ``yt_dlp`` module into ``sys.modules``."""

    class FakeDLError(Exception):
        pass

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=True):
            if raise_dl_error:
                raise FakeDLError("boom")
            if raise_other is not None:
                raise raise_other
            return info

    fake_module = mock.MagicMock()
    fake_module.YoutubeDL = FakeYDL
    fake_module.utils.DownloadError = FakeDLError
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    return fake_module, FakeDLError


def test_returns_expected_wav_path_when_file_exists(tmp_path, monkeypatch):
    _install_fake_ytdlp(monkeypatch, info={"id": "abc123"})
    # The download module checks for the file's existence; create it.
    (tmp_path / "abc123.wav").write_bytes(b"RIFF")

    path = download_audio("https://youtu.be/abc123", tmp_path)

    assert path == tmp_path / "abc123.wav"


def test_missing_extracted_file_raises(tmp_path, monkeypatch):
    _install_fake_ytdlp(monkeypatch, info={"id": "abc123"})
    # Don't write the wav file — simulates ffmpeg missing or extraction failed.

    with pytest.raises(DownloadError, match="ffmpeg"):
        download_audio("https://youtu.be/abc123", tmp_path)


def test_yt_dlp_download_error_is_wrapped(tmp_path, monkeypatch):
    _install_fake_ytdlp(monkeypatch, raise_dl_error=True)

    with pytest.raises(DownloadError, match="yt-dlp failed"):
        download_audio("https://youtu.be/abc123", tmp_path)


def test_missing_video_id_raises(tmp_path, monkeypatch):
    _install_fake_ytdlp(monkeypatch, info={})

    with pytest.raises(DownloadError, match="no video id"):
        download_audio("https://youtu.be/abc123", tmp_path)


def test_download_dir_is_created(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "downloads"
    _install_fake_ytdlp(monkeypatch, info={"id": "abc"})
    (tmp_path / "nested").mkdir()
    # The function should create the leaf directory.

    with pytest.raises(DownloadError):
        download_audio("https://youtu.be/abc", target)
    assert target.exists()
