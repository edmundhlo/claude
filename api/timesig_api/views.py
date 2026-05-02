"""
HTTP wrapper around `time_signature.analyze_url`.

Each request is persisted as an `AnalysisRequest` row so callers (and ops) can
look up past results by URL without re-running the analysis.

Heads-up: `analyze_url` is synchronous and can take ~30s+ on a non-trivial
clip (download + YAMNet + librosa). For high traffic, move this behind a task
queue (Cloud Tasks + a worker service) and have this endpoint return 202.
"""

from __future__ import annotations

from django.conf import settings
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AnalysisRequest
from .serializers import AnalyzeInputSerializer, AnalysisRequestSerializer


class _UnprocessableEntity(APIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = "could not analyze input"


class AnalyzeView(APIView):
    """POST /timesig/analyze — kick off (and persist) an analysis."""

    def post(self, request) -> Response:
        in_serializer = AnalyzeInputSerializer(data=request.data)
        in_serializer.is_valid(raise_exception=True)
        data = in_serializer.validated_data

        # Lazy import: keeps the heavy ML/audio deps optional at boot time.
        try:
            from time_signature import (
                AudioLoadError,
                BackendNotAvailableError,
                ClassificationError,
                DownloadError,
                MeterDetectionError,
                analyze_url,
            )
        except ImportError as e:
            return Response(
                {
                    "error": "time_signature_unavailable",
                    "detail": (
                        "this container was built without the time_signature deps "
                        f"(numpy/librosa/tensorflow/yt-dlp): {e}"
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        record = AnalysisRequest.objects.create(
            url=data["url"],
            backend=data.get("backend", "librosa"),
            status=AnalysisRequest.STATUS_PENDING,
        )

        max_duration = data.get("max_duration") or settings.TIMESIG_DEFAULT_MAX_DURATION

        try:
            result = analyze_url(
                data["url"],
                backend=data.get("backend", "librosa"),
                music_threshold=data.get("music_threshold", 0.5),
                start_time=data.get("start_time", 0.0),
                end_time=data.get("end_time"),
                max_duration=max_duration,
            )
        except (
            DownloadError,
            AudioLoadError,
            ClassificationError,
            MeterDetectionError,
            BackendNotAvailableError,
            ValueError,
        ) as e:
            record.status = AnalysisRequest.STATUS_ERROR
            record.error_message = f"{type(e).__name__}: {e}"
            record.completed_at = timezone.now()
            record.save()
            raise _UnprocessableEntity(detail=record.error_message) from e

        record.status = AnalysisRequest.STATUS_OK
        record.is_music = result.is_music
        record.music_confidence = result.music_confidence
        record.top_label = result.top_label
        record.time_signature = result.time_signature or ""
        record.beats_per_bar = result.beats_per_bar
        record.tempo_bpm = result.tempo_bpm
        record.backend = result.backend
        record.completed_at = timezone.now()
        record.save()

        return Response(
            AnalysisRequestSerializer(record).data, status=status.HTTP_201_CREATED
        )


class AnalysisListView(generics.ListAPIView):
    """GET /timesig/analyses — paginated list of past analyses."""

    queryset = AnalysisRequest.objects.all()
    serializer_class = AnalysisRequestSerializer


class AnalysisDetailView(generics.RetrieveAPIView):
    """GET /timesig/analyses/<id>"""

    queryset = AnalysisRequest.objects.all()
    serializer_class = AnalysisRequestSerializer
