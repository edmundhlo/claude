from django.db import models


class AnalysisRequest(models.Model):
    """A single time-signature analysis run, persisted for audit/lookup."""

    STATUS_PENDING = "pending"
    STATUS_OK = "ok"
    STATUS_ERROR = "error"
    STATUS_CHOICES = [
        (STATUS_PENDING, "pending"),
        (STATUS_OK, "ok"),
        (STATUS_ERROR, "error"),
    ]

    url = models.URLField(max_length=2048)
    backend = models.CharField(max_length=32, default="librosa")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)

    is_music = models.BooleanField(null=True, blank=True)
    music_confidence = models.FloatField(null=True, blank=True)
    top_label = models.CharField(max_length=128, blank=True, default="")
    time_signature = models.CharField(max_length=16, blank=True, default="")
    beats_per_bar = models.IntegerField(null=True, blank=True)
    tempo_bpm = models.FloatField(null=True, blank=True)

    error_message = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["url"])]
