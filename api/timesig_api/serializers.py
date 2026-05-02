from rest_framework import serializers

from .models import AnalysisRequest


class AnalysisRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisRequest
        fields = [
            "id",
            "url",
            "backend",
            "status",
            "is_music",
            "music_confidence",
            "top_label",
            "time_signature",
            "beats_per_bar",
            "tempo_bpm",
            "error_message",
            "created_at",
            "completed_at",
        ]
        read_only_fields = fields


class AnalyzeInputSerializer(serializers.Serializer):
    url = serializers.URLField(max_length=2048)
    backend = serializers.ChoiceField(
        choices=["librosa", "madmom"], default="librosa", required=False
    )
    music_threshold = serializers.FloatField(
        required=False, min_value=0.0, max_value=1.0, default=0.5
    )
    start_time = serializers.FloatField(required=False, min_value=0.0, default=0.0)
    end_time = serializers.FloatField(required=False, min_value=0.0, allow_null=True, default=None)
    max_duration = serializers.FloatField(required=False, min_value=1.0, default=None)
