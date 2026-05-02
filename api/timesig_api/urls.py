from django.urls import path

from .views import AnalysisDetailView, AnalysisListView, AnalyzeView

urlpatterns = [
    path("analyze", AnalyzeView.as_view()),
    path("analyses", AnalysisListView.as_view()),
    path("analyses/<int:pk>", AnalysisDetailView.as_view()),
]
