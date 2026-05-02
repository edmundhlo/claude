from django.http import JsonResponse
from django.urls import include, path


def health(_request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("health", health),
    path("eodhd/", include("eodhd_api.urls")),
    path("timesig/", include("timesig_api.urls")),
]
