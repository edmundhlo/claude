from django.urls import re_path

from .views import EODHDDispatchView, EODHDIndexView

dispatch = EODHDDispatchView.as_view()


urlpatterns = [
    re_path(r"^$", EODHDIndexView.as_view()),
    # Two-segment slugs (calendar/earnings, calendar/ipos, ...): match first.
    re_path(
        r"^(?P<slug>calendar/(?:earnings|trends|ipos|splits))/?$",
        dispatch,
    ),
    # Single-segment slug with no path arg.
    re_path(r"^(?P<slug>[A-Za-z0-9\-]+)/?$", dispatch),
    # Single-segment slug with a path arg (symbol/exchange/country/isin/query).
    re_path(r"^(?P<slug>[A-Za-z0-9\-]+)/(?P<path_arg>[^/]+)/?$", dispatch),
]
