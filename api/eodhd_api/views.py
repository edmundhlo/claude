"""
Thin REST surface over `eodhd.EODHDClient`.

Every public client method is mapped to a kebab-case URL slug. Query string
parameters are forwarded as keyword arguments; the `from` reserved word is
remapped to the client's `from_` kwarg automatically.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings
from rest_framework.exceptions import NotFound, ParseError
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from eodhd import (
    EODHDAuthError,
    EODHDClient,
    EODHDHTTPError,
    EODHDNotFoundError,
    EODHDRateLimitError,
)


# slug -> (client method, name of the URL path argument, or None)
ENDPOINTS: dict[str, tuple[str, str | None]] = {
    # market data
    "eod": ("eod", "symbol"),
    "real-time": ("real_time", "symbol"),
    "intraday": ("intraday", "symbol"),
    "dividends": ("dividends", "symbol"),
    "splits": ("splits", "symbol"),
    "technical": ("technical", "symbol"),
    "market-cap": ("market_cap", "symbol"),
    # fundamentals
    "fundamentals": ("fundamentals", "symbol"),
    "bulk-fundamentals": ("bulk_fundamentals", "exchange"),
    "insider-transactions": ("insider_transactions", None),
    # options
    "options": ("options", "symbol"),
    # search / exchanges
    "search": ("search", "query"),
    "exchanges-list": ("exchanges_list", None),
    "exchange-symbols": ("exchange_symbols", "exchange"),
    "exchange-details": ("exchange_details", "exchange"),
    "bulk-eod": ("bulk_eod", "exchange"),
    # news / sentiment
    "news": ("news", None),
    "sentiments": ("sentiments", None),
    "tweets-sentiments": ("tweets_sentiments", None),
    # calendar
    "calendar/earnings": ("calendar_earnings", None),
    "calendar/trends": ("calendar_trends", None),
    "calendar/ipos": ("calendar_ipos", None),
    "calendar/splits": ("calendar_splits", None),
    # macro / bonds / user
    "macro-indicator": ("macro_indicator", "country"),
    "economic-events": ("economic_events", None),
    "bond-fundamentals": ("bond_fundamentals", "isin"),
    "user": ("user", None),
}


def _coerce(value: str) -> Any:
    """Best-effort coerce a query-string value to int/float, falling back to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _kwargs_from_query(request: Request) -> dict[str, Any]:
    """Build keyword arguments from the request's query string.

    `from` is a Python keyword and the client uses `from_` instead — translate
    it transparently so callers can write `?from=2024-01-01`.
    """
    out: dict[str, Any] = {}
    for key in request.query_params.keys():
        values = request.query_params.getlist(key)
        target = "from_" if key == "from" else key
        if len(values) == 1:
            out[target] = _coerce(values[0])
        else:
            out[target] = [_coerce(v) for v in values]
    return out


def _get_client() -> EODHDClient:
    token = settings.EODHD_API_TOKEN
    if not token:
        raise ParseError("EODHD_API_TOKEN env var is not set")
    return EODHDClient(api_token=token)


class EODHDDispatchView(APIView):
    """Single dispatcher for every endpoint listed in ENDPOINTS."""

    def get(self, request: Request, slug: str, path_arg: str | None = None) -> Response:
        if slug not in ENDPOINTS:
            raise NotFound(f"unknown endpoint: {slug}")
        method_name, expected_path_arg = ENDPOINTS[slug]

        if expected_path_arg and path_arg is None:
            raise ParseError(f"this endpoint requires a path arg: /{slug}/<{expected_path_arg}>")
        if not expected_path_arg and path_arg is not None:
            raise ParseError(f"this endpoint does not take a path arg: /{slug}")

        kwargs = _kwargs_from_query(request)
        args: tuple = () if path_arg is None else (path_arg,)

        client = _get_client()
        try:
            method = getattr(client, method_name)
            data = method(*args, **kwargs)
        except EODHDNotFoundError as e:
            raise NotFound(str(e))
        except EODHDAuthError as e:
            return Response({"error": "auth", "detail": str(e)}, status=401)
        except EODHDRateLimitError as e:
            return Response({"error": "rate_limited", "detail": str(e)}, status=429)
        except EODHDHTTPError as e:
            return Response({"error": "upstream", "detail": str(e)}, status=502)
        except TypeError as e:
            # Unexpected kwarg / missing required kwarg from the caller.
            raise ParseError(str(e))
        finally:
            client.close()

        # Some EODHD endpoints (fmt=csv) return text; pass it through verbatim.
        if isinstance(data, str):
            return Response({"data": data})
        return Response(data)


class EODHDIndexView(APIView):
    """Lists every endpoint slug this service exposes."""

    def get(self, _request: Request) -> Response:
        return Response(
            {
                "endpoints": [
                    {
                        "slug": slug,
                        "method": method,
                        "path_arg": path_arg,
                    }
                    for slug, (method, path_arg) in sorted(ENDPOINTS.items())
                ]
            }
        )
