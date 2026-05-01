"""
Python client for the EODHD financial data APIs.

Reference: https://eodhd.com/financial-apis/

Every endpoint method returns parsed JSON by default. Pass ``fmt="csv"`` (where
the API supports it) to receive the raw CSV body as a string instead.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests

from .exceptions import (
    EODHDAuthError,
    EODHDHTTPError,
    EODHDNotFoundError,
    EODHDRateLimitError,
)

DEFAULT_BASE_URL = "https://eodhd.com/api"
DEFAULT_TIMEOUT = 30


def _csv(value: Any) -> str:
    """Coerce a string or iterable of strings into a comma-separated string."""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return ",".join(str(x) for x in value)
    return str(value)


def _clean(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop None values so we don't send empty query parameters."""
    return {k: v for k, v in params.items() if v is not None}


class EODHDClient:
    """Thin wrapper around the EODHD REST API.

    Parameters
    ----------
    api_token:
        Your EODHD API key. Use ``"demo"`` for limited testing on a few tickers.
    base_url:
        Override the API base URL (useful for testing).
    timeout:
        Per-request timeout in seconds.
    session:
        Optional ``requests.Session`` for connection pooling. One is created
        automatically if not supplied.
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        if not api_token:
            raise ValueError("api_token is required")
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "EODHDClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ core

    def _request(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        fmt: str | None = "json",
    ) -> Any:
        merged: dict[str, Any] = {"api_token": self.api_token}
        if fmt is not None:
            merged["fmt"] = fmt
        if params:
            merged.update(_clean(params))

        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=merged, timeout=self.timeout)
        self._raise_for_status(response)

        if fmt == "json":
            return response.json()
        return response.text

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        body = response.text[:500]
        if response.status_code in (401, 403):
            raise EODHDAuthError(response.status_code, body)
        if response.status_code == 404:
            raise EODHDNotFoundError(response.status_code, body)
        if response.status_code == 429:
            raise EODHDRateLimitError(response.status_code, body)
        raise EODHDHTTPError(response.status_code, body)

    # --------------------------------------------------------- market data

    def eod(
        self,
        symbol: str,
        period: str | None = None,
        order: str | None = None,
        from_: str | None = None,
        to: str | None = None,
        filter: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """End-of-day historical OHLCV (``GET /eod/{symbol}``).

        ``period`` is one of ``d`` / ``w`` / ``m``; ``order`` is ``a`` / ``d``.
        """
        params = {
            "period": period,
            "order": order,
            "from": from_,
            "to": to,
            "filter": filter,
        }
        return self._request(f"eod/{quote(symbol)}", params, fmt=fmt)

    def real_time(
        self,
        symbol: str,
        extra_symbols: str | Iterable[str] | None = None,
        fmt: str = "json",
    ) -> Any:
        """Live (delayed) quote for one or more tickers (``GET /real-time/{symbol}``).

        Pass ``extra_symbols`` as a list/tuple or comma-separated string to fetch
        up to ~15-20 tickers in a single request.
        """
        params = {"s": _csv(extra_symbols) if extra_symbols else None}
        return self._request(f"real-time/{quote(symbol)}", params, fmt=fmt)

    def intraday(
        self,
        symbol: str,
        interval: str | None = None,
        from_: int | None = None,
        to: int | None = None,
        split_dt: int | None = None,
        fmt: str = "json",
    ) -> Any:
        """Intraday OHLCV bars (``GET /intraday/{symbol}``).

        ``interval`` is one of ``1m`` / ``5m`` / ``1h``. ``from_`` and ``to`` are
        Unix timestamps in UTC.
        """
        params = {
            "interval": interval,
            "from": from_,
            "to": to,
            "split-dt": split_dt,
        }
        return self._request(f"intraday/{quote(symbol)}", params, fmt=fmt)

    def dividends(
        self,
        symbol: str,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Historical dividends (``GET /div/{symbol}``)."""
        params = {"from": from_, "to": to}
        return self._request(f"div/{quote(symbol)}", params, fmt=fmt)

    def splits(
        self,
        symbol: str,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Historical splits (``GET /splits/{symbol}``)."""
        params = {"from": from_, "to": to}
        return self._request(f"splits/{quote(symbol)}", params, fmt=fmt)

    def technical(
        self,
        symbol: str,
        function: str,
        period: int | None = None,
        from_: str | None = None,
        to: str | None = None,
        order: str | None = None,
        splitadjusted_only: int | None = None,
        filter: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Technical indicator values (``GET /technical/{symbol}``).

        ``function`` is one of ``sma``, ``ema``, ``wma``, ``rsi``, ``macd``,
        ``bbands``, ``stochastic``, ``stochrsi``, ``atr``, ``adx``, ``dmi``,
        ``cci``, ``sar``, ``volatility``, ``stddev``, ``slope``, ``beta``,
        ``avgvol``, ``avgvolccy``, ``splitadjusted``.
        """
        params = {
            "function": function,
            "period": period,
            "from": from_,
            "to": to,
            "order": order,
            "splitadjusted_only": splitadjusted_only,
            "filter": filter,
        }
        return self._request(f"technical/{quote(symbol)}", params, fmt=fmt)

    def market_cap(
        self,
        symbol: str,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Historical market capitalization (``GET /historical-market-cap/{ticker}``)."""
        params = {"from": from_, "to": to}
        return self._request(f"historical-market-cap/{quote(symbol)}", params, fmt=fmt)

    # --------------------------------------------------------- fundamentals

    def fundamentals(
        self,
        symbol: str,
        filter: str | None = None,
        historical: int | None = None,
        from_: str | None = None,
        to: str | None = None,
    ) -> Any:
        """Company / ETF / fund / index fundamentals (``GET /fundamentals/{symbol}``)."""
        params = {
            "filter": filter,
            "historical": historical,
            "from": from_,
            "to": to,
        }
        return self._request(f"fundamentals/{quote(symbol)}", params, fmt="json")

    def bulk_fundamentals(
        self,
        exchange: str,
        offset: int | None = None,
        limit: int | None = None,
        symbols: str | Iterable[str] | None = None,
    ) -> Any:
        """Bulk fundamentals for an exchange (``GET /bulk-fundamentals/{exchange}``)."""
        params = {
            "offset": offset,
            "limit": limit,
            "symbols": _csv(symbols) if symbols else None,
        }
        return self._request(f"bulk-fundamentals/{quote(exchange)}", params, fmt="json")

    def insider_transactions(
        self,
        code: str | None = None,
        limit: int | None = None,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """SEC Form 4 insider transactions (``GET /insider-transactions``)."""
        params = {"code": code, "limit": limit, "from": from_, "to": to}
        return self._request("insider-transactions", params, fmt=fmt)

    # ------------------------------------------------------------- options

    def options(
        self,
        symbol: str,
        from_: str | None = None,
        to: str | None = None,
        trade_date_from: str | None = None,
        trade_date_to: str | None = None,
        contract_name: str | None = None,
    ) -> Any:
        """End-of-day option chain for a US stock (``GET /options/{symbol}``)."""
        params = {
            "from": from_,
            "to": to,
            "trade_date_from": trade_date_from,
            "trade_date_to": trade_date_to,
            "contract_name": contract_name,
        }
        return self._request(f"options/{quote(symbol)}", params, fmt="json")

    # -------------------------------------------------------------- search

    def search(
        self,
        query: str,
        limit: int | None = None,
        type: str | None = None,
        exchange: str | None = None,
        bonds_only: int | None = None,
    ) -> Any:
        """Symbol search (``GET /search/{query}``).

        ``type`` is one of ``stock``, ``etf``, ``fund``, ``bond``, ``index``,
        ``crypto``, or ``all``.
        """
        params = {
            "limit": limit,
            "type": type,
            "exchange": exchange,
            "bonds_only": bonds_only,
        }
        return self._request(f"search/{quote(query)}", params, fmt="json")

    # ------------------------------------------------------------ exchanges

    def exchanges_list(self, fmt: str = "json") -> Any:
        """All supported exchanges (``GET /exchanges-list/``)."""
        return self._request("exchanges-list/", None, fmt=fmt)

    def exchange_symbols(
        self,
        exchange: str,
        delisted: int | None = None,
        type: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Tickers traded on an exchange (``GET /exchange-symbol-list/{exchange}``)."""
        params = {"delisted": delisted, "type": type}
        return self._request(f"exchange-symbol-list/{quote(exchange)}", params, fmt=fmt)

    def exchange_details(
        self,
        exchange: str,
        from_: str | None = None,
        to: str | None = None,
    ) -> Any:
        """Exchange trading hours and holidays (``GET /exchange-details/{exchange}``)."""
        params = {"from": from_, "to": to}
        return self._request(f"exchange-details/{quote(exchange)}", params, fmt="json")

    # ------------------------------------------------------------- bulk EOD

    def bulk_eod(
        self,
        exchange: str,
        type: str | None = None,
        date: str | None = None,
        symbols: str | Iterable[str] | None = None,
        filter: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Bulk EOD / splits / dividends (``GET /eod-bulk-last-day/{exchange}``).

        ``type`` is ``"splits"``, ``"dividends"``, or omitted for EOD prices.
        """
        params = {
            "type": type,
            "date": date,
            "symbols": _csv(symbols) if symbols else None,
            "filter": filter,
        }
        return self._request(f"eod-bulk-last-day/{quote(exchange)}", params, fmt=fmt)

    # -------------------------------------------------------------- news / sentiment

    def news(
        self,
        s: str | None = None,
        t: str | None = None,
        from_: str | None = None,
        to: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        fmt: str = "json",
    ) -> Any:
        """Financial news feed (``GET /news``).

        Provide either ``s`` (ticker) or ``t`` (topic).
        """
        if not s and not t:
            raise ValueError("news() requires either `s` (ticker) or `t` (topic)")
        params = {
            "s": s,
            "t": t,
            "from": from_,
            "to": to,
            "limit": limit,
            "offset": offset,
        }
        return self._request("news", params, fmt=fmt)

    def sentiments(
        self,
        s: str | Iterable[str],
        from_: str | None = None,
        to: str | None = None,
    ) -> Any:
        """Sentiment scores per ticker, per day (``GET /sentiments``).

        Scores are normalized to the range ``[-1, 1]``.
        """
        params = {"s": _csv(s), "from": from_, "to": to}
        return self._request("sentiments", params, fmt="json")

    def tweets_sentiments(
        self,
        s: str | Iterable[str],
        from_: str | None = None,
        to: str | None = None,
    ) -> Any:
        """Tweet-derived sentiment per ticker (``GET /tweets-sentiments``)."""
        params = {"s": _csv(s), "from": from_, "to": to}
        return self._request("tweets-sentiments", params, fmt="json")

    # -------------------------------------------------------------- calendar

    def calendar_earnings(
        self,
        symbols: str | Iterable[str] | None = None,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Upcoming earnings (``GET /calendar/earnings``)."""
        params = {
            "symbols": _csv(symbols) if symbols else None,
            "from": from_,
            "to": to,
        }
        return self._request("calendar/earnings", params, fmt=fmt)

    def calendar_trends(self, symbols: str | Iterable[str]) -> Any:
        """Earnings trend estimates (``GET /calendar/trends``)."""
        return self._request(
            "calendar/trends",
            {"symbols": _csv(symbols)},
            fmt="json",
        )

    def calendar_ipos(
        self,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Upcoming IPOs (``GET /calendar/ipos``)."""
        params = {"from": from_, "to": to}
        return self._request("calendar/ipos", params, fmt=fmt)

    def calendar_splits(
        self,
        symbols: str | None = None,
        from_: str | None = None,
        to: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """Upcoming splits (``GET /calendar/splits``)."""
        params = {"symbols": symbols, "from": from_, "to": to}
        return self._request("calendar/splits", params, fmt=fmt)

    # -------------------------------------------------------------- macro

    def macro_indicator(
        self,
        country: str,
        indicator: str | None = None,
        fmt: str = "json",
    ) -> Any:
        """World Bank style macro indicators (``GET /macro-indicator/{country}``).

        ``country`` is an ISO 3166 alpha-3 code (e.g. ``"USA"``).
        """
        params = {"indicator": indicator}
        return self._request(f"macro-indicator/{quote(country)}", params, fmt=fmt)

    def economic_events(
        self,
        from_: str | None = None,
        to: str | None = None,
        country: str | None = None,
        comparison: str | None = None,
        type: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> Any:
        """Economic events calendar (``GET /economic-events``).

        ``comparison`` is one of ``mom`` / ``qoq`` / ``yoy``.
        """
        params = {
            "from": from_,
            "to": to,
            "country": country,
            "comparison": comparison,
            "type": type,
            "offset": offset,
            "limit": limit,
        }
        return self._request("economic-events", params, fmt="json")

    # ----------------------------------------------------------------- bonds

    def bond_fundamentals(self, isin: str) -> Any:
        """Bond fundamentals by ISIN/CUSIP (``GET /bond-fundamentals/{isin}``)."""
        return self._request(f"bond-fundamentals/{quote(isin)}", None, fmt="json")

    # ------------------------------------------------------------------ user

    def user(self, fmt: str = "json") -> Any:
        """Account info and rate-limit usage (``GET /user``)."""
        return self._request("user", None, fmt=fmt)
