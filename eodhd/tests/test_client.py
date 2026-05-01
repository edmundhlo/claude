"""Unit tests for the EODHD client.

The HTTP layer is mocked end-to-end so these tests run offline and never need
a real API key. Each test verifies that the right URL was constructed, the
right query parameters were sent, and the response was decoded correctly.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
import requests

from eodhd import EODHDClient
from eodhd.exceptions import (
    EODHDAuthError,
    EODHDHTTPError,
    EODHDNotFoundError,
    EODHDRateLimitError,
)


# --------------------------------------------------------------------- helpers


def make_response(json_body=None, text=None, status=200):
    resp = mock.Mock(spec=requests.Response)
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.text = text if text is not None else json.dumps(json_body or {})
    resp.json = mock.Mock(return_value=json_body if json_body is not None else {})
    return resp


@pytest.fixture
def session():
    return mock.Mock(spec=requests.Session)


@pytest.fixture
def client(session):
    return EODHDClient(api_token="TESTKEY", session=session)


def call(session):
    """Return (url, params) of the last GET on the mock session."""
    assert session.get.call_count >= 1
    args, kwargs = session.get.call_args
    url = args[0] if args else kwargs["url"]
    params = kwargs.get("params", {})
    return url, params


# --------------------------------------------------------------- construction


class TestConstruction:
    def test_requires_api_token(self):
        with pytest.raises(ValueError):
            EODHDClient(api_token="")

    def test_strips_trailing_slash_on_base_url(self):
        c = EODHDClient(api_token="k", base_url="https://example.com/api/")
        assert c.base_url == "https://example.com/api"

    def test_context_manager_closes_session(self, session):
        with EODHDClient(api_token="k", session=session) as c:
            assert c.session is session
        session.close.assert_called_once()


# --------------------------------------------------------------- error mapping


class TestErrorMapping:
    @pytest.mark.parametrize(
        "status,exc",
        [
            (401, EODHDAuthError),
            (403, EODHDAuthError),
            (404, EODHDNotFoundError),
            (429, EODHDRateLimitError),
            (500, EODHDHTTPError),
            (502, EODHDHTTPError),
        ],
    )
    def test_http_errors_map_to_typed_exceptions(self, client, session, status, exc):
        session.get.return_value = make_response(text="boom", status=status)
        with pytest.raises(exc) as info:
            client.eod("AAPL.US")
        assert info.value.status_code == status

    def test_2xx_passes_through(self, client, session):
        session.get.return_value = make_response(json_body=[{"date": "2024-01-02"}])
        result = client.eod("AAPL.US")
        assert result == [{"date": "2024-01-02"}]


# ----------------------------------------------------------- core URL & params


class TestCoreRequest:
    def test_includes_api_token_and_default_json_fmt(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.eod("AAPL.US")
        url, params = call(session)
        assert url == "https://eodhd.com/api/eod/AAPL.US"
        assert params["api_token"] == "TESTKEY"
        assert params["fmt"] == "json"

    def test_drops_none_query_params(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.eod("AAPL.US", from_="2024-01-01")
        _, params = call(session)
        assert "from" in params and params["from"] == "2024-01-01"
        assert "to" not in params
        assert "period" not in params

    def test_csv_returns_raw_text(self, client, session):
        csv_body = "Date,Open,High,Low,Close\n2024-01-02,1,2,0.5,1.5\n"
        session.get.return_value = make_response(text=csv_body)
        out = client.eod("AAPL.US", fmt="csv")
        assert out == csv_body
        _, params = call(session)
        assert params["fmt"] == "csv"

    def test_url_encodes_symbol(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.eod("BRK-B.US")
        url, _ = call(session)
        assert url.endswith("/eod/BRK-B.US")


# --------------------------------------------------------------- market data


class TestMarketData:
    def test_eod_passes_all_params(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.eod(
            "AAPL.US",
            period="w",
            order="d",
            from_="2024-01-01",
            to="2024-12-31",
            filter="last_close",
        )
        url, params = call(session)
        assert url.endswith("/eod/AAPL.US")
        assert params == {
            "api_token": "TESTKEY",
            "fmt": "json",
            "period": "w",
            "order": "d",
            "from": "2024-01-01",
            "to": "2024-12-31",
            "filter": "last_close",
        }

    def test_real_time_single_symbol(self, client, session):
        session.get.return_value = make_response(json_body={"code": "AAPL.US"})
        client.real_time("AAPL.US")
        url, params = call(session)
        assert url.endswith("/real-time/AAPL.US")
        assert "s" not in params

    def test_real_time_multi_symbol_accepts_list(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.real_time("AAPL.US", extra_symbols=["VTI", "EUR.FOREX"])
        _, params = call(session)
        assert params["s"] == "VTI,EUR.FOREX"

    def test_real_time_multi_symbol_accepts_string(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.real_time("AAPL.US", extra_symbols="VTI,EUR.FOREX")
        _, params = call(session)
        assert params["s"] == "VTI,EUR.FOREX"

    def test_intraday_default_and_unix_timestamps(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.intraday("AAPL.US", interval="1m", from_=1627896900, to=1630575300)
        url, params = call(session)
        assert url.endswith("/intraday/AAPL.US")
        assert params["interval"] == "1m"
        assert params["from"] == 1627896900
        assert params["to"] == 1630575300

    def test_intraday_split_dt_param(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.intraday("AAPL.US", split_dt=1)
        _, params = call(session)
        assert params["split-dt"] == 1

    def test_dividends(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.dividends("AAPL.US", from_="2000-01-01")
        url, params = call(session)
        assert url.endswith("/div/AAPL.US")
        assert params["from"] == "2000-01-01"

    def test_splits(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.splits("AAPL.US")
        url, _ = call(session)
        assert url.endswith("/splits/AAPL.US")

    def test_technical_requires_function_and_passes_params(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.technical(
            "AAPL.US",
            function="sma",
            period=50,
            from_="2024-01-01",
            order="a",
            splitadjusted_only=1,
        )
        url, params = call(session)
        assert url.endswith("/technical/AAPL.US")
        assert params["function"] == "sma"
        assert params["period"] == 50
        assert params["splitadjusted_only"] == 1

    def test_market_cap(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.market_cap("AAPL.US", from_="2024-01-01", to="2024-06-30")
        url, params = call(session)
        assert url.endswith("/historical-market-cap/AAPL.US")
        assert params["from"] == "2024-01-01"
        assert params["to"] == "2024-06-30"


# --------------------------------------------------------------- fundamentals


class TestFundamentals:
    def test_fundamentals_passes_filter(self, client, session):
        session.get.return_value = make_response(json_body={"General": {}})
        client.fundamentals("AAPL.US", filter="General::Code")
        url, params = call(session)
        assert url.endswith("/fundamentals/AAPL.US")
        assert params["filter"] == "General::Code"

    def test_bulk_fundamentals_with_symbol_list(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.bulk_fundamentals("US", symbols=["AAPL", "MSFT"], limit=100)
        url, params = call(session)
        assert url.endswith("/bulk-fundamentals/US")
        assert params["symbols"] == "AAPL,MSFT"
        assert params["limit"] == 100

    def test_insider_transactions(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.insider_transactions(code="AAPL.US", limit=10, from_="2024-01-01")
        url, params = call(session)
        assert url.endswith("/insider-transactions")
        assert params["code"] == "AAPL.US"
        assert params["limit"] == 10
        assert params["from"] == "2024-01-01"


# --------------------------------------------------------------------- options


class TestOptions:
    def test_options_passes_date_range(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.options("AAPL.US", from_="2024-01-01", to="2024-12-31")
        url, params = call(session)
        assert url.endswith("/options/AAPL.US")
        assert params["from"] == "2024-01-01"
        assert params["to"] == "2024-12-31"


# ---------------------------------------------------------------------- search


class TestSearch:
    def test_search_with_filters(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.search("apple", limit=5, type="stock", exchange="US")
        url, params = call(session)
        assert url.endswith("/search/apple")
        assert params["limit"] == 5
        assert params["type"] == "stock"
        assert params["exchange"] == "US"

    def test_search_url_encodes_spaces(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.search("apple inc")
        url, _ = call(session)
        assert "/search/apple%20inc" in url


# ------------------------------------------------------------------ exchanges


class TestExchanges:
    def test_exchanges_list(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.exchanges_list()
        url, _ = call(session)
        assert url.endswith("/exchanges-list/")

    def test_exchange_symbols(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.exchange_symbols("US", delisted=1, type="common_stock")
        url, params = call(session)
        assert url.endswith("/exchange-symbol-list/US")
        assert params["delisted"] == 1
        assert params["type"] == "common_stock"

    def test_exchange_details(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.exchange_details("US", from_="2024-01-01")
        url, params = call(session)
        assert url.endswith("/exchange-details/US")
        assert params["from"] == "2024-01-01"


# -------------------------------------------------------------------- bulk EOD


class TestBulkEOD:
    def test_default_eod(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.bulk_eod("US", date="2024-06-03")
        url, params = call(session)
        assert url.endswith("/eod-bulk-last-day/US")
        assert params["date"] == "2024-06-03"
        assert "type" not in params

    def test_splits_type(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.bulk_eod("US", type="splits")
        _, params = call(session)
        assert params["type"] == "splits"

    def test_symbols_csv(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.bulk_eod("US", symbols=["AAPL", "MSFT", "BMW.XETRA"])
        _, params = call(session)
        assert params["symbols"] == "AAPL,MSFT,BMW.XETRA"


# ------------------------------------------------------------ news / sentiment


class TestNewsAndSentiment:
    def test_news_requires_s_or_t(self, client, session):
        with pytest.raises(ValueError):
            client.news()
        session.get.assert_not_called()

    def test_news_by_ticker(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.news(s="AAPL.US", limit=10, offset=0)
        url, params = call(session)
        assert url.endswith("/news")
        assert params["s"] == "AAPL.US"
        assert params["limit"] == 10
        assert params["offset"] == 0

    def test_news_by_topic(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.news(t="technology")
        _, params = call(session)
        assert params["t"] == "technology"

    def test_sentiments_csv_serializes_iterables(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.sentiments(s=["AAPL.US", "BTC-USD.CC"])
        url, params = call(session)
        assert url.endswith("/sentiments")
        assert params["s"] == "AAPL.US,BTC-USD.CC"

    def test_tweets_sentiments(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.tweets_sentiments(s="AAPL.US", from_="2024-01-01")
        url, params = call(session)
        assert url.endswith("/tweets-sentiments")
        assert params["s"] == "AAPL.US"
        assert params["from"] == "2024-01-01"


# -------------------------------------------------------------------- calendar


class TestCalendar:
    def test_earnings_with_symbols_iterable(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.calendar_earnings(symbols=["AAPL.US", "MSFT.US"])
        url, params = call(session)
        assert url.endswith("/calendar/earnings")
        assert params["symbols"] == "AAPL.US,MSFT.US"

    def test_trends_serializes_symbols(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.calendar_trends(["AAPL.US", "MSFT.US"])
        url, params = call(session)
        assert url.endswith("/calendar/trends")
        assert params["symbols"] == "AAPL.US,MSFT.US"

    def test_ipos(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.calendar_ipos(from_="2024-01-01", to="2024-12-31")
        url, params = call(session)
        assert url.endswith("/calendar/ipos")
        assert params["from"] == "2024-01-01"

    def test_splits(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.calendar_splits(symbols="AAPL.US")
        url, params = call(session)
        assert url.endswith("/calendar/splits")
        assert params["symbols"] == "AAPL.US"


# -------------------------------------------------------------------- macro


class TestMacro:
    def test_macro_indicator(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.macro_indicator("USA", indicator="inflation_consumer_prices_annual")
        url, params = call(session)
        assert url.endswith("/macro-indicator/USA")
        assert params["indicator"] == "inflation_consumer_prices_annual"

    def test_economic_events(self, client, session):
        session.get.return_value = make_response(json_body=[])
        client.economic_events(
            from_="2024-01-01",
            to="2024-12-31",
            country="US",
            comparison="yoy",
            type="GDP Growth Rate",
            limit=100,
            offset=0,
        )
        url, params = call(session)
        assert url.endswith("/economic-events")
        assert params["country"] == "US"
        assert params["comparison"] == "yoy"
        assert params["type"] == "GDP Growth Rate"
        assert params["limit"] == 100


# --------------------------------------------------------------------- bonds


class TestBonds:
    def test_bond_fundamentals(self, client, session):
        session.get.return_value = make_response(json_body={})
        client.bond_fundamentals("US912810TM03")
        url, _ = call(session)
        assert url.endswith("/bond-fundamentals/US912810TM03")


# ---------------------------------------------------------------------- user


class TestUser:
    def test_user(self, client, session):
        session.get.return_value = make_response(
            json_body={"name": "Test", "dailyRateLimit": 100000}
        )
        out = client.user()
        url, _ = call(session)
        assert url.endswith("/user")
        assert out["dailyRateLimit"] == 100000
