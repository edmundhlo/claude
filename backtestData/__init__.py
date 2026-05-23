"""
market_data.py — Interface for accessing historical EOD trade data.

Loads a single CSV of OHLCV data into memory and provides lookup methods
indexed by symbol and date, suitable for backtesting quant strategies.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator, Optional
import numpy as np
import pandas as pd

__all__ = ["MarketData"]


class MarketData:
    """
    In-memory interface to historical end-of-day OHLCV data.

    Expects a CSV with columns:
        Date, Open, High, Low, Close, Adjusted_close, Volume, exchange, symbol

    Symbol is treated as the unique security identifier.
    """

    OHLCV_COLS = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

    def __init__(self, csv_path: str | Path = Path(__file__).parent / 'US_common_stocks.csv'):
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame = self._load(self.csv_path)
        # Per-symbol view drops the redundant symbol column (it's the dict key).
        self._by_symbol: dict[str, pd.DataFrame] = {
            sym: g.drop(columns="symbol").sort_index()
            for sym, g in self._df.groupby("symbol", observed=True)
        }
        # _df is already sorted ascending, so .unique() preserves order.
        self._trading_days: pd.DatetimeIndex = self._df.index.unique()

    # ---------- Loading ----------

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            parse_dates=["Date"],
            dtype={
                "Open": "float32",
                "High": "float32",
                "Low": "float32",
                "Close": "float32",
                "Adjusted_close": "float32",
                "Volume": "int64",
                "exchange": "category",
                "symbol": "category",
            },
        )
        df = df.set_index("Date").sort_index()
        return df

    # ---------- Universe ----------

    @property
    def symbols(self) -> list[str]:
        """All unique symbols in the dataset."""
        return list(self._by_symbol.keys())

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """All unique trading days across the universe, sorted ascending."""
        return self._trading_days

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._trading_days[0], self._trading_days[-1]

    def symbols_on(self, date) -> list[str]:
        """Symbols with a price observation on the given date."""
        ts = pd.Timestamp(date)
        if ts not in self._df.index:
            return []
        slice_ = self._df.loc[ts, "symbol"]
        # Single-row hit returns a scalar; multi-row hit returns a Series.
        if isinstance(slice_, str):
            return [slice_]
        return slice_.unique().tolist()

    # ---------- Single-symbol access ----------

    def history(
        self,
        symbol: str,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Full OHLCV history for a symbol over an optional date window."""
        if symbol not in self._by_symbol:
            raise KeyError(f"Unknown symbol: {symbol}")
        df = self._by_symbol[symbol]
        if start is not None or end is not None:
            df = df.loc[pd.Timestamp(start) if start else None :
                       pd.Timestamp(end) if end else None]
        return df[self.OHLCV_COLS]

    def bar(self, symbol: str, date) -> Optional[pd.Series]:
        """Single OHLCV bar for a symbol on a date, or None if no data."""
        df = self._by_symbol.get(symbol)
        if df is None:
            return None
        ts = pd.Timestamp(date)
        if ts not in df.index:
            return None
        row = df.loc[ts, self.OHLCV_COLS]
        # Duplicate (symbol, date) rows can occur in raw EOD data; take the last.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    def price(self, symbol: str, date, field: str = "Adjusted_close") -> Optional[float]:
        """Single price for a symbol on a date. Defaults to adjusted close."""
        bar = self.bar(symbol, date)
        if bar is None:
            return None
        return float(bar[field])

    # ---------- Cross-section / panel access ----------

    def panel(
        self,
        field: str = "Adjusted_close",
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Wide-format panel: dates as rows, symbols as columns, one field.
        Useful for vectorized signal computation.
        """
        df = self._df
        if symbols is not None:
            df = df[df["symbol"].isin(list(symbols))]
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[:pd.Timestamp(end)]
        # .pivot raises on duplicate (Date, symbol); pivot_table would silently average them.
        return df.pivot(columns="symbol", values=field)

    def returns(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
        method: str = "simple",
    ) -> pd.DataFrame:
        """Wide panel of per-symbol returns from Adjusted_close."""
        prices = self.panel("Adjusted_close", symbols, start, end)
        if method == "simple":
            return prices.pct_change()
        elif method == "log":
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return method: {method}")

    # ---------- Iteration for event-driven backtests ----------

    def iter_days(
        self,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
        """
        Yield (date, cross_section_df) for each trading day in range.
        cross_section_df is indexed by symbol with OHLCV columns.
        Suitable for event-driven backtest loops.
        """
        days = self._trading_days
        if start is not None:
            days = days[days >= pd.Timestamp(start)]
        if end is not None:
            days = days[days <= pd.Timestamp(end)]
        if len(days) == 0:
            return

        # Slice once, then groupby — avoids an O(N) scan per day.
        sub = self._df.loc[days[0]:days[-1]]
        for day, group in sub.groupby(level=0, sort=False, observed=True):
            yield pd.Timestamp(day), group.set_index("symbol")[self.OHLCV_COLS]

    def last_price(
            self,
            symbol: str,
            date,
            field: str = "Adjusted_close",
    ) -> Optional[float]:
        """
        Most recent price at or before `date` for `symbol`.
        Returns None if the symbol has no observations on or before that date.
        """
        df = self._by_symbol.get(symbol)
        if df is None:
            return None
        ts = pd.Timestamp(date)
        # Index is sorted ascending; searchsorted finds the insertion point
        pos = df.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        return float(df.iloc[pos][field])


    # ---------- Diagnostics ----------

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        start, end = self.date_range
        return (
            f"MarketData(symbols={len(self.symbols)}, "
            f"rows={len(self):,}, "
            f"dates={start.date()}..{end.date()})"
        )