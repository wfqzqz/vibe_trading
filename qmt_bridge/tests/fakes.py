"""Shared fakes for bridge tests (no xtquant / no network)."""

from __future__ import annotations

import pandas as pd

from qmt_bridge.xtdata_client import QuoteBundle, XtdataUnavailableError


def daily_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.5, 10.6, 10.7],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100.0, 120.0, 130.0],
        },
        index=pd.DatetimeIndex(idx, name="trade_date"),
    )


class FakeProvider:
    """In-memory provider returning a fixed daily frame."""

    def __init__(self, available: bool = True, raise_on_fetch: bool = False) -> None:
        self._available = available
        self._raise_on_fetch = raise_on_fetch

    def available(self) -> bool:
        return self._available

    def _maybe_raise(self) -> None:
        if self._raise_on_fetch:
            raise XtdataUnavailableError("feed down")

    def daily(self, symbol: str, start: str, end: str, adjust: str) -> QuoteBundle:
        self._maybe_raise()
        return QuoteBundle(frame=daily_frame())

    def minute(self, symbol: str, period: str, start: str, end: str, adjust: str) -> QuoteBundle:
        self._maybe_raise()
        return QuoteBundle(frame=daily_frame())

    def tick(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        self._maybe_raise()
        return daily_frame()

    def meta(self, symbol: str) -> dict:
        self._maybe_raise()
        return {
            "symbol": symbol,
            "price_limit_ratio": 0.10,
            "adjust": "qfq",
            "suspended_dates": [],
            "ex_dividend_dates": [],
            "adjust_factors": [],
        }
