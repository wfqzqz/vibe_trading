"""Tests for the xtdata client's frame coercion (volume normalization).

The bridge is the ingestion boundary between ``xtquant.xtdata`` and the rest of
the system. ``xtdata`` reports volume in single shares (股); the bridge must
normalize to board lots (1 lot = 100 shares) so the cache and HTTP bars carry
the same "lots" unit as every other A-share source (DORA-156 条件 1 / D-04
门禁 1). ``amount`` is a money value (元) and must be left untouched.
"""

from __future__ import annotations

import pandas as pd

from qmt_bridge.xtdata_client import _coerce_market_data_frame


def _xtdata_fields(symbol: str, volume_shares: list[float], amount: list[float]) -> dict:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return {
        "open": pd.DataFrame({symbol: [10.0, 10.1]}, index=idx),
        "high": pd.DataFrame({symbol: [10.5, 10.6]}, index=idx),
        "low": pd.DataFrame({symbol: [9.9, 10.0]}, index=idx),
        "close": pd.DataFrame({symbol: [10.1, 10.2]}, index=idx),
        "volume": pd.DataFrame({symbol: volume_shares}, index=idx),
        "amount": pd.DataFrame({symbol: amount}, index=idx),
    }


def test_volume_normalized_shares_to_lots() -> None:
    frame = _coerce_market_data_frame(
        _xtdata_fields("600519.SH", [12300.0, 45600.0], [1_230_000.0, 4_560_000.0]),
        "600519.SH",
    )
    assert list(frame["volume"]) == [123.0, 456.0]
    assert list(frame["amount"]) == [1_230_000.0, 4_560_000.0]


def test_volume_normalization_preserves_fractional_lots() -> None:
    # Odd-lot volume (e.g. 50 shares) is a valid fractional lot; no rounding.
    frame = _coerce_market_data_frame(
        _xtdata_fields("600519.SH", [50.0, 1.0], [5_000.0, 100.0]),
        "600519.SH",
    )
    assert list(frame["volume"]) == [0.5, 0.01]


def test_volume_column_optional() -> None:
    # A frame without a volume column must not raise (e.g. a partial feed).
    fields = _xtdata_fields("600519.SH", [12300.0], [1_230_000.0])
    del fields["volume"]
    frame = _coerce_market_data_frame(fields, "600519.SH")
    assert "volume" not in frame.columns
    assert list(frame["close"]) == [10.1, 10.2]


def test_empty_data_returns_empty_ohlcv() -> None:
    frame = _coerce_market_data_frame({}, "600519.SH")
    assert frame.empty
    assert frame.index.name == "trade_date"
