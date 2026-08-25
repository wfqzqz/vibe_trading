"""Tests for the A-share metadata column builders."""

from __future__ import annotations

import pandas as pd
import pytest

from qmt_bridge.metadata import (
    attach_metadata,
    compute_limit_bands,
    mark_ex_dividend,
    mark_suspension,
    normalize_symbol,
    price_limit_ratio,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600519.SH", 0.10),
        ("000001.SZ", 0.10),
        ("300750.SZ", 0.20),
        ("688981.SH", 0.20),
        ("830799.BJ", 0.30),
        ("920003.BJ", 0.30),
        ("430047.BJ", 0.30),
        ("920003", 0.30),
        ("sh600519", 0.10),
    ],
)
def test_price_limit_ratio(symbol: str, expected: float) -> None:
    assert price_limit_ratio(symbol) == expected


def test_normalize_symbol() -> None:
    assert normalize_symbol("  sh600519 ") == "SH600519"
    assert normalize_symbol("600519.SH") == "600519.SH"


def test_compute_limit_bands() -> None:
    pre_close = pd.Series([10.0, 10.0, 20.0])
    up, down = compute_limit_bands(pre_close, "600519.SH")
    assert up.tolist() == [11.0, 11.0, 22.0]
    assert down.tolist() == [9.0, 9.0, 18.0]


def test_mark_suspension_flags_zero_volume() -> None:
    frame = pd.DataFrame(
        {"volume": [100.0, 0.0, float("nan")]},
        index=pd.RangeIndex(3),
    )
    flags = mark_suspension(frame)
    assert flags.tolist() == [False, True, True]


def test_mark_suspension_without_volume_column() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.RangeIndex(2))
    assert mark_suspension(frame).tolist() == [False, False]


def test_mark_ex_dividend() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    flags = mark_ex_dividend(idx, ["2024-01-03"])
    assert flags.tolist() == [False, True, False]


def test_mark_ex_dividend_empty() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    flags = mark_ex_dividend(idx, None)
    assert flags.tolist() == [False, False]


def test_attach_metadata_columns() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.1],
            "high": [10.5, 10.4],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100.0, 0.0],
        },
        index=pd.DatetimeIndex(idx, name="trade_date"),
    )
    out = attach_metadata(frame, symbol="600519.SH", ex_dividend_dates=["2024-01-02"])
    for col in ("pre_close", "suspended", "limit_up", "limit_down", "ex_dividend", "adj_factor"):
        assert col in out.columns
    assert out["suspended"].tolist() == [False, True]
    assert out["ex_dividend"].tolist() == [True, False]
    # pre_close is the shifted close; the first bar has none, bands fall back to close.
    assert out["limit_up"].iloc[1] == pytest.approx(round(10.1 * 1.1, 2))
