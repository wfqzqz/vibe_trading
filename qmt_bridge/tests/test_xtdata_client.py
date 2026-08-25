"""Tests for the xtdata client's frame coercion (real return shape + volume unit).

The bridge is the ingestion boundary between ``xtquant.xtdata`` and the rest of
the system. Empirically (DORA-156 条件 1), ``get_market_data_ex`` reports A-share
volume already in board lots (1 lot = 100 shares) — ``amount / (volume * 100)``
lands on the average per-share price — so the bridge passes ``volume`` through
unchanged; the old ``÷100`` shares→lots assumption made every miniqmt volume
100x too small. ``amount`` is a money value (元) and is untouched.

``get_market_data_ex`` returns ``{symbol: DataFrame}`` (the value is a DataFrame
indexed by the period timestamp string with one column per field); the bridge
originally coerced ``{field: DataFrame}`` (the legacy ``get_market_data`` shape),
which silently returned an empty frame for every real fetch (DORA-156 条件 3).
"""

from __future__ import annotations

import pandas as pd

from qmt_bridge.xtdata_client import (
    _coerce_factor_series,
    _coerce_market_data_frame,
    _ex_dates_from_factors,
    _to_xtdate,
)


def _xtdata_ex_shape(symbol: str, volume: list[float], amount: list[float], index=None) -> dict:
    """Build the ACTUAL ``get_market_data_ex`` return: ``{symbol: DataFrame}``."""
    n = len(volume)
    idx = index if index is not None else pd.Index([f"2024020{i+1}" for i in range(n)], name="trade_date")
    frame = pd.DataFrame(
        {
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.9 + i for i in range(n)],
            "close": [10.1 + i for i in range(n)],
            "volume": volume,
            "amount": amount,
        },
        index=idx,
    )
    return {symbol: frame}


def test_real_ex_shape_parsed() -> None:
    data = _xtdata_ex_shape("600519.SH", [26485.0, 39382.0], [4.272299e9, 6.336691e9])
    frame = _coerce_market_data_frame(data, "600519.SH")
    assert frame.index.name == "trade_date"
    assert list(frame.index) == list(pd.to_datetime(["20240201", "20240202"]))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert frame["volume"].tolist() == [26485.0, 39382.0]
    assert frame["amount"].tolist() == [4.272299e9, 6.336691e9]


def test_volume_not_scaled_xtdata_is_lots() -> None:
    # xtdata volume is already lots — it must be passed through, not ÷100.
    frame = _coerce_market_data_frame(
        _xtdata_ex_shape("600519.SH", [26485.0, 39382.0], [4.272299e9, 6.336691e9]),
        "600519.SH",
    )
    assert frame["volume"].tolist() == [26485.0, 39382.0]
    assert frame["volume"].tolist() != [264.85, 393.82]


def test_minute_index_parsed() -> None:
    # Minute bars carry a ``YYYYMMDDHHMMSS`` index token.
    data = _xtdata_ex_shape(
        "600519.SH",
        [752.0, 1069.0],
        [1.10499182e8, 1.57311886e8],
        index=pd.Index(["20250825093000", "20250825093100"]),
    )
    frame = _coerce_market_data_frame(data, "600519.SH")
    assert list(frame.index) == list(pd.to_datetime(["2025-08-25 09:30:00", "2025-08-25 09:31:00"]))
    assert frame["volume"].tolist() == [752.0, 1069.0]


def test_amount_left_untouched() -> None:
    frame = _coerce_market_data_frame(
        _xtdata_ex_shape("600519.SH", [26485.0], [4.272299e9]),
        "600519.SH",
    )
    assert frame["amount"].tolist() == [4.272299e9]


def test_volume_column_optional() -> None:
    # A frame without a volume column must not raise (e.g. a partial feed).
    idx = pd.Index(["20240201"], name="trade_date")
    data = {
        "600519.SH": pd.DataFrame(
            {"open": [10.0], "high": [10.5], "low": [9.9], "close": [10.1], "amount": [1.0]},
            index=idx,
        )
    }
    frame = _coerce_market_data_frame(data, "600519.SH")
    assert "volume" not in frame.columns
    assert frame["close"].tolist() == [10.1]


def test_legacy_field_dict_shape_returns_empty() -> None:
    # The OLD (wrong) `{field: DataFrame}` shape — the legacy get_market_data
    # layout — is no longer what get_market_data_ex returns; coercing it must
    # yield an empty frame rather than fabricating bogus data.
    data = {
        "open": pd.DataFrame({"600519.SH": [10.0]}, index=pd.Index(["20240201"])),
        "close": pd.DataFrame({"600519.SH": [10.1]}, index=pd.Index(["20240201"])),
    }
    frame = _coerce_market_data_frame(data, "600519.SH")
    assert frame.empty


def test_empty_data_returns_empty_ohlcv() -> None:
    frame = _coerce_market_data_frame({}, "600519.SH")
    assert frame.empty
    assert frame.index.name == "trade_date"


def test_symbol_missing_returns_empty() -> None:
    frame = _coerce_market_data_frame(_xtdata_ex_shape("600519.SH", [1.0], [1.0]), "000001.SZ")
    assert frame.empty


def test_drop_rows_without_ohlc() -> None:
    idx = pd.Index(["20240201", "20240202"], name="trade_date")
    data = {
        "600519.SH": pd.DataFrame(
            {
                "open": [10.0, float("nan")],
                "high": [10.5, float("nan")],
                "low": [9.9, float("nan")],
                "close": [10.1, float("nan")],
                "volume": [1.0, 1.0],
            },
            index=idx,
        )
    }
    frame = _coerce_market_data_frame(data, "600519.SH")
    assert len(frame) == 1
    assert frame["volume"].tolist() == [1.0]


def test_to_xtdate_normalizes_dashed_and_slashed() -> None:
    # xtdata rejects dashed/slashed date arguments (verified DORA-156 条件 3).
    assert _to_xtdate("2024-02-01") == "20240201"
    assert _to_xtdate("2024/02/01") == "20240201"
    assert _to_xtdate("20240201") == "20240201"


def test_coerce_factor_series_uses_dr_column() -> None:
    # get_divid_factors returns a DataFrame indexed by ex-date with a `dr` column;
    # the old heuristic picked the `time` (ms-epoch) column and flagged nothing.
    factors = pd.DataFrame(
        {
            "time": [1.6e12, 1.7e12],
            "interest": [25.911, 19.106],
            "dr": [1.015351, 1.011540],
        },
        index=pd.Index(["20230630", "20231220"]),
    )
    series = _coerce_factor_series(factors)
    assert list(series.index) == list(pd.to_datetime(["20230630", "20231220"]))
    assert series.tolist() == [1.015351, 1.011540]


def test_ex_dates_from_factors_flags_only_dividend_days() -> None:
    factors = pd.Series(
        [1.0, 1.015351, 1.0, 1.011540],
        index=pd.to_datetime(["2023-01-01", "2023-06-30", "2023-09-01", "2023-12-20"]),
    )
    ex = _ex_dates_from_factors(factors)
    assert ex == ["2023-06-30", "2023-12-20"]
