"""Tests for the loader-cache writer (byte-compatibility with base.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import qmt_bridge.cache as bridge_cache
from backtest.loaders.base import make_loader_cache_key as agent_key


def _frame() -> pd.DataFrame:
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


@pytest.mark.parametrize(
    "fields",
    [
        None,
        ["open", "high", "low", "close", "volume"],
        ["open", "close"],
    ],
)
def test_key_matches_agent_base_py(fields: list[str] | None) -> None:
    kwargs = dict(
        source="miniqmt",
        symbol="600519.SH",
        timeframe="1d",
        start_date="2024-01-02",
        end_date="2024-01-31",
        fields=fields,
    )
    assert bridge_cache.make_loader_cache_key(**kwargs) == agent_key(**kwargs)


def test_key_normalizes_dates_identically() -> None:
    # base.py normalizes via pd.Timestamp(...).strftime("%Y-%m-%d").
    assert bridge_cache.make_loader_cache_key(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024/01/02", end_date="2024-1-31", fields=None,
    ) == agent_key(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024/01/02", end_date="2024-1-31", fields=None,
    )


def test_key_partitions_on_forward_adjust() -> None:
    # A forward-adjusted (qfq/hfq) payload must never collide with a raw
    # (unadjusted) payload for the same symbol/date window — and the bridge key
    # must stay byte-identical to base.py for both 口径. Regression (DORA-177).
    kwargs = dict(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date="2024-01-31", fields=None,
    )
    raw_key = bridge_cache.make_loader_cache_key(**kwargs, forward_adjust=False)
    qfq_key = bridge_cache.make_loader_cache_key(**kwargs, forward_adjust=True)
    assert raw_key != qfq_key
    assert raw_key == agent_key(**kwargs, forward_adjust=False)
    assert qfq_key == agent_key(**kwargs, forward_adjust=True)
    # The unadjusted key must match the default (no explicit flag) derivation,
    # which is what the miniqmt loader reads with at runtime.
    raw_default = bridge_cache.make_loader_cache_key(**kwargs)
    assert raw_default == raw_key


def test_path_layout_matches_convention(tmp_path: Path) -> None:
    key = bridge_cache.make_loader_cache_key(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date="2024-01-31", fields=None,
    )
    path = bridge_cache.loader_cache_path(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date="2024-01-31", fields=None,
        cache_root=str(tmp_path),
    )
    assert path.parent == tmp_path / "miniqmt"
    assert path.name == f"{key}.parquet"


def test_range_is_final_settled_only() -> None:
    yesterday = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    future = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert bridge_cache.range_is_final(yesterday) is True
    assert bridge_cache.range_is_final(today) is False
    assert bridge_cache.range_is_final(future) is False
    assert bridge_cache.range_is_final("not-a-date") is False


def test_range_is_final_never_for_forward_adjust() -> None:
    # Regression (DORA-177): a forward-adjusted (qfq/hfq) series is a moving
    # anchor — after a new corporate action the provider re-calibrates the whole
    # history — so even a fully-elapsed past range must never be treated as a
    # final, cacheable snapshot. This mirrors base.py's
    # test_loader_cache_range_is_final_never_for_forward_adjust.
    yesterday = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert bridge_cache.range_is_final(yesterday, forward_adjust=True) is False
    assert bridge_cache.range_is_final("2020-01-01", forward_adjust=True) is False


def test_write_disabled_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_TRADING_DATA_CACHE", raising=False)
    ok = bridge_cache.write_frame(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date="2024-01-31", fields=None,
        frame=_frame(), cache_root=str(tmp_path),
    )
    assert ok is False
    assert not list(tmp_path.rglob("*.parquet"))


def test_write_unsettled_range_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_CACHE", "1")
    future = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ok = bridge_cache.write_frame(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date=future, fields=None,
        frame=_frame(), cache_root=str(tmp_path),
    )
    assert ok is False


def test_write_and_read_columns_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_CACHE", "1")
    frame = _frame()
    frame["suspended"] = False
    frame["limit_up"] = 11.0
    frame["limit_down"] = 9.0
    frame["ex_dividend"] = False
    frame["adj_factor"] = 1.0

    ok = bridge_cache.write_frame(
        source="miniqmt", symbol="600519.SH", timeframe="1d",
        start_date="2024-01-02", end_date="2024-01-31", fields=None,
        frame=frame, cache_root=str(tmp_path),
        extra_metadata={"adjust": "qfq"},
    )
    assert ok is True

    parquet = next(tmp_path.rglob("*.parquet"))
    sidecar = json.loads(parquet.with_suffix(".parquet.json").read_text(encoding="utf-8"))
    assert sidecar["adjust"] == "qfq"
    assert sidecar["index_columns"] == ["trade_date"]

    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        read_back = con.execute(f"SELECT * FROM read_parquet('{parquet}')").fetchdf()
    finally:
        con.close()
    for col in ("open", "high", "low", "close", "volume", "suspended", "limit_up", "limit_down"):
        assert col in read_back.columns
