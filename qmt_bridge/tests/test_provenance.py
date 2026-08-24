"""Tests for the unified response provenance."""

from __future__ import annotations

import pandas as pd

from qmt_bridge.provenance import build_provenance


def _yesterday() -> str:
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def test_provenance_shape() -> None:
    prov = build_provenance(
        symbol="600519.SH", timeframe="1d", adjust="qfq", end_date=_yesterday()
    )
    assert set(prov) == {"source", "symbol", "timeframe", "adjust", "volume_unit", "is_final"}
    assert prov["source"] == "miniqmt"
    assert prov["symbol"] == "600519.SH"
    assert prov["timeframe"] == "1d"
    assert prov["adjust"] == "qfq"
    assert prov["volume_unit"] == "lots"


def test_is_final_true_for_settled_range() -> None:
    prov = build_provenance(symbol="600519.SH", timeframe="1d", adjust="qfq", end_date=_yesterday())
    assert prov["is_final"] is True


def test_is_final_false_for_today() -> None:
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    prov = build_provenance(symbol="600519.SH", timeframe="1d", adjust="qfq", end_date=today)
    assert prov["is_final"] is False


def test_extra_keys_merged() -> None:
    prov = build_provenance(
        symbol="600519.SH", timeframe="1d", adjust="qfq", end_date=_yesterday(),
        extra={"resolved_symbol": "600519.SH"},
    )
    assert prov["resolved_symbol"] == "600519.SH"
