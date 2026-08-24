"""Tests for the bridge service (fetch → metadata → cache → response)."""

from __future__ import annotations

import pandas as pd
import pytest

from qmt_bridge.service import BridgeService, ValidationError
from fakes import FakeProvider


def _yesterday() -> str:
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def test_health_reports_availability() -> None:
    service = BridgeService(FakeProvider(available=True))
    health = service.health()
    assert health["status"] == "ok"
    assert health["xtdata_available"] is True
    assert health["read_only"] is True


def test_daily_returns_bars_and_provenance() -> None:
    service = BridgeService(FakeProvider(available=True))
    result = service.daily("600519.SH", "2024-01-01", _yesterday(), "qfq")
    assert result["unavailable"] is False
    assert result["provenance"]["source"] == "miniqmt"
    assert result["provenance"]["adjust"] == "qfq"
    assert result["provenance"]["volume_unit"] == "lots"
    assert len(result["bars"]) == 3
    # daily bars carry the metadata columns.
    assert "suspended" in result["bars"][0]
    assert "limit_up" in result["bars"][0]


def test_daily_unavailable_provider() -> None:
    service = BridgeService(FakeProvider(available=False))
    result = service.daily("600519.SH", "2024-01-01", _yesterday(), "qfq")
    assert result["unavailable"] is True
    assert result["bars"] == []


def test_daily_feed_error_degrades() -> None:
    service = BridgeService(FakeProvider(available=True, raise_on_fetch=True))
    result = service.daily("600519.SH", "2024-01-01", _yesterday(), "qfq")
    assert result["unavailable"] is True
    assert result["reason"] == "feed down"


def test_daily_writes_cache_when_final(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_CACHE", "1")
    service = BridgeService(FakeProvider(available=True), cache_root=str(tmp_path))
    service.daily("600519.SH", "2024-01-01", _yesterday(), "qfq")
    parquets = list(tmp_path.rglob("*.parquet"))
    assert parquets, "expected a cache write for a settled qfq range"


def test_daily_does_not_cache_hfq(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_CACHE", "1")
    service = BridgeService(FakeProvider(available=True), cache_root=str(tmp_path))
    service.daily("600519.SH", "2024-01-01", _yesterday(), "hfq")
    assert not list(tmp_path.rglob("*.parquet"))


def test_daily_rejects_empty_symbol() -> None:
    service = BridgeService(FakeProvider(available=True))
    with pytest.raises(ValidationError, match="symbol"):
        service.daily("", "2024-01-01", "2024-01-05", "qfq")


def test_daily_rejects_symbol_without_venue() -> None:
    service = BridgeService(FakeProvider(available=True))
    with pytest.raises(ValidationError, match="venue"):
        service.daily("600519", "2024-01-01", "2024-01-05", "qfq")


def test_daily_rejects_start_after_end() -> None:
    service = BridgeService(FakeProvider(available=True))
    with pytest.raises(ValidationError, match="start_date"):
        service.daily("600519.SH", "2024-01-05", "2024-01-01", "qfq")


def test_daily_rejects_unsupported_adjust() -> None:
    service = BridgeService(FakeProvider(available=True))
    with pytest.raises(ValidationError, match="unsupported"):
        service.daily("600519.SH", "2024-01-01", "2024-01-05", "bogus")


def test_minute_validation() -> None:
    service = BridgeService(FakeProvider(available=True))
    with pytest.raises(ValidationError, match="period"):
        service.minute("600519.SH", "2024-01-01", "2024-01-05", period="7m")


def test_meta_unavailable() -> None:
    service = BridgeService(FakeProvider(available=False))
    assert service.meta("600519.SH") == {"symbol": "600519.SH", "unavailable": True}
