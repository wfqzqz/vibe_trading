"""Tests for miniqmt_loader: cache-first fetch, cold-read HTTP, and availability.

All HTTP is mocked at :func:`backtest.loaders._http.throttled_get` (imported into
the loader module), so no test touches a live QMT Bridge. The contract points:

  1. Registration - ``miniqmt`` is a valid source and leads the ``a_share`` chain.
  2. Availability - True when the bridge is reachable OR the cache has data;
     False only when both are unreachable (so the chain falls to free sources).
  3. Cache identity - the frame cache key uses the bridge's ``source="miniqmt"`` /
     ``fields=None`` / bridge timeframe token (``1d``, not ``1D``), so a hit
     matches the bridge's own write.
  4. Cold read - daily/minute cold reads hit the read-only endpoints; a 503
     "unavailable" (xtdata down) degrades to no-data, never an error.
  5. Parsing - bars reconstruct an OHLCV frame indexed by ``trade_date`` and
     preserve the qfq-relative metadata columns for ``china_a.py`` (条件 2).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from backtest.loaders import miniqmt_loader as mq
from backtest.loaders.miniqmt_loader import DataLoader, _parse_bars, _to_timeframe


class _Resp:
    """A requests.Response look-alike carrying status_code + json()."""

    def __init__(self, status_code: int, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


_DAILY_BARS = [
    {"trade_date": "2024-01-03", "open": 10.0, "high": 10.5, "low": 9.9,
     "close": 10.1, "volume": 120.0, "amount": 1_200_000.0,
     "pre_close": 10.0, "suspended": False, "limit_up": 11.0, "limit_down": 9.0,
     "ex_dividend": False, "adj_factor": 1.0},
    {"trade_date": "2024-01-04", "open": 10.1, "high": 10.6, "low": 10.0,
     "close": 10.2, "volume": 130.0, "amount": 1_300_000.0,
     "pre_close": 10.1, "suspended": False, "limit_up": 11.1, "limit_down": 9.1,
     "ex_dividend": False, "adj_factor": 1.0},
]


def _daily_payload():
    return {"provenance": {"source": "miniqmt", "volume_unit": "lots"},
            "bars": _DAILY_BARS, "unavailable": False}


def _route(daily=None, minute=None, health_status=200):
    """Build a throttled_get side_effect routing by endpoint path."""
    def _side(url, **kwargs):
        if url.endswith("/health"):
            return _Resp(health_status, {"status": "ok"})
        if url.endswith("/v1/quotes/daily"):
            return _Resp(200, daily if daily is not None else _daily_payload())
        if url.endswith("/v1/quotes/minute"):
            return _Resp(200, minute if minute is not None else _daily_payload())
        raise AssertionError(f"unexpected URL {url}")

    return _side


@pytest.fixture(autouse=True)
def _no_bridge(monkeypatch):
    """Default: no token, loopback host, and no cache — so availability is off
    unless a test explicitly enables a path."""
    monkeypatch.delenv("QMT_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("QMT_BRIDGE_HOST", raising=False)
    monkeypatch.delenv("QMT_BRIDGE_PORT", raising=False)
    monkeypatch.setattr(mq, "loader_cache_enabled", lambda: False)


class TestRegistration:
    def test_registered_in_registry(self):
        from backtest.loaders import registry

        registry._ensure_registered()
        assert registry.LOADER_REGISTRY.get("miniqmt") is DataLoader

    def test_in_valid_sources(self):
        from backtest.loaders import registry

        assert "miniqmt" in registry.VALID_SOURCES

    def test_leads_a_share_chain(self):
        from backtest.loaders import registry

        assert registry.FALLBACK_CHAINS["a_share"][0] == "miniqmt"

    def test_metadata(self):
        assert DataLoader.name == "miniqmt"
        assert DataLoader.markets == {"a_share"}
        assert DataLoader.requires_auth is False
        assert DataLoader.volume_units == {"a_share": "lots"}
        assert DataLoader.intervals == {"1d", "1m", "5m", "15m", "30m", "60m", "tick"}


class TestIsAvailable:
    def test_available_when_bridge_reachable(self, monkeypatch):
        monkeypatch.setattr(mq, "loader_cache_enabled", lambda: False)
        with patch.object(mq, "throttled_get", side_effect=_route(health_status=200)):
            assert DataLoader().is_available() is True

    def test_available_from_cache_when_bridge_down(self, monkeypatch):
        monkeypatch.setattr(mq, "loader_cache_enabled", lambda: True)
        monkeypatch.setattr(mq, "loader_cache_root", lambda: _FakeCacheRoot(has_data=True))
        with patch.object(mq, "throttled_get", side_effect=_route(health_status=503)):
            assert DataLoader().is_available() is True

    def test_unavailable_when_bridge_down_and_no_cache(self, monkeypatch):
        monkeypatch.setattr(mq, "loader_cache_enabled", lambda: False)
        with patch.object(mq, "throttled_get", side_effect=_route(health_status=503)):
            assert DataLoader().is_available() is False

    def test_unavailable_when_bridge_refused(self):
        def _raise(*args, **kwargs):
            raise ConnectionError("refused")

        with patch.object(mq, "throttled_get", side_effect=_raise):
            assert DataLoader().is_available() is False

    def test_unavailable_when_cache_empty_and_bridge_down(self, monkeypatch):
        monkeypatch.setattr(mq, "loader_cache_enabled", lambda: True)
        monkeypatch.setattr(mq, "loader_cache_root", lambda: _FakeCacheRoot(has_data=False))
        with patch.object(mq, "throttled_get", side_effect=_route(health_status=503)):
            assert DataLoader().is_available() is False


class _FakeCacheRoot:
    """A Path stand-in that reports whether any .parquet exists."""

    def __init__(self, has_data: bool) -> None:
        self._has_data = has_data

    def __truediv__(self, _other):
        return self

    def is_dir(self) -> bool:
        return self._has_data

    def glob(self, _pattern):
        return [object()] if self._has_data else []


class TestFetchColdRead:
    def test_daily_cold_read_returns_frame_with_metadata(self, monkeypatch):
        monkeypatch.setattr(mq, "cached_loader_fetch", lambda *, fetch, **kw: fetch())
        with patch.object(mq, "throttled_get", side_effect=_route()):
            out = DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="1D")

        assert "600519.SH" in out
        frame = out["600519.SH"]
        assert list(frame.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
        assert list(frame.columns[:5]) == ["open", "high", "low", "close", "volume"]
        # qfq-relative metadata columns are preserved for china_a.py (条件 2).
        for col in ("pre_close", "limit_up", "limit_down", "suspended", "ex_dividend", "adj_factor"):
            assert col in frame.columns
        assert frame["volume"].dtype == float

    def test_minute_cold_read_hits_minute_endpoint(self, monkeypatch):
        monkeypatch.setattr(mq, "cached_loader_fetch", lambda *, fetch, **kw: fetch())
        seen = {}

        def _side(url, **kwargs):
            seen["url"] = url
            seen["params"] = kwargs.get("params", {})
            return _Resp(200, _daily_payload())

        with patch.object(mq, "throttled_get", side_effect=_side):
            DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="1H")

        assert seen["url"].endswith("/v1/quotes/minute")
        assert seen["params"]["period"] == "60m"
        assert seen["params"]["adjust"] == "qfq"

    def test_unsupported_interval_rejected(self, monkeypatch):
        with patch.object(mq, "throttled_get", side_effect=_route()) as http:
            out = DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="4H")
        assert out == {}
        assert http.call_count == 0

    def test_bridge_503_degrades_to_no_data(self, monkeypatch):
        monkeypatch.setattr(mq, "cached_loader_fetch", lambda *, fetch, **kw: fetch())

        def _side(url, **kwargs):
            if url.endswith("/health"):
                return _Resp(200)
            return _Resp(503, {"unavailable": True, "bars": []})

        with patch.object(mq, "throttled_get", side_effect=_side):
            out = DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="1D")
        assert out == {}

    def test_transient_error_skips_symbol(self, monkeypatch):
        monkeypatch.setattr(mq, "cached_loader_fetch", lambda *, fetch, **kw: fetch())

        def _side(url, **kwargs):
            raise RuntimeError("network blip")

        with patch.object(mq, "throttled_get", side_effect=_side):
            out = DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="1D")
        assert out == {}

    def test_auth_header_and_bearer_token(self, monkeypatch):
        monkeypatch.setenv("QMT_BRIDGE_TOKEN", "secret-token")
        monkeypatch.setattr(mq, "cached_loader_fetch", lambda *, fetch, **kw: fetch())
        seen = {}

        def _side(url, **kwargs):
            seen["headers"] = kwargs.get("headers", {})
            return _Resp(200, _daily_payload())

        with patch.object(mq, "throttled_get", side_effect=_side):
            DataLoader().fetch(["600519.SH"], "2024-01-01", "2024-01-31", interval="1D")

        assert seen["headers"]["Authorization"] == "Bearer secret-token"


class TestCacheIdentity:
    def test_cache_key_uses_bridge_timeframe_and_fields_none(self, monkeypatch):
        captured = {}

        def _fake_cached(*, source, symbol, timeframe, start_date, end_date, fields, fetch):
            captured.update(
                source=source, symbol=symbol, timeframe=timeframe, fields=fields
            )
            return fetch()

        monkeypatch.setattr(mq, "cached_loader_fetch", _fake_cached)
        with patch.object(mq, "throttled_get", side_effect=_route()):
            DataLoader().fetch(["600519.sh"], "2024-01-01", "2024-01-31", interval="1D")

        assert captured["source"] == "miniqmt"
        assert captured["timeframe"] == "1d"  # bridge token, not the engine's "1D"
        assert captured["fields"] is None
        assert captured["symbol"] == "600519.SH"  # normalized to match the bridge


class TestParseBars:
    def test_sorts_ascending_and_indexes(self):
        df = _parse_bars(_daily_payload(), "2024-01-01", "2024-01-31")
        assert list(df.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
        assert df.index.name == "trade_date"

    def test_volume_is_float_and_passthrough(self):
        # The bridge already normalized volume to lots; the loader passes it
        # through unchanged (no double division).
        df = _parse_bars(_daily_payload(), "2024-01-01", "2024-01-31")
        assert df["volume"].dtype == float
        assert df["volume"].iloc[0] == 120.0

    def test_empty_bars_returns_none(self):
        assert _parse_bars({"bars": [], "unavailable": False}, "2024-01-01", "2024-01-31") is None

    def test_missing_index_column_returns_none(self):
        assert _parse_bars({"bars": [{"open": 1.0}]}, "2024-01-01", "2024-01-31") is None


class TestIntervalMap:
    def test_daily_and_hourly_mapping(self):
        assert _to_timeframe("1D") == "1d"
        assert _to_timeframe("1H") == "60m"
        assert _to_timeframe("1h") == "60m"
        assert _to_timeframe("5m") == "5m"
        assert _to_timeframe("tick") == "tick"

    def test_unsupported_interval(self):
        assert _to_timeframe("4H") is None
