"""Lazy, read-only ``xtquant.xtdata`` client for the QMT Bridge.

This is the **only** place in the bridge that may touch ``xtquant``, and it
imports just ``xtquant.xtdata`` — never ``xtquant.xttrader`` (there is no
trade / write surface in the bridge at all; see ``capabilities.py``). The
import is lazy (inside functions), so the bridge and its tests run anywhere,
and a missing ``xtquant`` degrades to an explicit
:class:`XtdataUnavailableError` instead of an import-time crash.

The bridge depends on the :class:`MarketDataProvider` protocol, not on this
class, so unit tests inject a fake provider and never require a real QMT
terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

__all__ = [
    "QuoteBundle",
    "MarketDataProvider",
    "XtdataClient",
    "XtdataUnavailableError",
    "is_xtdata_available",
]

#: dividend_type token per bridge adjustment 口径.
_ADJUST_TO_DIVIDEND_TYPE = {"qfq": "front", "hfq": "back", "none": "none"}

#: A-share board-lot size: 1 lot = 100 shares. ``xtdata`` reports volume in
#: single shares (股), while the bridge's provenance — and the whole A-share
#: loader convention (backtest/loaders/base.py, HKUDS/Vibe-Trading#1062) — uses
#: board lots. Normalizing shares → lots at this ingestion boundary keeps the
#: cache and HTTP bars on the same "lots" unit so D-04 门禁 1 sees no 100x jump.
#: (DORA-156 条件 1; the ÷100 factor is re-verified empirically by D-05.)
_VOLUME_SHARES_PER_LOT = 100.0


class XtdataUnavailableError(Exception):
    """Raised when ``xtquant.xtdata`` cannot be imported or the feed is down."""


@dataclass
class QuoteBundle:
    """One symbol's quotes plus the raw material for metadata columns.

    Attributes:
        frame: OHLCV frame indexed by ``trade_date`` with float
            ``open/high/low/close/volume`` (and optionally ``amount``) columns.
        adj_factor: Optional 前复权 factor series aligned to ``frame``'s index.
        ex_dividend_dates: Optional ``YYYY-MM-DD`` ex-date strings.
    """

    frame: pd.DataFrame
    adj_factor: pd.Series | None = None
    ex_dividend_dates: list[str] = field(default_factory=list)


class MarketDataProvider(Protocol):
    """Read-only market-data source the bridge service depends on."""

    def available(self) -> bool:
        """Return whether the underlying feed is usable right now."""
        ...

    def daily(self, symbol: str, start: str, end: str, adjust: str) -> QuoteBundle:
        """Fetch daily quotes for ``symbol`` in ``[start, end]``."""
        ...

    def minute(self, symbol: str, period: str, start: str, end: str, adjust: str) -> QuoteBundle:
        """Fetch minute quotes for ``symbol`` in ``[start, end]``."""
        ...

    def tick(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """Fetch tick data (optional, tiered) — ``None`` when unsupported."""
        ...

    def meta(self, symbol: str) -> dict[str, Any]:
        """Fetch 复权 / 停牌 / 涨跌停 / 除权除息 metadata for ``symbol``."""
        ...


def is_xtdata_available() -> bool:
    """Return whether ``xtquant.xtdata`` is importable (no terminal attach)."""
    try:
        import xtquant.xtdata  # noqa: F401  (import-only availability probe)
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


class XtdataClient:
    """Thin read-only adapter over ``xtquant.xtdata``.

    ``xtquant`` is imported lazily inside each call; every method raises
    :class:`XtdataUnavailableError` when the SDK is absent or the call fails,
    so the service can degrade to a structured "unavailable" response.
    """

    _MINUTE_PERIODS = {"1m", "5m", "15m", "30m", "60m", "1h"}

    @staticmethod
    def _xtdata() -> Any:
        try:
            import xtquant.xtdata as xtdata
        except Exception as exc:  # noqa: BLE001 - surface one clear error
            raise XtdataUnavailableError(
                "xtquant.xtdata is not importable; install miniQMT's xtquant "
                "package on this Windows host."
            ) from exc
        return xtdata

    def available(self) -> bool:
        """Return whether ``xtquant.xtdata`` is importable (the runtime probe).

        Terminal connectivity is probed on the first real fetch, not here, so
        the health endpoint can distinguish "SDK present" from "feed up".
        """
        return is_xtdata_available()

    def _market_data(self, symbol: str, period: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        xtdata = self._xtdata()
        dividend_type = _ADJUST_TO_DIVIDEND_TYPE.get(adjust, "front")
        try:
            xtdata.download_history_data(symbol, period, start, end)
            data = xtdata.get_market_data_ex(
                ["open", "high", "low", "close", "volume", "amount"],
                [symbol],
                period,
                start,
                end,
                dividend_type=dividend_type,
                fill_data=True,
            )
        except Exception as exc:  # noqa: BLE001 - any feed failure degrades
            raise XtdataUnavailableError(f"xtdata fetch failed for {symbol}: {exc}") from exc

        frame = _coerce_market_data_frame(data, symbol)
        return frame

    def _dividend_factors(self, symbol: str, start: str, end: str) -> pd.Series | None:
        xtdata = self._xtdata()
        try:
            factors = xtdata.get_divid_factors(symbol, start, end)
        except Exception:  # noqa: BLE001 - factors are best-effort metadata
            return None
        if factors is None or len(factors) == 0:
            return None
        return _coerce_factor_series(factors)

    def daily(self, symbol: str, start: str, end: str, adjust: str) -> QuoteBundle:
        """Fetch daily quotes (adjusted) plus dividend factors."""
        frame = self._market_data(symbol, "1d", start, end, adjust)
        factors = self._dividend_factors(symbol, start, end)
        ex_dates: list[str] = []
        if factors is not None:
            ex_dates = _ex_dates_from_factors(factors)
        return QuoteBundle(frame=frame, adj_factor=factors, ex_dividend_dates=ex_dates)

    def minute(self, symbol: str, period: str, start: str, end: str, adjust: str) -> QuoteBundle:
        """Fetch minute quotes (adjusted)."""
        if period not in self._MINUTE_PERIODS:
            raise ValueError(f"unsupported minute period: {period!r}")
        frame = self._market_data(symbol, period, start, end, adjust)
        return QuoteBundle(frame=frame)

    def tick(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """Fetch tick data (optional). Returns ``None`` on any failure."""
        try:
            return self._market_data(symbol, "tick", start, end, "none")
        except XtdataUnavailableError:
            return None

    def meta(self, symbol: str) -> dict[str, Any]:
        """Return 复权 / 停牌 / 涨跌停 / 除权除息 metadata for ``symbol``.

        ``xtdata`` does not expose a single "停牌 list" call; 停牌 days are
        derived as trading-calendar days missing from the daily frame. The
        price-limit ratio is board-derived (see ``metadata.py``).
        """
        from qmt_bridge.metadata import price_limit_ratio, normalize_symbol

        norm = normalize_symbol(symbol)
        calendar = self._trading_calendar(norm)
        daily = self.daily(norm, _min_of(calendar), _max_of(calendar), "none")

        traded_days = {ts.strftime("%Y-%m-%d") for ts in daily.frame.index}
        suspended_dates = [d for d in calendar if d not in traded_days]

        return {
            "symbol": norm,
            "price_limit_ratio": price_limit_ratio(norm),
            "adjust": "qfq",
            "suspended_dates": suspended_dates,
            "ex_dividend_dates": daily.ex_dividend_dates,
            "adjust_factors": _factors_as_records(daily.adj_factor),
        }

    def _trading_calendar(self, symbol: str) -> list[str]:
        xtdata = self._xtdata()
        market = "SH" if symbol.endswith(".SH") else "SZ"
        try:
            days = xtdata.get_trading_calendar(market, "20000101", "20991231")
        except Exception:  # noqa: BLE001
            return []
        return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in (days or [])]


# ---------------------------------------------------------------------------
# Helpers (pure, testable).
# ---------------------------------------------------------------------------


def _coerce_market_data_frame(data: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Map ``get_market_data_ex`` output to the bridge's OHLCV frame contract."""
    if not data:
        return _empty_ohlcv()
    frame = pd.DataFrame(index=None)
    pieces = []
    for field in ("open", "high", "low", "close", "volume", "amount"):
        series = data.get(field)
        if series is None:
            continue
        values = series.get(symbol) if isinstance(series, dict) else None
        if values is None:
            try:
                values = series[symbol] if symbol in getattr(series, "columns", ()) else None
            except Exception:  # noqa: BLE001
                values = None
        if values is not None:
            pieces.append(pd.Series(values, name=field))
    if not pieces:
        return _empty_ohlcv()
    frame = pd.concat(pieces, axis=1)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.dropna(subset=frame.columns.intersection(["open", "high", "low", "close"]))
    frame.index.name = "trade_date"
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    # xtdata reports volume in single shares (股); normalize to board lots
    # (1 lot = 100 shares) to match the bridge's provenance unit and every other
    # A-share source (see module docstring / DORA-156 条件 1). ``amount`` is a
    # money value (元) and is left untouched.
    if "volume" in frame.columns:
        frame["volume"] = frame["volume"] / _VOLUME_SHARES_PER_LOT
    return frame.sort_index()


def _coerce_factor_series(factors: Any) -> pd.Series:
    if isinstance(factors, pd.Series):
        series = factors
    elif isinstance(factors, pd.DataFrame):
        col = "factor" if "factor" in factors.columns else factors.columns[0]
        series = factors[col]
    else:
        return pd.Series(dtype=float)
    series.index = pd.to_datetime(series.index, errors="coerce")
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def _ex_dates_from_factors(factors: pd.Series) -> list[str]:
    """Return the ex-dates where a dividend factor differs from 1.0."""
    if factors is None or factors.empty:
        return []
    changed = factors[factors.round(6) != 1.0]
    return [ts.strftime("%Y-%m-%d") for ts in changed.index]


def _factors_as_records(factors: pd.Series | None) -> list[dict[str, str]]:
    if factors is None or factors.empty:
        return []
    return [
        {"date": ts.strftime("%Y-%m-%d"), "factor": float(value)}
        for ts, value in factors.items()
    ]


def _min_of(dates: list[str]) -> str:
    return min(dates) if dates else "20000101"


def _max_of(dates: list[str]) -> str:
    return max(dates) if dates else "20991231"


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "amount"],
        index=pd.DatetimeIndex([], name="trade_date"),
    )
