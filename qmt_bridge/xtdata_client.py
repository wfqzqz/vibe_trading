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

import datetime as dt
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

#: ``xtdata`` reports A-share volume already in board lots (1 lot = 100 shares).
#: This was verified empirically against the running miniQMT terminal
#: (DORA-156 条件 1): for every sampled symbol/day, ``amount / (volume * 100)``
#: lands on the average per-share price while ``amount / volume`` is ~100x the
#: price. So no scaling is applied at this ingestion boundary — the bridge (and
#: the whole A-share loader convention, ``backtest/loaders/base.py``) uses lots,
#: and the raw xtdata volume already is lots. `amount` is a money value (元) and
#: is likewise left untouched.

#: Symbol suffix → xtdata market token for the trading calendar. ``get_market_data_ex``
#: accepts ``.BJ`` Beijing-exchange instruments, but the calendar must be queried
#: with the exchange token — a `.BJ` symbol must never fall through to `SZ`
#: (DORA-156 条件 3).
_MARKET_BY_SUFFIX = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}

#: ``get_trading_dates`` returns millisecond-epoch timestamps; ``get_trading_calendar``
#: is not implemented on the miniQMT tunnel (it fails with ``function not realize``),
#: so the bridge uses ``get_trading_dates`` and converts explicitly (DORA-156 条件 3).
_EPOCH_MS = 1000.0


class XtdataUnavailableError(Exception):
    """Raised when ``xtquant.xtdata`` cannot be imported or the feed is down."""


@dataclass
class QuoteBundle:
    """One symbol's quotes plus the raw material for metadata columns.

    Attributes:
        frame: OHLCV frame indexed by ``trade_date`` with float
            ``open/high/low/close/volume`` (and optionally ``amount``) columns.
        adj_factor: Optional 前复权 factor series aligned to ``frame``'s index
            (built from ``get_divid_factors``'s ``dr`` per-event factor; sparse —
            present only on ex-dividend dates).
        ex_dividend_dates: Optional ``YYYY-MM-DD`` ex-date strings (where ``dr``
            differs from ``1.0``).
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
        # ``xtdata`` date arguments are ``YYYYMMDD`` tokens; the service (and the
        # HTTP contract) use ``YYYY-MM-DD``. Pass 8-digit tokens or every fetch
        # raises "起始时间错误" / ``TypeError: 'NoneType' object is not iterable``
        # (verified against the miniQMT terminal, DORA-156 条件 3).
        start_token, end_token = _to_xtdate(start), _to_xtdate(end)
        try:
            xtdata.download_history_data(symbol, period, start_token, end_token)
            data = xtdata.get_market_data_ex(
                ["open", "high", "low", "close", "volume", "amount"],
                [symbol],
                period,
                start_token,
                end_token,
                dividend_type=dividend_type,
                fill_data=True,
            )
        except Exception as exc:  # noqa: BLE001 - any feed failure degrades
            raise XtdataUnavailableError(f"xtdata fetch failed for {symbol}: {exc}") from exc

        frame = _coerce_market_data_frame(data, symbol)
        return frame

    def _dividend_factors(self, symbol: str, start: str, end: str) -> pd.Series | None:
        """Return the 除权除息 factor series (``dr``) for ``symbol``'s ex-dates."""
        xtdata = self._xtdata()
        try:
            factors = xtdata.get_divid_factors(symbol, _to_xtdate(start), _to_xtdate(end))
        except Exception:  # noqa: BLE001 - factors are best-effort metadata
            return None
        if factors is None or len(factors) == 0:
            return None
        return _coerce_factor_series(factors)

    def daily(self, symbol: str, start: str, end: str, adjust: str) -> QuoteBundle:
        """Fetch daily quotes (adjusted) plus the 除权除息 factor series.

        The 前复权/后复权 OHLCV is fetched directly from ``xtdata`` (its own
        ``dividend_type``). The ``dr`` factor series is best-effort metadata used
        to flag ``ex_dividend`` dates; it is only available when the provider is
        willing to return it.
        """
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
            # The suspension set is adjustment-independent; the underlying daily
            # frame here is fetched raw (``none``), so label it truthfully.
            "adjust": "none",
            "suspended_dates": suspended_dates,
            "ex_dividend_dates": daily.ex_dividend_dates,
            "adjust_factors": _factors_as_records(daily.adj_factor),
        }

    def _trading_calendar(self, symbol: str) -> list[str]:
        """Return the trading days in ``YYYY-MM-DD`` form for ``symbol``'s market.

        ``get_trading_calendar`` is not implemented on the miniQMT tunnel, so the
        bridge queries ``get_trading_dates`` (millisecond-epoch timestamps) and
        converts them. A ``.BJ`` symbol is mapped to the ``BJ`` market token; the
        old code defaulted every non-``SH`` symbol to ``SZ``, which silently
        served the wrong calendar for Beijing-exchange names (DORA-156 条件 3).
        """
        xtdata = self._xtdata()
        market = _MARKET_BY_SUFFIX.get(symbol[-3:].upper(), "SH")
        try:
            timestamps = xtdata.get_trading_dates(market, "20000101", "20991231")
        except Exception:  # noqa: BLE001 - an unusable calendar degrades to empty
            return []
        dates: list[str] = []
        for tt in timestamps or []:
            try:
                as_dt = dt.datetime.fromtimestamp(tt / _EPOCH_MS)
            except (TypeError, ValueError, OSError):
                continue
            dates.append(as_dt.strftime("%Y-%m-%d"))
        return sorted(set(dates))


# ---------------------------------------------------------------------------
# Helpers (pure, testable).
# ---------------------------------------------------------------------------


def _coerce_market_data_frame(data: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Map ``get_market_data_ex`` output to the bridge's OHLCV frame contract.

    The real ``get_market_data_ex`` shape is ``{symbol: DataFrame}`` — the value
    is a DataFrame indexed by the period's timestamp string (``YYYYMMDD`` daily,
    ``YYYYMMDDHHMMSS`` minute) with one column per requested field. The bridge
    originally assumed ``{field: DataFrame}`` (the legacy ``get_market_data``
    shape), which silently returned an empty frame for every real fetch
    (DORA-156 条件 3). ``volume`` arrives in board lots and is passed through
    unchanged (DORA-156 条件 1).
    """
    if not data:
        return _empty_ohlcv()
    raw_frame = data.get(symbol)
    if raw_frame is None:
        return _empty_ohlcv()
    try:
        frame = pd.DataFrame(raw_frame).copy()
    except Exception:  # noqa: BLE001 - a malformed value degrades to empty
        return _empty_ohlcv()

    # Normalize the index to the DT index; xtdata returns an object index of
    # period tokens (``YYYYMMDD`` daily, ``YYYYMMDDHHMMSS`` minute).
    frame.index = pd.to_datetime(frame.index.astype(str), errors="coerce")
    frame.index.name = "trade_date"

    # Coerce numeric columns; keep only rows that still carry an OHLC bar.
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keep_cols = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    frame = frame.dropna(subset=keep_cols)
    return frame.sort_index()


def _coerce_factor_series(factors: Any) -> pd.Series:
    """Coerce ``get_divid_factors`` output into a factor series.

    ``get_divid_factors`` returns a DataFrame indexed by ex-date with a ``dr``
    column (the 除权除息 factor per event). The previous heuristic picked
    ``factors.columns[0]`` (``time`` — a millisecond epoch) and produced a
    garbage factor series that flagged almost every day as an ex-date; ``dr`` is
    the correct factor column (DORA-156 条件 2).
    """
    if isinstance(factors, pd.Series):
        series = factors
    elif isinstance(factors, pd.DataFrame):
        if not factors.empty and "dr" in factors.columns:
            series = factors["dr"]
        elif "factor" in factors.columns:
            series = factors["factor"]
        else:
            return pd.Series(dtype=float)
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


def _to_xtdate(value: str) -> str:
    """Normalize a date to the 8-digit ``YYYYMMDD`` token ``xtdata`` expects.

    ``xtdata`` rejects dashed/slashed strings ("起始时间错误" / a TypeError from a
    None metadata payload), so a value like ``2024-02-01`` or ``2024/02/01`` is
    normalized to ``20240201``. An already-8-digit or unparseable value is
    returned unchanged (best-effort — xtdata surfaces the real error otherwise).
    """
    text = str(value).strip()
    try:
        return pd.Timestamp(text).strftime("%Y%m%d")
    except Exception:  # noqa: BLE001 - leave the token for xtdata to reject
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits if len(digits) == 8 else text


def _min_of(dates: list[str]) -> str:
    return min(dates) if dates else "20000101"


def _max_of(dates: list[str]) -> str:
    return max(dates) if dates else "20991231"


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "amount"],
        index=pd.DatetimeIndex([], name="trade_date"),
    )
