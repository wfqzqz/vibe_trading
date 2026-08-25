"""A-share metadata columns for the QMT Bridge cache.

The bridge writes authoritative metadata *alongside* qfq-adjusted OHLCV so the
``china_a.py`` engine has the semantics it needs without re-deriving them from
free sources:

- ``pre_close``   — previous trading day's close (basis for limits / pct_chg),
- ``suspended``   — halted / 停牌 day (no volume),
- ``limit_up`` / ``limit_down`` — daily 涨跌停 band prices,
- ``ex_dividend`` — 除权除息 ex-date flag,
- ``adj_factor``  — 前复权 adjustment factor (when the source provides it).

All price-derived columns share the same qfq-adjusted basis as the OHLCV, so a
band computed here lines up with the bars ``china_a.py`` actually trades.

This module is pure (no ``xtquant`` import); the xtdata extraction lives in
:class:`qmt_bridge.xtdata_client.XtdataClient` and feeds these functions.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

__all__ = [
    "price_limit_ratio",
    "normalize_symbol",
    "compute_limit_bands",
    "mark_suspension",
    "mark_ex_dividend",
    "attach_metadata",
]


def normalize_symbol(symbol: str) -> str:
    """Return the canonical upper-case symbol (e.g. ``600519.SH``)."""
    return str(symbol).strip().upper()


def _board_code(symbol: str) -> str:
    """Extract the 6-digit code part from ``600519.SH`` / ``600519`` / ``sh600519``."""
    code = normalize_symbol(symbol).split(".")[0]
    if code.lower().startswith("sh") or code.lower().startswith("sz") or code.lower().startswith("bj"):
        code = code[2:]
    return code


def _is_bj(symbol: str) -> bool:
    """Return whether ``symbol`` trades on the Beijing Stock Exchange (北交所).

    The reliable signal is the ``.BJ`` venue suffix (real bridge symbols always
    carry it — ``920xxx.BJ``, ``830799.BJ``). Beijing-exchange codes also include
    the new ``92xxxx`` block and the older ``8xxxxx`` / ``4xxxxx`` (NEEQ select)
    blocks, so a suffix-less code is recognized by those ranges too. The old
    code-only ``startswith("8")`` heuristic silently missed ``920xxx.BJ`` and
    priced it at ±10% (DORA-156 条件 3).
    """
    norm = normalize_symbol(symbol)
    if norm.endswith(".BJ"):
        return True
    code = _board_code(norm)
    if len(code) != 6:
        return False
    # Beijing exchange code ranges: ``8xxxxx``, the new ``92xxxx`` block, and the
    # older ``4xxxxx`` (NEEQ select). Matches ``china_a._price_limit`` (DORA-156
    # 条件 2 / 条件 3), so the bridge's precomputed band and the engine's band
    # always agree for a Beijing symbol.
    return code.startswith(("43", "40", "8", "92"))


def price_limit_ratio(symbol: str) -> float:
    """Return the daily price-limit band as a fraction, mirroring ``china_a.py``.

    Args:
        symbol: A-share symbol.

    Returns:
        ``0.20`` for ChiNext (300xxx) / STAR (688xxx), ``0.30`` for Beijing
        (``.BJ``, 92xxxx / 8xxxxx), ``0.05`` for ST (best-effort, see below),
        else ``0.10``.
    """
    code = _board_code(normalize_symbol(symbol))
    # Beijing exchange: ±30% (DORA-156 条件 3 — covers 920xxx.BJ / 8xxxxx / 4xxxxx).
    if _is_bj(symbol):
        return 0.30
    # ChiNext (300xxx) / STAR (688xxx): ±20%
    if code.startswith("300") or code.startswith("688"):
        return 0.20
    # ST stocks trade ±5%: cannot be reliably detected from the code alone, so
    # this is a best-effort default the engine already mirrors (china_a.py).
    return 0.10


def compute_limit_bands(pre_close: pd.Series, symbol: str) -> tuple[pd.Series, pd.Series]:
    """Compute the daily 涨跌停 band prices from the adjusted ``pre_close``.

    A-share prices tick at 0.01 CNY, so bands are rounded to the nearest tick;
    the upper band rounds up, the lower band rounds down (the exchange's
    rounding rule), keeping fills just inside a lock.

    Args:
        pre_close: Previous-close series aligned to the OHLCV index.
        symbol: Symbol, used for the board ratio.

    Returns:
        ``(limit_up, limit_down)`` series.
    """
    ratio = price_limit_ratio(symbol)
    limit_up = (pre_close * (1.0 + ratio)).round(2)
    limit_down = (pre_close * (1.0 - ratio)).round(2)
    return limit_up, limit_down


def mark_suspension(frame: pd.DataFrame) -> pd.Series:
    """Flag 停牌 (halted) bars — a quoted bar with no traded volume.

    A fully suspended day typically has no bar at all (and is absent from the
    frame); a partially halted / zero-volume bar is what this flags. Missing
    bars are left for ``china_a.py`` to treat as 停牌 per DORA-124 D-04.
    """
    if "volume" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["volume"].fillna(0) <= 0


def mark_ex_dividend(index: pd.Index, ex_dates: Iterable[str] | None) -> pd.Series:
    """Flag 除权除息 ex-dates in the frame index.

    Args:
        index: The frame's DatetimeIndex.
        ex_dates: Iterable of ex-date strings (``YYYY-MM-DD``). ``None`` / empty
            yields all-``False``.

    Returns:
        A boolean series aligned to ``index``.
    """
    flags = pd.Series(False, index=index)
    if not ex_dates:
        return flags
    try:
        dates = pd.to_datetime(list(ex_dates), errors="coerce").dropna().normalize()
    except Exception:  # noqa: BLE001 - a malformed ex-date set degrades to no flags
        return flags
    if dates.empty:
        return flags
    normalized_index = pd.DatetimeIndex(index).normalize()
    flags = normalized_index.isin(dates)
    return pd.Series(flags, index=index)


def attach_metadata(
    frame: pd.DataFrame,
    *,
    symbol: str,
    pre_close: pd.Series | None = None,
    adj_factor: pd.Series | None = None,
    ex_dividend_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Attach the metadata columns to an OHLCV frame (returns a copy).

    Args:
        frame: OHLCV frame indexed by ``trade_date`` with ``open/high/low/close/
            volume`` columns.
        symbol: Symbol (for the limit-band ratio).
        pre_close: Previous-close series aligned to ``frame``'s index. When
            omitted, it is derived from the shifted ``close``.
        adj_factor: Optional 前复权 factor series.
        ex_dividend_dates: Optional ex-date strings.

    Returns:
        A new frame with ``pre_close`` / ``suspended`` / ``limit_up`` /
        ``limit_down`` / ``ex_dividend`` / ``adj_factor`` columns appended.
    """
    out = frame.copy()

    if pre_close is None:
        pre_close = out["close"].shift(1)
    out["pre_close"] = pre_close.reindex(out.index)

    out["suspended"] = mark_suspension(out)

    limit_up, limit_down = compute_limit_bands(out["pre_close"].fillna(out["close"]), symbol)
    out["limit_up"] = limit_up.reindex(out.index)
    out["limit_down"] = limit_down.reindex(out.index)

    out["ex_dividend"] = mark_ex_dividend(out.index, ex_dividend_dates)

    if adj_factor is not None:
        out["adj_factor"] = adj_factor.reindex(out.index)
    else:
        # Unknown factor: a float NaN column (parquet-safe, JSON-safe -> null).
        out["adj_factor"] = float("nan")

    return out
