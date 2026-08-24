"""Bridge service — the business logic behind the read-only HTTP surface.

The service owns input validation, the fetch → metadata → cache pipeline, and
the JSON-safe response shape. It depends on a :class:`MarketDataProvider`
(``xtdata`` in production, a fake in tests) and never touches HTTP itself, so
it can be unit-tested directly.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from qmt_bridge.cache import SOURCE_NAME, write_frame
from qmt_bridge.config import Settings
from qmt_bridge.metadata import attach_metadata, normalize_symbol
from qmt_bridge.provenance import build_provenance
from qmt_bridge.xtdata_client import MarketDataProvider, XtdataUnavailableError

__all__ = ["BridgeService", "ValidationError"]

logger = logging.getLogger(__name__)

_VALID_ADJUST = {"qfq", "hfq", "none"}
_VALID_MINUTE_PERIODS = {"1m", "5m", "15m", "30m", "60m", "1h"}
#: The only 口径 the bridge persists to the loader cache (DORA-124 "前复权为主").
_CACHEABLE_ADJUST = "qfq"


class ValidationError(ValueError):
    """Raised for an invalid request parameter (maps to HTTP 422)."""


class BridgeService:
    """Read-only market-data service backed by a pluggable provider."""

    def __init__(
        self,
        provider: MarketDataProvider,
        settings: Settings | None = None,
        cache_root: str | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings or Settings()
        self._cache_root = cache_root or self._settings.cache_root

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _validate_range(start: str, end: str) -> tuple[str, str]:
        try:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
        except Exception as exc:
            raise ValidationError(f"invalid date range: {start!r}..{end!r}") from exc
        if start_ts > end_ts:
            raise ValidationError(f"start_date ({start}) > end_date ({end})")
        return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    @staticmethod
    def _validate_adjust(adjust: str | None, default: str) -> str:
        token = (adjust or default).strip().lower()
        if token not in _VALID_ADJUST:
            raise ValidationError(f"unsupported adjust {adjust!r}; use qfq/hfq/none")
        return token

    @staticmethod
    def _validate_symbol(symbol: str | None) -> str:
        if not symbol or not str(symbol).strip():
            raise ValidationError("symbol is required")
        norm = normalize_symbol(str(symbol))
        if "." not in norm:
            raise ValidationError(f"symbol {symbol!r} must include a venue suffix (e.g. 600519.SH)")
        return norm

    # -- public -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Report bridge + xtdata availability (the runtime probe)."""
        return {
            "status": "ok",
            "service": "qmt-bridge",
            "xtdata_available": self._provider.available(),
            "read_only": True,
        }

    def daily(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str | None = None,
    ) -> dict[str, Any]:
        """Fetch daily quotes and return records + provenance."""
        norm = self._validate_symbol(symbol)
        start, end = self._validate_range(start, end)
        adjust = self._validate_adjust(adjust, self._settings.adjust)
        return self._serve_quotes(norm, "1d", start, end, adjust)

    def minute(
        self,
        symbol: str,
        start: str,
        end: str,
        period: str = "1m",
        adjust: str | None = None,
    ) -> dict[str, Any]:
        """Fetch minute quotes and return records + provenance."""
        norm = self._validate_symbol(symbol)
        start, end = self._validate_range(start, end)
        adjust = self._validate_adjust(adjust, self._settings.adjust)
        if period not in _VALID_MINUTE_PERIODS:
            raise ValidationError(f"unsupported period {period!r}")
        return self._serve_quotes(norm, period, start, end, adjust)

    def tick(self, symbol: str, start: str, end: str) -> dict[str, Any]:
        """Fetch tick data (optional, tiered)."""
        norm = self._validate_symbol(symbol)
        start, end = self._validate_range(start, end)
        provenance = build_provenance(symbol=norm, timeframe="tick", adjust="none", end_date=end)
        if not self._provider.available():
            return {"provenance": provenance, "bars": [], "unavailable": True}
        try:
            frame = self._provider.tick(norm, start, end)
        except XtdataUnavailableError as exc:
            return {"provenance": provenance, "bars": [], "unavailable": True, "reason": str(exc)}
        if frame is None or frame.empty:
            return {"provenance": provenance, "bars": [], "unavailable": False}
        if self._settings.tick_persist:
            # Tick is tiered and volume-heavy: only short-lived persistence.
            self._cache(norm, "tick", start, end, "none", frame)
        return {"provenance": provenance, "bars": _frame_to_records(frame), "unavailable": False}

    def meta(self, symbol: str) -> dict[str, Any]:
        """Return 复权 / 停牌 / 涨跌停 / 除权除息 metadata."""
        norm = self._validate_symbol(symbol)
        if not self._provider.available():
            return {"symbol": norm, "unavailable": True}
        try:
            return self._provider.meta(norm)
        except XtdataUnavailableError as exc:
            return {"symbol": norm, "unavailable": True, "reason": str(exc)}

    # -- internals ----------------------------------------------------------

    def _serve_quotes(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjust: str,
    ) -> dict[str, Any]:
        provenance = build_provenance(
            symbol=symbol, timeframe=timeframe, adjust=adjust, end_date=end
        )
        if not self._provider.available():
            return {"provenance": provenance, "bars": [], "unavailable": True}

        try:
            if timeframe == "1d":
                bundle = self._provider.daily(symbol, start, end, adjust)
            else:
                bundle = self._provider.minute(symbol, timeframe, start, end, adjust)
        except XtdataUnavailableError as exc:
            return {"provenance": provenance, "bars": [], "unavailable": True, "reason": str(exc)}

        frame = bundle.frame
        if frame is None or frame.empty:
            return {"provenance": provenance, "bars": [], "unavailable": False}

        if timeframe == "1d":
            frame = attach_metadata(
                frame,
                symbol=symbol,
                adj_factor=bundle.adj_factor,
                ex_dividend_dates=bundle.ex_dividend_dates,
            )

        if adjust == _CACHEABLE_ADJUST:
            self._cache(symbol, timeframe, start, end, adjust, frame)

        return {"provenance": provenance, "bars": _frame_to_records(frame), "unavailable": False}

    def _cache(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjust: str,
        frame: pd.DataFrame,
    ) -> None:
        write_frame(
            source=SOURCE_NAME,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            # fields=None so the cache key matches the agent loader's
            # cached_loader_fetch(source="miniqmt", ..., fields=None) read.
            fields=None,
            frame=frame,
            cache_root=self._cache_root,
            extra_metadata={"adjust": adjust, "source": SOURCE_NAME},
        )


# ---------------------------------------------------------------------------
# JSON-safe serialization.
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Coerce a frame cell into a JSON-native value (mirrors ``market_data.py``).

    ``.isoformat()`` normalizes timestamps, ``.item()`` unwraps numpy scalars,
    non-finite floats become ``null``, and ``pd.NA`` becomes ``null``.
    """
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, int, str)):
        return value
    if value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index_value, row in frame.reset_index().iterrows():
        record: dict[str, Any] = {}
        for key, value in row.items():
            record[str(key)] = _json_safe(value)
        records.append(record)
    return records
