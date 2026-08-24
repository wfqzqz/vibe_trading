"""Unified response provenance for the QMT Bridge.

Every quote / meta response carries the same provenance envelope so a consumer
(``china_a.py`` via the ``miniqmt`` loader) can state exactly what a number
was produced from: the source, the symbol, the timeframe, the adjustment
口径, the volume unit, and whether the range is final (settled).

The shape is the DORA-124 §4.1 contract:
``{source, symbol, timeframe, adjust, volume_unit, is_final}``.
"""

from __future__ import annotations

from typing import Any, Mapping

from qmt_bridge.cache import range_is_final

__all__ = ["build_provenance", "SOURCE_NAME"]

#: The bridge is the sole writer; the loader reports this source name.
SOURCE_NAME = "miniqmt"

#: A-share volume is expressed in board lots (1 lot = 100 shares), matching
#: the agent's ``volume_units`` convention.
VOLUME_UNIT = "lots"


def build_provenance(
    *,
    symbol: str,
    timeframe: str,
    adjust: str,
    end_date: str,
    source: str = SOURCE_NAME,
    volume_unit: str = VOLUME_UNIT,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the unified provenance dict for one response.

    Args:
        symbol: The requested symbol (e.g. ``600519.SH``).
        timeframe: The timeframe token (e.g. ``1d``, ``1m``).
        adjust: The adjustment 口径 (``qfq`` / ``hfq`` / ``none``).
        end_date: The range end date, used to compute ``is_final``.
        source: Provenance source name (default ``"miniqmt"``).
        volume_unit: Provenance volume unit (default ``"lots"``).
        extra: Optional additional provenance keys (e.g. ``resolved_symbol``).

    Returns:
        A JSON-serializable provenance dict.
    """
    payload: dict[str, Any] = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "adjust": adjust,
        "volume_unit": volume_unit,
        "is_final": range_is_final(end_date),
    }
    if extra:
        payload.update(extra)
    return payload
