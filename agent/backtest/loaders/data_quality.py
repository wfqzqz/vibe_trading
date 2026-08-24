"""A-share data-quality gates, landed as pure functions (DORA-124 §4.4).

The architecture contract fixes five data-quality gates and the provenance
envelope. This module lands the four gates that need a *decision function* as
pure, deterministic helpers so each is a runnable regression test in CI without
touching the network; the fifth gate (OHLC structure) is already wired at the
single fetch boundary in :func:`backtest.runner._sanitize_data_map` over
:func:`backtest.loaders.base.validate_ohlc`.

Gates:

1. **成交量单位门禁** — cross-source volume units (lots vs shares) must agree
   after normalization; a silent 100x jump (HKUDS/Vibe-Trading#1062) is a
   violation.
2. **复权门禁** — a source claiming 前复权 (qfq) must match a reference
   (MiniQMT复权序列 / adj_factor-derived) on ex-dividend dates, else the
   source is dropped for that symbol ("超阈弃用该源该标的").
3. **停牌/涨跌停语义门禁** — a suspended day is not a 0% move; the actual
   return semantics live in :func:`backtest.metrics.bar_returns` and
   :func:`backtest.engines.base._align` (verified by regression tests).
4. **跨源一致性门禁** — the same settled day's close agrees across sources
   within 1% (the upstream regression standard).
5. **OHLC 结构门禁** — ``base.validate_ohlc`` (high<low, non-positive prices).

Plus the provenance contract (§4.1): ``{source, volume_unit, adjust, is_final}``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

#: A-share board lot size (1 lot = 100 shares). Baostock and the QMT Bridge
#: normalize raw share-denominated volume to lots at ingestion (DORA-156 条件 1);
#: this constant is the canonical conversion factor for the volume-unit gate.
LOT_SIZE = 100.0

#: Cross-source close tolerance for gate 4 (DORA-124 §4.4: ≤1%).
DEFAULT_CLOSE_TOLERANCE = 0.01

#: Relative tolerance for volume-unit drift detection (gate 1).
VOLUME_UNIT_TOLERANCE = 0.02

#: Relative tolerance for the qfq adjustment check on ex-dates (gate 2).
ADJUST_TOLERANCE = 0.01

_VALID_VOLUME_UNITS = frozenset({"lots", "shares"})
_VALID_ADJUST = frozenset({"qfq", "hfq", "none"})

#: The four provenance fields the DORA-124 §4.1 envelope must always carry.
_REQUIRED_PROVENANCE = ("source", "volume_unit", "adjust", "is_final")


# ---------------------------------------------------------------------------
# Gate 1 — 成交量单位门禁 (lots vs shares, no 100x jump)
# ---------------------------------------------------------------------------


def check_volume_unit_consistency(
    samples: Mapping[str, tuple[str, float]],
    *,
    tolerance: float = VOLUME_UNIT_TOLERANCE,
) -> list[str]:
    """Detect cross-source volume-unit drift for one symbol/day (gate 1).

    Args:
        samples: Source name → ``(volume_unit, volume)`` for the *same* symbol
            and trading day. ``volume_unit`` is ``"lots"`` or ``"shares"``.
        tolerance: Maximum allowed relative deviation after normalizing every
            volume to a common basis (shares).

    Returns:
        A list of human-readable violations; empty means the gate passes.
        A source with an undeclared/unknown unit is a violation in itself; a
        non-positive or non-finite volume is skipped (it cannot be judged).
    """
    violations: list[str] = []
    normalized: dict[str, float] = {}
    for source in sorted(samples):
        unit, value = samples[source]
        unit_norm = (unit or "").strip().lower()
        if unit_norm not in _VALID_VOLUME_UNITS:
            violations.append(
                f"{source}: volume_unit {unit!r} is not lots|shares"
            )
            continue
        try:
            volume = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(volume) or volume <= 0:
            continue
        # Normalize both units to a single basis so a correct "lots vs shares"
        # pair (differing by exactly LOT_SIZE) reads as agreement.
        normalized[source] = volume * LOT_SIZE if unit_norm == "lots" else volume

    if len(normalized) < 2:
        return violations

    # Median baseline so a single drifted source (the historical 100x outlier)
    # is reported as the deviation, rather than blaming every correct source
    # against an arbitrary first sample.
    baseline = float(np.median(list(normalized.values())))
    for source in sorted(normalized):
        ratio = normalized[source] / baseline
        if abs(ratio - 1.0) > tolerance:
            violations.append(
                f"{source}: {ratio:.2f}x vs median after unit normalization "
                f"— unit drift? see #1062"
            )
    return violations


# ---------------------------------------------------------------------------
# Gate 2 — 复权门禁 (前复权 source must match reference on ex-dates)
# ---------------------------------------------------------------------------


def validate_qfq_adjustment(
    candidate_close: pd.Series,
    reference_close: pd.Series,
    ex_dates: Iterable[str],
    *,
    rel_tolerance: float = ADJUST_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Validate a claimed 前复权 (qfq) series against a reference (gate 2).

    A qfq series and its reference (MiniQMT 复权序列, or an adj_factor-derived
    series) agree by construction off ex-dates; the discriminating days are the
    ex-dividend dates, where an unadjusted or wrongly-adjusted series prints the
    mechanical price gap while a correct qfq series does not.

    Args:
        candidate_close: The source's close series (claimed qfq).
        reference_close: The reference close series.
        ex_dates: Ex-dividend dates (``YYYY-MM-DD`` or parseable).
        rel_tolerance: Maximum allowed relative close error on an ex-date.

    Returns:
        ``(passed, violations)`` — ``passed`` is ``False`` when any ex-date in
        both series deviates beyond ``rel_tolerance`` (the caller then drops the
        source for that symbol). Missing ex-dates in either series are skipped,
        not treated as evidence of a problem.
    """
    violations: list[str] = []
    if candidate_close.empty or reference_close.empty:
        return True, violations

    dates = pd.to_datetime(list(ex_dates), errors="coerce").dropna().normalize()
    if dates.empty:
        return True, violations

    cand = candidate_close[~candidate_close.index.duplicated(keep="last")]
    ref = reference_close[~reference_close.index.duplicated(keep="last")]
    cand.index = pd.DatetimeIndex(cand.index).normalize()
    ref.index = pd.DatetimeIndex(ref.index).normalize()

    for ex in dates:
        if ex not in cand.index or ex not in ref.index:
            continue
        candidate_value = float(cand.at[ex])
        reference_value = float(ref.at[ex])
        if not (np.isfinite(candidate_value) and np.isfinite(reference_value)):
            continue
        if reference_value <= 0:
            continue
        err = abs(candidate_value - reference_value) / abs(reference_value)
        if err > rel_tolerance:
            violations.append(
                f"{ex.date()}: candidate close {candidate_value:.4f} vs "
                f"reference {reference_value:.4f} ({err:.2%} > {rel_tolerance:.2%})"
            )
    return (not violations), violations


# ---------------------------------------------------------------------------
# Gate 3 — 停牌/涨跌停语义门禁 (suspension identification)
# ---------------------------------------------------------------------------


def suspension_days(frame: pd.DataFrame) -> pd.Series:
    """Flag suspended bars (gate 3): a bar with no traded volume.

    A fully suspended day usually has no bar at all (and is absent from the
    frame); a zero-volume bar — 停牌, or 涨跌停 with no liquidity — is what this
    flags so it is not mistaken for a real 0% trading move. Missing bars are
    left for the return/alignment layer (``bar_returns`` / ``_align``), which
    keeps long halts visible as NaN rather than a fabricated flat session.

    Args:
        frame: OHLCV frame indexed by ``trade_date``.

    Returns:
        A boolean Series aligned to ``frame``: ``True`` where the bar has no
        traded volume.
    """
    if "volume" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["volume"].fillna(0) <= 0


# ---------------------------------------------------------------------------
# Gate 4 — 跨源一致性门禁 (same settled-day close within 1%)
# ---------------------------------------------------------------------------


def check_cross_source_close(
    closes: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_CLOSE_TOLERANCE,
) -> list[str]:
    """Cross-source close consistency for one settled day (gate 4).

    Args:
        closes: Source name → the settled day's close price.
        tolerance: Maximum allowed relative deviation from the median close.

    Returns:
        A list of violations (source vs median, with the relative error);
        empty means every participating source agrees within ``tolerance``.
        Non-positive or non-finite closes are skipped.
    """
    violations: list[str] = []
    values: dict[str, float] = {}
    for source in sorted(closes):
        value = closes[source]
        try:
            close = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(close) or close <= 0:
            continue
        values[source] = close

    if len(values) < 2:
        return violations

    baseline = float(np.median(list(values.values())))
    for source, close in values.items():
        err = abs(close - baseline) / baseline
        if err > tolerance:
            violations.append(
                f"{source} close {close:.4f} vs median {baseline:.4f} "
                f"({err:.2%} > {tolerance:.2%})"
            )
    return violations


# ---------------------------------------------------------------------------
# Provenance (§4.1): {source, volume_unit, adjust, is_final}
# ---------------------------------------------------------------------------


def validate_provenance(provenance: Mapping[str, Any]) -> list[str]:
    """Validate a DORA-124 §4.1 provenance envelope.

    The envelope must carry the four fields that let a consumer state exactly
    what a number was produced from: ``source`` (actual source), ``volume_unit``
    (lots|shares), ``adjust`` (复权口径: qfq|hfq|none), and ``is_final`` (the
    range is settled). Extra fields (``symbol``, ``timeframe``) are permitted.

    Args:
        provenance: A provenance mapping.

    Returns:
        A list of violations; empty means the envelope is well-formed.
    """
    violations: list[str] = []
    for key in _REQUIRED_PROVENANCE:
        if key not in provenance:
            violations.append(f"missing provenance field {key!r}")

    source = provenance.get("source")
    if not isinstance(source, str) or not source.strip():
        violations.append("provenance.source must be a non-empty string")

    volume_unit = provenance.get("volume_unit")
    if volume_unit not in _VALID_VOLUME_UNITS:
        violations.append(
            f"provenance.volume_unit must be lots|shares, got {volume_unit!r}"
        )

    adjust = provenance.get("adjust")
    if adjust not in _VALID_ADJUST:
        violations.append(f"provenance.adjust must be qfq|hfq|none, got {adjust!r}")

    if not isinstance(provenance.get("is_final"), bool):
        violations.append("provenance.is_final must be a bool")

    return violations
