"""Regime-conditional Kelly selection (K-05).

The K-02 optimizer sizes Kelly from a per-symbol *return window* (binary
win/loss picture). K-03 added the evidence layer's ``(strategy, regime)``
``win_rate`` / ``payoff_ratio``. This module is the K-05 bridge between the two:
it decides, for one ``(strategy_id, regime)`` pair, whether the *evidence-backed*
Kelly sizing is authorized, and falls back to the baseline when it is not.

The authorization rule is deliberately strict and matches the epic's wording:

* evidence row exists for ``(strategy_id, regime)``;
* ``evidence_quality == adequate`` (never ``marginal`` / ``insufficient``);
* ``trades_in_regime >= min_trades`` (the sample-size threshold);
* both ``win_rate`` and ``payoff_ratio`` are present (not ``None``).

Every other state falls back to the baseline — fail-closed, never guessing a
p/b for a regime whose evidence cannot vouch for it. The returned
:class:`RegimeKellyDecision` carries a stable machine-readable ``reason`` so
callers and reports can cite *why* Kelly was refused, not just that it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.strategy_discovery.models import (
    QUALITY_ADEQUATE,
    QUALITY_ORDER,
    MIN_TRADES,
    EvidenceRow,
)

#: Stable reason tokens, surfaced on :class:`RegimeKellyDecision`.
REASON_ENABLED = "adequate"
REASON_NO_EVIDENCE = "no_evidence"
REASON_QUALITY_BELOW_ADEQUATE = "quality_below_adequate"
REASON_INSUFFICIENT_TRADES = "insufficient_trades"
REASON_MISSING_PB = "missing_pb"


@dataclass(frozen=True)
class RegimeKellyDecision:
    """The authorization verdict for evidence-backed Kelly sizing.

    ``enabled`` is the single gate callers branch on. When ``False``, ``reason``
    explains why (see the ``REASON_*`` tokens) and the p/b fields are ``None``.
    """

    strategy_id: str
    regime: str
    enabled: bool
    reason: str
    win_rate: float | None = None
    payoff_ratio: float | None = None
    trades_in_regime: int | None = None
    evidence_quality: str | None = None


def evaluate_regime_evidence(
    row: EvidenceRow,
    *,
    min_quality: str = QUALITY_ADEQUATE,
    min_trades: int = MIN_TRADES,
) -> RegimeKellyDecision:
    """Authorize one evidence row for Kelly sizing.

    Args:
        row: A single ``EvidenceRow`` for ``(strategy_id, regime)``.
        min_quality: Lowest acceptable ``evidence_quality`` by the
            ``QUALITY_ORDER`` ladder (default ``"adequate"`` — the top rung, so
            only exactly-adequate rows pass by default).
        min_trades: Minimum ``trades_in_regime`` for the sample-size gate.

    Returns:
        A :class:`RegimeKellyDecision`. Enabled only when quality meets the
        floor AND the sample size meets the threshold AND p/b are present.
    """
    if row is None:
        return RegimeKellyDecision(
            strategy_id="", regime="", enabled=False, reason=REASON_NO_EVIDENCE
        )

    quality_floor = QUALITY_ORDER.get(min_quality, QUALITY_ORDER[QUALITY_ADEQUATE])
    if QUALITY_ORDER.get(row.evidence_quality, 0) < quality_floor:
        return RegimeKellyDecision(
            strategy_id=row.strategy_id,
            regime=row.regime,
            enabled=False,
            reason=REASON_QUALITY_BELOW_ADEQUATE,
            trades_in_regime=row.trades_in_regime,
            evidence_quality=row.evidence_quality,
        )

    if row.trades_in_regime < min_trades:
        return RegimeKellyDecision(
            strategy_id=row.strategy_id,
            regime=row.regime,
            enabled=False,
            reason=REASON_INSUFFICIENT_TRADES,
            trades_in_regime=row.trades_in_regime,
            evidence_quality=row.evidence_quality,
        )

    if row.win_rate is None or row.payoff_ratio is None:
        return RegimeKellyDecision(
            strategy_id=row.strategy_id,
            regime=row.regime,
            enabled=False,
            reason=REASON_MISSING_PB,
            trades_in_regime=row.trades_in_regime,
            evidence_quality=row.evidence_quality,
        )

    return RegimeKellyDecision(
        strategy_id=row.strategy_id,
        regime=row.regime,
        enabled=True,
        reason=REASON_ENABLED,
        win_rate=row.win_rate,
        payoff_ratio=row.payoff_ratio,
        trades_in_regime=row.trades_in_regime,
        evidence_quality=row.evidence_quality,
    )


def regime_kelly_inputs(
    rows: Sequence[EvidenceRow],
    *,
    strategy_id: str,
    regime: str,
    min_quality: str = QUALITY_ADEQUATE,
    min_trades: int = MIN_TRADES,
) -> RegimeKellyDecision:
    """Select the matching evidence row and authorize it for Kelly sizing.

    Args:
        rows: Evidence rows (e.g. ``EvidenceStore.get_rows(strategy_id=..., regime=...)``).
        strategy_id: The strategy to size.
        regime: The regime to size under (one of ``REGIMES``).
        min_quality: Quality floor (default ``"adequate"``).
        min_trades: Sample-size threshold (default :data:`MIN_TRADES`).

    Returns:
        A disabled decision with reason ``"no_evidence"`` when no row matches,
        else :func:`evaluate_regime_evidence` applied to the first matching row.
    """
    matches = [
        row
        for row in rows
        if row is not None and row.strategy_id == strategy_id and row.regime == regime
    ]
    if not matches:
        return RegimeKellyDecision(
            strategy_id=strategy_id, regime=regime, enabled=False, reason=REASON_NO_EVIDENCE
        )
    # The store is keyed on (strategy_id, regime); a second row can only come
    # from a caller-supplied list, so take the first deterministically.
    return evaluate_regime_evidence(
        matches[0], min_quality=min_quality, min_trades=min_trades
    )


def apply_regime_conditional(
    decision: RegimeKellyDecision,
    kelly_weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Return the Kelly panel when authorized, else the baseline panel.

    This is the "fall back to baseline" boundary in one function: the caller
    computes both candidate weight panels, and this returns exactly one of
    them based on ``decision.enabled`` — no mixing, no partial application.

    Args:
        decision: The authorization verdict from :func:`regime_kelly_inputs`.
        kelly_weights: The evidence-backed Kelly weight panel.
        baseline_weights: The baseline weight panel (1/N or equal-volatility).

    Returns:
        ``kelly_weights`` when ``decision.enabled`` else ``baseline_weights``.
    """
    return kelly_weights if decision.enabled else baseline_weights


__all__ = [
    "REASON_ENABLED",
    "REASON_NO_EVIDENCE",
    "REASON_QUALITY_BELOW_ADEQUATE",
    "REASON_INSUFFICIENT_TRADES",
    "REASON_MISSING_PB",
    "RegimeKellyDecision",
    "evaluate_regime_evidence",
    "regime_kelly_inputs",
    "apply_regime_conditional",
]
