"""Tests for K-05 regime-conditional Kelly selection (evidence → baseline fallback)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.optimizers.kelly_regime import (
    REASON_ENABLED,
    REASON_INSUFFICIENT_TRADES,
    REASON_MISSING_PB,
    REASON_NO_EVIDENCE,
    REASON_QUALITY_BELOW_ADEQUATE,
    apply_regime_conditional,
    evaluate_regime_evidence,
    regime_kelly_inputs,
)
from src.strategy_discovery.models import (
    QUALITY_ADEQUATE,
    QUALITY_INSUFFICIENT,
    QUALITY_MARGINAL,
    MIN_TRADES,
    EvidenceRow,
)


def _row(
    *,
    strategy_id: str = "alpha_zoo:mom",
    regime: str = "bull_market",
    trades: int = 30,
    quality: str = QUALITY_ADEQUATE,
    win_rate: float | None = 0.6,
    payoff_ratio: float | None = 1.5,
) -> EvidenceRow:
    return EvidenceRow(
        strategy_id=strategy_id,
        regime=regime,
        trades_in_regime=trades,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        evidence_quality=quality,
        evidence_stage="backtest",
        provenance="test-run",
    )


class TestEvaluateRegimeEvidence:
    def test_adequate_and_sufficient_enables(self) -> None:
        decision = evaluate_regime_evidence(_row())
        assert decision.enabled is True
        assert decision.reason == REASON_ENABLED
        assert decision.win_rate == pytest.approx(0.6)
        assert decision.payoff_ratio == pytest.approx(1.5)

    @pytest.mark.parametrize("quality", [QUALITY_MARGINAL, QUALITY_INSUFFICIENT])
    def test_below_adequate_quality_disables(self, quality: str) -> None:
        decision = evaluate_regime_evidence(_row(quality=quality))
        assert decision.enabled is False
        assert decision.reason == REASON_QUALITY_BELOW_ADEQUATE
        assert decision.win_rate is None

    def test_insufficient_trades_disables(self) -> None:
        decision = evaluate_regime_evidence(_row(trades=MIN_TRADES - 1))
        assert decision.enabled is False
        assert decision.reason == REASON_INSUFFICIENT_TRADES

    @pytest.mark.parametrize(
        "win_rate,payoff_ratio",
        [(None, 1.5), (0.6, None), (None, None)],
    )
    def test_missing_pb_disables(self, win_rate, payoff_ratio) -> None:
        decision = evaluate_regime_evidence(
            _row(win_rate=win_rate, payoff_ratio=payoff_ratio)
        )
        assert decision.enabled is False
        assert decision.reason == REASON_MISSING_PB

    def test_configurable_min_trades(self) -> None:
        assert evaluate_regime_evidence(_row(trades=5), min_trades=5).enabled is True
        assert evaluate_regime_evidence(_row(trades=5), min_trades=6).enabled is False

    def test_none_row_fails_closed(self) -> None:
        decision = evaluate_regime_evidence(None)  # type: ignore[arg-type]
        assert decision.enabled is False
        assert decision.reason == REASON_NO_EVIDENCE


class TestRegimeKellyInputs:
    def test_no_matching_row_means_no_evidence(self) -> None:
        decision = regime_kelly_inputs(
            [_row(strategy_id="alpha_zoo:other")],
            strategy_id="alpha_zoo:mom",
            regime="bull_market",
        )
        assert decision.enabled is False
        assert decision.reason == REASON_NO_EVIDENCE

    def test_matching_row_delegates(self) -> None:
        decision = regime_kelly_inputs(
            [_row(strategy_id="alpha_zoo:mom", regime="bull_market")],
            strategy_id="alpha_zoo:mom",
            regime="bull_market",
        )
        assert decision.enabled is True
        assert decision.reason == REASON_ENABLED

    def test_regime_mismatch_means_no_evidence(self) -> None:
        decision = regime_kelly_inputs(
            [_row(strategy_id="alpha_zoo:mom", regime="bull_market")],
            strategy_id="alpha_zoo:mom",
            regime="bear_market",
        )
        assert decision.enabled is False
        assert decision.reason == REASON_NO_EVIDENCE


class TestApplyRegimeConditional:
    def test_enabled_returns_kelly_panel(self) -> None:
        kelly = pd.DataFrame({"A": [0.8], "B": [0.2]})
        baseline = pd.DataFrame({"A": [0.5], "B": [0.5]})
        decision = evaluate_regime_evidence(_row())
        out = apply_regime_conditional(decision, kelly, baseline)
        pd.testing.assert_frame_equal(out, kelly)

    def test_disabled_falls_back_to_baseline(self) -> None:
        kelly = pd.DataFrame({"A": [0.8], "B": [0.2]})
        baseline = pd.DataFrame({"A": [0.5], "B": [0.5]})
        decision = evaluate_regime_evidence(_row(quality=QUALITY_INSUFFICIENT))
        out = apply_regime_conditional(decision, kelly, baseline)
        pd.testing.assert_frame_equal(out, baseline)
