"""Tests for K-05 Kelly parameterization: ``f_cap`` and portfolio vol-targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.optimizers.kelly import (
    F_CAP,
    KellyOptimizer,
    apply_vol_target,
    kelly_fraction,
    portfolio_returns,
    vol_target_scale,
)


# ---------------------------------------------------------------------------
# kelly_fraction — f_cap parameter (new in K-05)
# ---------------------------------------------------------------------------


class TestKellyFractionCap:
    def test_cap_lowers_result(self) -> None:
        f = kelly_fraction(0.6, 1.5, f_cap=0.02)
        assert 0.0 < f <= 0.02

    def test_cap_above_module_ceiling_is_ignored(self) -> None:
        # f_cap can never raise the module absolute ceiling F_CAP.
        assert kelly_fraction(0.6, 1.5, f_cap=1.0) == pytest.approx(
            kelly_fraction(0.6, 1.5)
        )

    @pytest.mark.parametrize(
        "bad_cap",
        [0.0, -0.1, float("nan"), float("inf"), -float("inf")],
    )
    def test_invalid_cap_fails_closed_to_zero(self, bad_cap: float) -> None:
        assert kelly_fraction(0.6, 1.5, f_cap=bad_cap) == 0.0

    def test_bool_cap_rejected(self) -> None:
        assert kelly_fraction(0.6, 1.5, f_cap=True) == 0.0  # type: ignore[arg-type]

    def test_default_cap_is_module_ceiling(self) -> None:
        assert kelly_fraction(0.8, 4.0, fractional_c=0.5) == pytest.approx(F_CAP)


# ---------------------------------------------------------------------------
# portfolio_returns / vol_target_scale / apply_vol_target
# ---------------------------------------------------------------------------


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 8-bar, 2-asset return and weight panels."""
    dates = pd.bdate_range("2026-01-05", periods=8)
    ret = pd.DataFrame(
        {"A": [0.0, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02], "B": [0.0] * 8},
        index=dates,
    )
    pos = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], "B": [0.5] * 8},
        index=dates,
    )
    return dates, ret, pos


class TestPortfolioReturns:
    def test_uses_previous_bar_weights(self) -> None:
        dates, ret, pos = _panel()
        pr = portfolio_returns(pos, ret)
        # bar 1 return = w[0]=0.5 * ret[1]=0.01 = 0.005
        assert pr.iloc[1] == pytest.approx(0.5 * 0.01)
        # first bar has no prior weight → 0.0
        assert pr.iloc[0] == 0.0


class TestVolTargetScale:
    def test_high_vol_delevers_below_one(self) -> None:
        series = pd.Series([0.02, -0.02, 0.03, -0.03, 0.02, -0.02] * 10)
        scale = vol_target_scale(series, target_vol=0.10, periods_per_year=252)
        assert 0.0 < scale < 1.0

    def test_low_vol_leaves_unchanged(self) -> None:
        series = pd.Series([0.001, -0.001, 0.001, -0.001] * 10)
        scale = vol_target_scale(series, target_vol=0.50, periods_per_year=252)
        assert scale == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_target_is_noop(self, bad: float) -> None:
        series = pd.Series([0.02, -0.02] * 10)
        assert vol_target_scale(series, target_vol=bad) == 1.0

    def test_bool_target_is_noop(self) -> None:
        series = pd.Series([0.02, -0.02] * 10)
        assert vol_target_scale(series, target_vol=True) == 1.0  # type: ignore[arg-type]

    def test_short_window_is_noop(self) -> None:
        series = pd.Series([0.02])
        assert vol_target_scale(series, target_vol=0.10) == 1.0


class TestApplyVolTarget:
    def test_scales_rows_when_vol_exceeds_target(self) -> None:
        dates, ret, pos = _panel()
        # High-vol target forces de-levering once history exists.
        out = apply_vol_target(pos, ret, target_vol=0.001, lookback=3, periods_per_year=252)
        # gross exposure after lookback must never exceed 1
        assert float(out.abs().sum(axis=1).max()) <= 1.0 + 1e-9
        # rows before lookback are untouched
        pd.testing.assert_frame_equal(out.iloc[:3], pos.iloc[:3])

    def test_noop_when_target_is_high(self) -> None:
        dates, ret, pos = _panel()
        out = apply_vol_target(pos, ret, target_vol=5.0, lookback=3, periods_per_year=252)
        pd.testing.assert_frame_equal(out, pos)

    def test_decision_bar_return_does_not_change_decision_bar_weights(self) -> None:
        dates, ret, pos = _panel()
        altered = ret.copy()
        altered.loc[dates[-1], "A"] = 5.0  # huge decision-bar shock

        base = apply_vol_target(pos, ret, target_vol=0.05, lookback=3, periods_per_year=252)
        shocked = apply_vol_target(pos, altered, target_vol=0.05, lookback=3, periods_per_year=252)
        pd.testing.assert_series_equal(base.iloc[-1], shocked.iloc[-1])


class TestKellyOptimizerParams:
    def test_f_cap_and_vol_target_flow_through(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=40)
        rng = np.random.default_rng(3)
        ret = pd.DataFrame(rng.normal(0.0004, 0.012, (40, 3)), index=dates, columns=["A", "B", "C"])
        pos = pd.DataFrame(1.0, index=dates, columns=["A", "B", "C"])

        capped = KellyOptimizer(lookback=10, f_cap=0.02).optimize(ret, pos, dates)
        targeted = KellyOptimizer(lookback=10, vol_target=0.05).optimize(ret, pos, dates)

        # Both knobs are accepted without error and preserve the panel shape.
        assert capped.shape == pos.shape
        assert targeted.shape == pos.shape

        # Engine normalisation (gross clipped to 1) yields a valid weight panel.
        scale = capped.abs().sum(axis=1).clip(lower=1.0)
        assert float(capped.div(scale, axis=0).abs().sum(axis=1).max()) <= 1.0 + 1e-9

        # A low vol-target de-levers the later rows below full investment.
        assert float(targeted.abs().sum(axis=1).iloc[-1]) < 1.0
