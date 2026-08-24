"""Tests for the Kelly position-sizing optimizer."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from backtest.optimizers.kelly import F_CAP, KellyOptimizer, kelly_fraction, optimize


# ---------------------------------------------------------------------------
# Pure function: kelly_fraction
# ---------------------------------------------------------------------------


class TestKellyFraction:
    """Unit tests for the binary Kelly fraction."""

    def test_positive_edge(self) -> None:
        # f* = 0.6 - 0.4/1.5 = 0.3333; fractional 0.25 → 0.08333
        assert kelly_fraction(0.6, 1.5) == pytest.approx(0.25 * (0.6 - 0.4 / 1.5))

    def test_zero_edge_returns_zero(self) -> None:
        # f* = 0.5 - 0.5/1.0 = 0
        assert kelly_fraction(0.5, 1.0) == 0.0

    def test_negative_edge_returns_zero(self) -> None:
        # f* = 0.4 - 0.6/1.0 = -0.2 → no bet
        assert kelly_fraction(0.4, 1.0) == 0.0

    def test_nonpositive_payoff_ratio_returns_zero(self) -> None:
        assert kelly_fraction(0.6, 0.0) == 0.0
        assert kelly_fraction(0.6, -2.0) == 0.0

    def test_fractional_discount_scales(self) -> None:
        f_star = 0.6 - 0.4 / 1.5
        assert kelly_fraction(0.6, 1.5, fractional_c=0.5) == pytest.approx(0.5 * f_star)
        assert kelly_fraction(0.6, 1.5, fractional_c=0.1) == pytest.approx(0.1 * f_star)

    def test_shrinkage_reduces_with_small_sample(self) -> None:
        base = 0.25 * (0.6 - 0.4 / 1.5)
        assert kelly_fraction(0.6, 1.5, n_trades=10, shrink_k=10) == pytest.approx(
            base * (10 / 20)
        )
        assert kelly_fraction(0.6, 1.5, n_trades=40, shrink_k=10) == pytest.approx(
            base * (40 / 50)
        )

    def test_none_trades_skips_shrinkage(self) -> None:
        assert kelly_fraction(0.6, 1.5, n_trades=None) == pytest.approx(
            0.25 * (0.6 - 0.4 / 1.5)
        )

    def test_zero_or_negative_trades_returns_zero(self) -> None:
        assert kelly_fraction(0.6, 1.5, n_trades=0) == 0.0
        assert kelly_fraction(0.6, 1.5, n_trades=-5) == 0.0

    def test_cap_truncates(self) -> None:
        # f* = 0.8 - 0.2/4.0 = 0.75; 0.5 * 0.75 = 0.375 → capped to F_CAP
        assert kelly_fraction(0.8, 4.0, fractional_c=0.5) == pytest.approx(F_CAP)
        assert kelly_fraction(0.8, 4.0, fractional_c=0.5) <= F_CAP

    def test_nan_win_rate_returns_zero(self) -> None:
        assert kelly_fraction(float("nan"), 1.5) == 0.0

    def test_nan_payoff_returns_zero(self) -> None:
        assert kelly_fraction(0.6, float("nan")) == 0.0

    def test_win_rate_out_of_range_returns_zero(self) -> None:
        assert kelly_fraction(1.5, 1.0) == 0.0
        assert kelly_fraction(-0.1, 1.0) == 0.0

    def test_infinite_payoff_uses_win_rate(self) -> None:
        # No losses → b = +inf → q/b = 0 → f* = p
        assert kelly_fraction(0.6, float("inf")) == pytest.approx(0.25 * 0.6)

    def test_certain_win_caps_at_fractional(self) -> None:
        # p = 1, q = 0 → f* = 1.0 → 0.25 * 1.0 = 0.25
        assert kelly_fraction(1.0, 2.0) == pytest.approx(0.25)

    def test_bool_input_rejected(self) -> None:
        assert kelly_fraction(True, 1.5) == 0.0  # type: ignore[arg-type]
        assert kelly_fraction(0.6, True) == 0.0  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "win_rate,payoff",
        [(0.55, 1.2), (0.7, 2.5), (0.9, 3.0), (0.6, 1.5)],
    )
    def test_result_stays_in_cap_range(self, win_rate: float, payoff: float) -> None:
        f = kelly_fraction(win_rate, payoff, n_trades=25)
        assert 0.0 <= f <= F_CAP


# ---------------------------------------------------------------------------
# KellyOptimizer._calc_weights
# ---------------------------------------------------------------------------


class TestKellyCalcWeights:
    """Unit tests for the per-symbol fraction computation."""

    def test_fractions_in_cap_range(self) -> None:
        ctx = {
            "win_rate": np.array([0.6, 0.7]),
            "payoff_ratio": np.array([1.5, 2.0]),
            "n_trades": np.array([60.0, 60.0]),
        }
        fractions = KellyOptimizer()._calc_weights(ctx)
        assert fractions.shape == (2,)
        assert np.all(fractions >= 0.0)
        assert np.all(fractions <= F_CAP)

    def test_no_edge_symbol_gets_zero_fraction(self) -> None:
        ctx = {
            "win_rate": np.array([0.6, 0.4]),
            "payoff_ratio": np.array([1.5, 1.0]),
            "n_trades": np.array([60.0, 60.0]),
        }
        fractions = KellyOptimizer()._calc_weights(ctx)
        assert fractions[0] > 0.0
        assert fractions[1] == 0.0

    def test_shrink_k_passthrough(self) -> None:
        ctx = {
            "win_rate": np.array([0.6]),
            "payoff_ratio": np.array([1.5]),
            "n_trades": np.array([10.0]),
        }
        base = KellyOptimizer(shrink_k=10.0)._calc_weights(ctx)
        tight = KellyOptimizer(shrink_k=100.0)._calc_weights(ctx)
        assert tight[0] < base[0]


# ---------------------------------------------------------------------------
# KellyOptimizer.optimize / module-level optimize
# ---------------------------------------------------------------------------


def _small_inputs() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """7 business days, two assets with a clear edge split."""
    dates = pd.bdate_range("2026-01-05", periods=7)
    ret = pd.DataFrame(
        {"A": [0.01] * 7, "B": [-0.01] * 7},
        index=dates,
    )
    pos = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    return dates, ret, pos


class TestKellyOptimize:
    """Integration tests through the optimizer's optimize()."""

    def test_preserves_sign_and_drops_no_edge(self) -> None:
        dates, ret, pos = _small_inputs()
        result = KellyOptimizer(lookback=6).optimize(ret, pos, dates)
        last = result.iloc[-1]
        # A has only wins → keeps the (long) weight; B has only losses → zeroed.
        assert last["A"] == pytest.approx(1.0)
        assert last["B"] == pytest.approx(0.0)

    def test_gross_normalized_to_one(self) -> None:
        dates, ret, pos = _small_inputs()
        result = KellyOptimizer(lookback=6).optimize(ret, pos, dates)
        assert result.iloc[-1].abs().sum() == pytest.approx(1.0)

    def test_higher_edge_gets_higher_weight(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=7)
        # A: 5 wins (+2%) / 1 loss (-1%) → strong edge.
        # B: 4 wins (+1%) / 2 losses (-1%) → weak edge.
        ret = pd.DataFrame(
            {
                "A": [0.02, 0.02, 0.02, 0.02, 0.02, -0.01, 0.0],
                "B": [0.01, 0.01, 0.01, 0.01, -0.01, -0.01, 0.0],
            },
            index=dates,
        )
        pos = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
        result = KellyOptimizer(lookback=6).optimize(ret, pos, dates)
        last = result.iloc[-1]
        assert last["A"] > last["B"] > 0.0

    def test_single_asset_unchanged(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=100)
        ret = pd.DataFrame(
            np.random.default_rng(1).normal(0, 0.02, (100, 1)),
            index=dates,
            columns=["A"],
        )
        pos = pd.DataFrame(1.0, index=dates, columns=["A"])
        result = optimize(ret, pos, dates, lookback=60)
        pd.testing.assert_frame_equal(result, pos)

    def test_decision_bar_return_cannot_change_decision_bar_weights(self) -> None:
        dates, ret, pos = _small_inputs()
        altered = ret.copy()
        altered.loc[dates[-1], ["A", "B"]] = [-100.0, 100.0]

        baseline = KellyOptimizer(lookback=6).optimize(ret, pos, dates)
        shocked = KellyOptimizer(lookback=6).optimize(altered, pos, dates)

        pd.testing.assert_series_equal(baseline.loc[dates[-1]], shocked.loc[dates[-1]])

    def test_all_nan_column_does_not_raise(self) -> None:
        dates, ret, pos = _small_inputs()
        ret["B"] = np.nan
        result = KellyOptimizer(lookback=6).optimize(ret, pos, dates)
        assert result.shape == pos.shape

    def test_module_level_optimize_entry(self) -> None:
        dates, ret, pos = _small_inputs()
        result = optimize(ret, pos, dates, lookback=6)
        assert result.shape == pos.shape
        assert result.iloc[-1]["B"] == pytest.approx(0.0)

    def test_module_dynamically_loadable(self) -> None:
        """The config path ``optimizer: "kelly"`` imports this module by name."""
        mod = importlib.import_module("backtest.optimizers.kelly")
        dates, ret, pos = _small_inputs()
        out = mod.optimize(ret, pos, dates, lookback=6)
        assert out.shape == pos.shape
        assert out.iloc[-1]["A"] == pytest.approx(1.0)
