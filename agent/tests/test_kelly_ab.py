"""Tests for the K-05 reproducible A/B backtest harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.kelly_ab import (
    FRACTIONAL_GRID,
    ab_compare,
    kelly_exposure,
    pooled_pb,
    raw_signal,
    run_strategy,
    synthetic_market,
)


class TestSyntheticMarket:
    def test_deterministic_given_seed(self) -> None:
        a = synthetic_market(seed=7)
        b = synthetic_market(seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = synthetic_market(seed=7)
        b = synthetic_market(seed=8)
        assert not a.equals(b)

    def test_shape_and_columns(self) -> None:
        ret = synthetic_market(seed=7, n_bars=500, drift=[0.1, 0.0], vol=[0.2, 0.3])
        assert ret.shape == (500, 2)
        assert list(ret.columns) == ["A0", "A1"]

    def test_drift_vol_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            synthetic_market(drift=[0.1, 0.0], vol=[0.2])


class TestPooledPB:
    def test_pooled_stats(self) -> None:
        ret = pd.DataFrame(
            {"A": [0.02, -0.01, 0.03], "B": [0.01, 0.02, -0.01]},
        )
        win_rate, payoff, n = pooled_pb(ret, ["A", "B"])
        assert n == 6
        wins = [0.02, 0.03, 0.01, 0.02]
        losses = [0.01, 0.01]
        expected_wr = len(wins) / 6
        expected_payoff = np.mean(wins) / np.mean(losses)
        assert win_rate == pytest.approx(expected_wr)
        assert payoff == pytest.approx(expected_payoff)


class TestKellyExposure:
    def test_monotonic_in_fractional_c(self) -> None:
        e_low = kelly_exposure(0.55, 1.2, fractional_c=0.1)
        e_high = kelly_exposure(0.55, 1.2, fractional_c=0.5)
        assert e_high > e_low > 0.0

    def test_no_edge_is_zero(self) -> None:
        assert kelly_exposure(0.4, 1.0, fractional_c=0.25) == 0.0


class TestRunStrategy:
    def test_kelly_beats_baselines_in_default_market(self) -> None:
        """Regression lock for the headline K-05 finding (seed 7, defaults)."""
        ret = synthetic_market(seed=7, n_bars=600)
        pos = raw_signal(ret)
        dates = ret.index
        kwargs = dict(
            lookback=60, fractional_c=0.25, vol_target=None,
            cost_bps=10.0, periods_per_year=252,
        )
        one_n = run_strategy("1/N", ret, pos, dates, optimizer="1n", exposure=1.0, **kwargs)
        eq_vol = run_strategy(
            "equal_vol", ret, pos, dates, optimizer="equal_volatility", exposure=1.0, **kwargs
        )
        kelly = run_strategy("kelly", ret, pos, dates, optimizer="kelly", exposure=1.0, **kwargs)

        assert kelly.sharpe > one_n.sharpe
        assert kelly.sharpe > eq_vol.sharpe
        assert kelly.total_return > one_n.total_return
        # Kelly concentrates, so it trades more and drags more on cost.
        assert kelly.avg_turnover > one_n.avg_turnover
        assert kelly.cost_drag_return > one_n.cost_drag_return


class TestAbCompare:
    def test_structure_and_determinism(self) -> None:
        a = ab_compare(n_bars=600)
        b = ab_compare(n_bars=600)
        assert a == b

        assert set(a) == {"meta", "ab", "sensitivity"}
        assert len(a["ab"]) == 3
        assert [row["strategy"] for row in a["sensitivity"]] == ["kelly c=0.1", "kelly c=0.25", "kelly c=0.5"]

    def test_sensitivity_monotonic_in_fractional_c(self) -> None:
        result = ab_compare(n_bars=600)
        rows = {row["strategy"]: row for row in result["sensitivity"]}
        exposures = [rows[f"kelly c={c}"]["exposure"] for c in FRACTIONAL_GRID]
        returns = [rows[f"kelly c={c}"]["total_return"] for c in FRACTIONAL_GRID]
        # Higher fractional_c → higher exposure → higher (scaled) return.
        assert exposures == sorted(exposures)
        assert returns == sorted(returns)

    def test_vol_target_variant_added_when_requested(self) -> None:
        result = ab_compare(vol_target=0.10, n_bars=600)
        names = [row["strategy"] for row in result["ab"]]
        assert any("vol_target" in name for name in names)
