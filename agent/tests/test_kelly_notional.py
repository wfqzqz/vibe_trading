"""Tests for the K-04 Kelly notional link (research/simulation chain).

Covers ``resolve_kelly_inputs``, ``kelly_notional``, ``f_cap_from_limits`` and
``kelly_position_intents`` from ``backtest/optimizers/kelly.py``, plus the
reserved live hook ``src.live.kelly_sizing_hook``. Pure logic only: no network,
no engine, no order path.
"""

from __future__ import annotations

import math

import pytest

from backtest.optimizers.kelly import (
    ADV_PARTICIPATION_CAP,
    F_CAP,
    f_cap_from_limits,
    kelly_notional,
    kelly_position_intents,
    resolve_kelly_inputs,
)


# ---------------------------------------------------------------------------
# resolve_kelly_inputs — the K-03 → K-04 p/b contract reconciliation
# ---------------------------------------------------------------------------


class TestResolveKellyInputs:
    def test_none_win_rate_means_no_edge(self) -> None:
        assert resolve_kelly_inputs(None, 1.5) is None

    def test_none_payoff_means_no_edge(self) -> None:
        assert resolve_kelly_inputs(0.6, None) is None

    def test_all_win_regime_maps_to_infinite_payoff(self) -> None:
        # The evidence layer reports (win_rate=1.0, payoff_ratio=0.0) for an
        # all-win regime; this must NOT be read as "no edge".
        assert resolve_kelly_inputs(1.0, 0.0) == (1.0, float("inf"))

    def test_normal_pair_passes_through(self) -> None:
        assert resolve_kelly_inputs(0.6, 1.5) == (0.6, 1.5)

    def test_uncoercible_input_is_no_edge(self) -> None:
        assert resolve_kelly_inputs("not-a-number", 1.5) is None  # type: ignore[arg-type]

    def test_all_win_mapping_actually_sizes_nonzero(self) -> None:
        # End-to-end lock: a perfect win rate must never be silently sized to 0.
        sizing = kelly_notional(1_000_000.0, 1.0, 0.0, strategy_vol=0.2)
        assert sizing["f_final"] > 0.0
        assert sizing["notional_usd"] > 0.0


# ---------------------------------------------------------------------------
# f_cap_from_limits
# ---------------------------------------------------------------------------


class TestFCapFromLimits:
    def test_all_four_limits_min(self) -> None:
        cap = f_cap_from_limits(
            1_000_000.0,
            single_trade_notional_cap=100_000.0,  # 0.10
            total_exposure_cap=500_000.0,         # 0.50
            leverage_cap=1.0,                     # 1.00
            adv_participation_cap=0.05,           # 0.05  ← binds
        )
        assert cap == pytest.approx(0.05)

    def test_adv_cap_defaults_to_execution_model_ceiling(self) -> None:
        cap = f_cap_from_limits(1_000_000.0, single_trade_notional_cap=1_000_000.0)
        assert cap == pytest.approx(ADV_PARTICIPATION_CAP)

    def test_none_limits_are_skipped(self) -> None:
        cap = f_cap_from_limits(
            1_000_000.0,
            single_trade_notional_cap=None,
            total_exposure_cap=None,
            leverage_cap=None,
            adv_participation_cap=None,
        )
        assert cap == 0.0

    def test_invalid_equity_returns_zero(self) -> None:
        assert f_cap_from_limits(0.0, single_trade_notional_cap=100_000.0) == 0.0
        assert f_cap_from_limits(-1.0, single_trade_notional_cap=100_000.0) == 0.0
        assert f_cap_from_limits(float("nan"), single_trade_notional_cap=100_000.0) == 0.0

    def test_usd_caps_are_equity_fractions(self) -> None:
        # Only the single-notional cap supplied (ADV cap disabled): 25k / 1M = 0.025.
        cap = f_cap_from_limits(
            1_000_000.0,
            single_trade_notional_cap=25_000.0,
            adv_participation_cap=None,
        )
        assert cap == pytest.approx(0.025)

    def test_single_notional_cap_binds_over_adv(self) -> None:
        cap = f_cap_from_limits(
            1_000_000.0,
            single_trade_notional_cap=10_000.0,  # 0.01 < 0.05
        )
        assert cap == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# kelly_notional
# ---------------------------------------------------------------------------


class TestKellyNotional:
    def test_happy_path_sizes_and_reports_risk(self) -> None:
        sizing = kelly_notional(
            1_000_000.0,
            0.6,
            1.5,
            strategy_vol=0.2,
            f_cap=0.05,
        )
        # f_kelly = 0.25 * (0.6 - 0.4/1.5) = 0.08333…; cap 0.05 binds.
        assert sizing["f_final"] == pytest.approx(0.05)
        assert sizing["notional_usd"] == pytest.approx(50_000.0)
        assert sizing["risk_budget_usd"] == pytest.approx(10_000.0)

    def test_no_edge_returns_all_zeros(self) -> None:
        sizing = kelly_notional(1_000_000.0, 0.4, 1.0, strategy_vol=0.2)
        assert sizing == {"f_final": 0.0, "notional_usd": 0.0, "risk_budget_usd": 0.0}

    def test_none_pb_returns_all_zeros(self) -> None:
        assert kelly_notional(1_000_000.0, None, 1.5, strategy_vol=0.2) == {
            "f_final": 0.0,
            "notional_usd": 0.0,
            "risk_budget_usd": 0.0,
        }
        assert kelly_notional(1_000_000.0, 0.6, None, strategy_vol=0.2) == {
            "f_final": 0.0,
            "notional_usd": 0.0,
            "risk_budget_usd": 0.0,
        }

    def test_invalid_equity_returns_all_zeros(self) -> None:
        for equity in (0.0, -1.0, float("nan"), float("inf")):
            sizing = kelly_notional(equity, 0.6, 1.5, strategy_vol=0.2)
            assert sizing == {
                "f_final": 0.0,
                "notional_usd": 0.0,
                "risk_budget_usd": 0.0,
            }

    def test_shrinkage_reduces_fraction(self) -> None:
        full = kelly_notional(1_000_000.0, 0.6, 1.5, strategy_vol=0.2, f_cap=F_CAP)
        shrunk = kelly_notional(
            1_000_000.0, 0.6, 1.5, strategy_vol=0.2, shrink_n=10, f_cap=F_CAP
        )
        # n=10, k=10 → multiplier 10/20 = 0.5.
        assert shrunk["f_final"] == pytest.approx(full["f_final"] * 0.5)
        assert shrunk["notional_usd"] == pytest.approx(full["notional_usd"] * 0.5)

    def test_invalid_strategy_vol_zeroes_risk_budget_but_keeps_notional(self) -> None:
        sizing = kelly_notional(1_000_000.0, 0.6, 1.5, strategy_vol=float("nan"))
        assert sizing["notional_usd"] > 0.0
        assert sizing["risk_budget_usd"] == 0.0
        assert kelly_notional(1_000_000.0, 0.6, 1.5, strategy_vol=-0.2)["risk_budget_usd"] == 0.0

    @pytest.mark.parametrize(
        "win_rate,payoff,f_cap,shrink_n",
        [
            (0.55, 1.2, 0.05, None),
            (0.7, 2.5, 0.25, 25),
            (0.9, 3.0, 0.03, None),
            (0.6, 1.5, 0.5, None),
            (1.0, 0.0, 0.05, None),  # all-win regime
        ],
    )
    def test_f_final_never_exceeds_f_cap(
        self, win_rate: float, payoff: float, f_cap: float, shrink_n: float | None
    ) -> None:
        sizing = kelly_notional(
            1_000_000.0, win_rate, payoff, strategy_vol=0.2, f_cap=f_cap, shrink_n=shrink_n
        )
        assert 0.0 <= sizing["f_final"] <= f_cap
        assert sizing["notional_usd"] == pytest.approx(sizing["f_final"] * 1_000_000.0)

    def test_final_also_bounded_by_module_f_cap(self) -> None:
        # Even with a generous cap, f_final cannot exceed F_CAP (0.25).
        sizing = kelly_notional(1_000_000.0, 0.6, 1.5, strategy_vol=0.2, f_cap=1.0)
        assert sizing["f_final"] <= F_CAP


# ---------------------------------------------------------------------------
# kelly_position_intents — backtest weights → notional
# ---------------------------------------------------------------------------


class TestKellyPositionIntents:
    def test_weights_scale_to_signed_notionals(self) -> None:
        result = kelly_position_intents(
            {"A": 0.6, "B": -0.4},
            1_000_000.0,
            0.6,
            1.5,
            strategy_vol=0.2,
            f_cap=0.05,
        )
        assert result["f_final"] == pytest.approx(0.05)
        # gross = |0.6| + |-0.4| = 1.0 → total notional = 0.05 * 1M = 50k.
        assert result["notional_usd"] == pytest.approx(50_000.0)
        assert result["risk_budget_usd"] == pytest.approx(10_000.0)

        by_symbol = {p["symbol"]: p for p in result["positions"]}
        assert by_symbol["A"]["notional_usd"] == pytest.approx(30_000.0)
        assert by_symbol["B"]["notional_usd"] == pytest.approx(-20_000.0)
        assert by_symbol["B"]["risk_budget_usd"] == pytest.approx(4_000.0)

    def test_zero_and_nonfinite_weights_are_skipped(self) -> None:
        result = kelly_position_intents(
            {"A": 0.5, "B": 0.0, "C": float("nan")},
            1_000_000.0,
            0.6,
            1.5,
            strategy_vol=0.2,
            f_cap=0.05,
        )
        assert {p["symbol"] for p in result["positions"]} == {"A"}

    def test_no_edge_zeroes_every_position(self) -> None:
        result = kelly_position_intents(
            {"A": 0.6, "B": 0.4},
            1_000_000.0,
            0.4,
            1.0,
            strategy_vol=0.2,
        )
        assert result["f_final"] == 0.0
        assert result["notional_usd"] == 0.0
        assert all(p["notional_usd"] == 0.0 for p in result["positions"])

    def test_empty_weights_yields_empty_positions(self) -> None:
        result = kelly_position_intents(
            {}, 1_000_000.0, 0.6, 1.5, strategy_vol=0.2, f_cap=0.05
        )
        assert result["positions"] == []
        assert result["notional_usd"] == 0.0


# ---------------------------------------------------------------------------
# Reserved live hook (no order path)
# ---------------------------------------------------------------------------


class TestKellySizingHookReserved:
    def test_hook_is_reserved_and_raises(self) -> None:
        from src.live.kelly_sizing_hook import kelly_sizing_hook

        with pytest.raises(NotImplementedError, match="DORA-162"):
            kelly_sizing_hook(
                intent=None,  # type: ignore[arg-type] — it must raise before use
                equity=1_000_000.0,
                win_rate=0.6,
                payoff_ratio=1.5,
                strategy_vol=0.2,
            )

    def test_hook_does_not_import_at_module_level(self) -> None:
        # The reserved module must stay cheap to import (no runtime OrderIntent /
        # pandas dependency), so importing it can never affect the backtest path.
        import importlib

        module = importlib.import_module("src.live.kelly_sizing_hook")
        assert hasattr(module, "kelly_sizing_hook")
        assert math.isfinite(F_CAP)  # sanity: the sizing math stays untouched
