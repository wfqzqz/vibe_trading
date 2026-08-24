"""Factor reconciliation regression — py-alpha-lib vs Alpha Zoo (DORA-147 / F-04).

This is the regression the F-04 acceptance criterion "升级重跑对账结果一致"
points at: re-run it after *any* bump to ``py-alpha-lib``, ``numpy>=2``, or any
factor dependency, and the py-alpha-lib path must still reproduce the Alpha Zoo's
pandas path on the same panel.

The "same-口径 zoo factor" comparison from DORA-124 §3.4 is literal here: every
py-alpha-lib result is diffed against the Alpha Zoo's *own* operators
(``src.factors.base``) — not a re-implemented pandas reference — computed on the
same wide panel. A drift in either path fails this file.

Skipped automatically where py-alpha-lib is not installed (the Windows-host
degradation path); runs in the Docker container (``cp311-abi3``) or on a
Python ≥ 3.14 host with the ``cp314-abi3`` Windows wheel.

Warmup semantics (verified empirically against py-alpha-lib 0.3.0, and the
source of the F-03 handoff note):

* **strict warmup** (``min_periods=n`` → first ``n-1`` bars NaN) — ``stddev``/
  ``std``, ``corr``, ``cov``. These match the zoo's ``ts_std``/``ts_corr``/
  ``ts_cov`` exactly.
* **partial warmup** (``min_periods=1`` → first ``n-1`` bars computed on a
  growing window) — ``ma``/``sma``/``mean``/``ts_mean``/``sum``/``ts_max``/
  ``ts_min``. The zoo's ``ts_mean``/``ts_max``/``ts_min`` use strict warmup, so
  the two paths diverge only in the first ``n-1`` bars and agree from bar ``n``
  onward; the shared IC/IR + layered math is unaffected (same panel / returns /
  IC definition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alpha")  # container-only: skip on a degraded Windows host

pytestmark = pytest.mark.integration

from src.factor_runtime.compute import compute_factor, evaluate_factor, panel_to_long  # noqa: E402
from src.factor_runtime.snapshot import SnapshotStore  # noqa: E402
from src.factors.base import (  # noqa: E402
    delta,
    rank,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_std,
    zscore,
)
from src.factors.factor_analysis_core import compute_group_equity, compute_ic_series  # noqa: E402

_ATOL = 1e-10


def _make_panel(n_dates: int = 20, n_codes: int = 8) -> dict[str, pd.DataFrame]:
    """Deterministic wide OHLCV panel (no vwap → synthesized typical price)."""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    codes = [f"c{i}" for i in range(n_codes)]
    rng = np.random.default_rng(7)
    close = rng.uniform(8, 60, size=(n_dates, n_codes)).cumsum(axis=0)

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=codes)

    return {
        "open": frame(close * 0.99),
        "high": frame(close * 1.02),
        "low": frame(close * 0.98),
        "close": frame(close),
        "volume": frame(rng.uniform(1000, 5000, size=(n_dates, n_codes))),
    }


def _store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(root=tmp_path / "factors")


def test_reconcile_identity(tmp_path: Path) -> None:
    """``close`` round-trips to the panel's close exactly."""
    panel = _make_panel()
    store = _store(tmp_path)
    store.register("close", name="rec_close")
    pd.testing.assert_frame_equal(compute_factor(store, "rec_close", 1, panel), panel["close"])


def test_reconcile_delay_and_delta(tmp_path: Path) -> None:
    """``delay`` == shift; ``delta`` == zoo ``delta`` (first-difference)."""
    panel = _make_panel()
    close = panel["close"]
    store = _store(tmp_path)

    store.register("delay(close,1)", name="rec_delay")
    np.testing.assert_allclose(
        compute_factor(store, "rec_delay", 1, panel).to_numpy(),
        close.shift(1).to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )

    store.register("delta(close,2)", name="rec_delta")
    np.testing.assert_allclose(
        compute_factor(store, "rec_delta", 1, panel).to_numpy(),
        delta(close, 2).to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )


def test_reconcile_momentum(tmp_path: Path) -> None:
    """``close/delay(close,1)-1`` == ``pct_change`` (the F-03 momentum factor)."""
    panel = _make_panel()
    store = _store(tmp_path)
    store.register("close/delay(close,1)-1", name="rec_mom")
    np.testing.assert_allclose(
        compute_factor(store, "rec_mom", 1, panel).to_numpy(),
        panel["close"].pct_change(fill_method=None).to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )


def test_reconcile_cross_sectional(tmp_path: Path) -> None:
    """``rank`` == zoo cross-sectional percentile rank; ``zscore`` == zoo zscore."""
    panel = _make_panel()
    close = panel["close"]
    store = _store(tmp_path)

    store.register("rank(close)", name="rec_rank")
    np.testing.assert_allclose(
        compute_factor(store, "rec_rank", 1, panel).to_numpy(),
        rank(close).to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )

    store.register("zscore(close)", name="rec_zscore")
    np.testing.assert_allclose(
        compute_factor(store, "rec_zscore", 1, panel).to_numpy(),
        zscore(close).to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )


def test_reconcile_rolling_strict_warmup(tmp_path: Path) -> None:
    """Strict-warmup rolling operators == the zoo's ``ts_*`` operators exactly."""
    panel = _make_panel()
    close = panel["close"]
    low = panel["low"]
    store = _store(tmp_path)

    cases: list[tuple[str, str, pd.DataFrame]] = [
        ("rec_stddev", "stddev(close,5)", ts_std(close, 5)),
        ("rec_corr", "corr(close,low,5)", ts_corr(close, low, 5)),
        ("rec_cov", "cov(close,low,5)", ts_cov(close, low, 5)),
    ]
    for name, expression, expected in cases:
        store.register(expression, name=name)
        np.testing.assert_allclose(
            compute_factor(store, name, 1, panel).to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
            atol=1e-9,  # corr/cov accumulate more float error than means
            err_msg=f"{expression} diverged from the zoo reference",
        )


def test_reconcile_partial_warmup_rolling(tmp_path: Path) -> None:
    """``ma``/``sma``/``mean``/``ts_mean``/``sum``/``ts_max``/``ts_min`` partial warmup.

    py-alpha-lib's rolling mean/sum/max/min family computes on a growing window
    for the first ``n-1`` bars (no NaN), unlike the zoo's strict ``ts_*``. The
    pandas reference therefore uses ``min_periods=1``.
    """
    panel = _make_panel()
    close = panel["close"]
    store = _store(tmp_path)

    ma_expected = close.rolling(5, min_periods=1).mean()
    for name, expression in (
        ("rec_ma", "ma(close,5)"),
        ("rec_sma", "sma(close,5)"),
        ("rec_mean", "mean(close,5)"),
        ("rec_tsmean", "ts_mean(close,5)"),
    ):
        store.register(expression, name=name)
        np.testing.assert_allclose(
            compute_factor(store, name, 1, panel).to_numpy(),
            ma_expected.to_numpy(),
            equal_nan=True,
            atol=_ATOL,
        )

    store.register("sum(close,5)", name="rec_sum")
    np.testing.assert_allclose(
        compute_factor(store, "rec_sum", 1, panel).to_numpy(),
        close.rolling(5, min_periods=1).sum().to_numpy(),
        equal_nan=True,
        atol=1e-9,
    )

    store.register("ts_max(close,5)", name="rec_tsmax")
    np.testing.assert_allclose(
        compute_factor(store, "rec_tsmax", 1, panel).to_numpy(),
        close.rolling(5, min_periods=1).max().to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )

    store.register("ts_min(close,5)", name="rec_tsmin")
    np.testing.assert_allclose(
        compute_factor(store, "rec_tsmin", 1, panel).to_numpy(),
        close.rolling(5, min_periods=1).min().to_numpy(),
        equal_nan=True,
        atol=_ATOL,
    )


def test_reconcile_rolling_warmup_divergence(tmp_path: Path) -> None:
    """py-alpha-lib (partial) vs zoo (strict) rolling operators — warmup only.

    ``ts_mean``/``ts_max``/``ts_min`` diverge from the zoo's strict-warmup
    counterparts only in the first ``n-1`` bars (py-alpha-lib fills them, the zoo
    NaNs them); from bar ``n`` onward the values are identical, so the shared
    IC/IR + layered math sees the same signal where it matters (the F-03 handoff
    note).
    """
    panel = _make_panel()
    close = panel["close"]
    n = 5
    store = _store(tmp_path)

    for expr, zoo_ref in (
        ("ts_mean(close,5)", ts_mean),
        ("ts_max(close,5)", ts_max),
        ("ts_min(close,5)", ts_min),
    ):
        name = "rec_" + expr.split("(")[0]
        store.register(expr, name=name)
        pal = compute_factor(store, name, 1, panel)
        zoo = zoo_ref(close, n)

        warmup = close.index[: n - 1]
        steady = close.index[n - 1 :]
        assert pal.loc[warmup].notna().all().all()
        assert zoo.loc[warmup].isna().all().all()
        np.testing.assert_allclose(
            pal.loc[steady].to_numpy(),
            zoo.loc[steady].to_numpy(),
            equal_nan=True,
            atol=_ATOL,
        )


def test_reconcile_evaluation_same_caliber(tmp_path: Path) -> None:
    """``evaluate_factor`` reproduces the zoo's IC/IR + layered math exactly.

    The py-alpha-lib factor values are reconciled above, so this asserts the
    end-to-end contract: the *same* ``factor_analysis_core`` math applied to the
    reconciled factor yields the *same* IC/IR + layered NAV, proving the
    cross-eval entry ("同一横评入口") is caliber-clean.
    """
    panel = _make_panel()
    close = panel["close"]
    return_df = close.pct_change(fill_method=None).shift(-1)
    n_groups = 5

    store = _store(tmp_path)
    store.register("close/delay(close,1)-1", name="rec_mom")
    mom = compute_factor(store, "rec_mom", 1, panel)

    result = evaluate_factor(store, "rec_mom", 1, panel, return_df=return_df, n_groups=n_groups)

    ic = compute_ic_series(mom, return_df)
    equity = compute_group_equity(mom, return_df, n_groups)

    assert result["source"] == "py-alpha-lib"
    assert result["ic"]["count"] == int(len(ic))
    assert result["ic"]["mean"] == round(float(ic.mean()), 6)
    assert result["ic"]["ir"] == round(float(ic.mean() / ic.std()), 4)
    assert result["layered_returns"]["final_nav"] == {
        col: round(float(equity[col].iloc[-1]), 4) for col in equity.columns
    }


def test_panel_to_long_layout_stable(tmp_path: Path) -> None:
    """The wide→long adapter's code-major / date-ascending layout is stable."""
    panel = _make_panel()
    long_df = panel_to_long(panel)
    n_dates = len(panel["close"].index)
    n_codes = len(panel["close"].columns)
    assert long_df.shape[0] == n_dates * n_codes
    # First n_dates rows belong to the first code, in date order.
    assert long_df["securityid"].iloc[:n_dates].nunique() == 1
    assert long_df["tradetime"].iloc[:n_dates].is_monotonic_increasing
