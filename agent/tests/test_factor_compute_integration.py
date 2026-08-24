"""Integration test for F-03 against the REAL py-alpha-lib ``ExecContext``.

Skipped automatically where py-alpha-lib is not installed (the Windows-host
degradation path). Runs inside the Docker container, where ``import alpha``
succeeds (``cp311-abi3`` wheel), and proves the wide-panel adapter round-trips a
dense panel and that the IC/IR + layered-return math matches ``factor_analysis_core``.
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
from src.factors.factor_analysis_core import compute_group_equity, compute_ic_series  # noqa: E402


def _make_panel(n_dates: int = 12, n_codes: int = 8) -> dict[str, pd.DataFrame]:
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


def test_real_exec_context_roundtrip(tmp_path: Path) -> None:
    """Register via the real ``alpha.lang`` translator and round-trip a panel."""
    store = SnapshotStore(root=tmp_path / "factors")
    store.register("close", name="cl")
    store.register("close/delay(close,1)-1", name="mom")

    panel = _make_panel()

    close_frame = compute_factor(store, "cl", 1, panel)
    pd.testing.assert_frame_equal(close_frame, panel["close"])

    mom_frame = compute_factor(store, "mom", 1, panel)
    np.testing.assert_allclose(
        mom_frame.to_numpy(),
        panel["close"].pct_change(fill_method=None).to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )

    long_df = panel_to_long(panel)
    assert long_df.shape[0] == len(panel["close"].index) * len(panel["close"].columns)


def test_real_exec_context_evaluate_matches_zoo_math(tmp_path: Path) -> None:
    """``evaluate_factor`` must reproduce ``factor_analysis_core`` exactly."""
    store = SnapshotStore(root=tmp_path / "factors")
    store.register("close/delay(close,1)-1", name="mom")

    panel = _make_panel()
    return_df = panel["close"].pct_change(fill_method=None).shift(-1)
    mom_frame = compute_factor(store, "mom", 1, panel)

    result = evaluate_factor(store, "mom", 1, panel, return_df=return_df, n_groups=4)

    ic = compute_ic_series(mom_frame, return_df)
    equity = compute_group_equity(mom_frame, return_df, 4)
    assert result["source"] == "py-alpha-lib"
    assert result["ic"]["count"] == int(len(ic))
    assert result["ic"]["mean"] == round(float(ic.mean()), 6)
    assert result["layered_returns"]["final_nav"] == {
        col: round(float(equity[col].iloc[-1]), 4) for col in equity.columns
    }
