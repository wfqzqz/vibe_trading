"""Tests for factor compute / evaluate (DORA-146 / F-03).

No py-alpha-lib wheel is required: a fake ``alpha`` package provides a minimal
``ExecContext`` that mirrors the real one's flat *code-major / date-ascending*
layout, so the wide-panel → ExecContext → wide-panel round-trip and the shared
IC/IR + layered-return math are exercised end-to-end without the Rust extension.
The real py-alpha-lib round-trip is validated separately (see
``factor_runtime/COMPATIBILITY.md`` and the F-03 delivery note).

Covers:
    * ``panel_to_long`` — volume→vol mapping, vwap synthesis, row ordering;
    * ``reshape_factor_result`` — scalar/mis-shape rejection;
    * ``compute_factor`` — snapshot round-trip equals ``panel["close"]``;
    * ``evaluate_factor`` — IC/IR + layered returns match ``factor_analysis_core``;
    * degradation — compute/evaluate raise the Docker hint without ``alpha``;
    * API — ``/alpha/custom/{id}/compute`` and ``/evaluate`` provenance + errors.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.factor_runtime.availability as availability
import src.factor_runtime.snapshot as snapshot
from src.factor_runtime import (
    FactorComputeError,
    FactorRuntimeUnavailableError,
    SnapshotNotFoundError,
    SnapshotStore,
    compute_factor,
    evaluate_factor,
    panel_to_long,
    reshape_factor_result,
)
from src.factors.factor_analysis_core import compute_group_equity, compute_ic_series

#: Translated body the whitelist accepts: factor value == ctx('CLOSE').
_BODY = "def compute(ctx):\n    return ctx('CLOSE')\n"


class _FakeExecContext:
    """Minimal ExecContext: reads the long frame's OHLCV in flat row order."""

    def __init__(self, data: Any, securities: int = 0, trades: int = 0, fill: bool = True) -> None:
        self.OPEN = data["open"].to_numpy(dtype=np.float64)
        self.HIGH = data["high"].to_numpy(dtype=np.float64)
        self.LOW = data["low"].to_numpy(dtype=np.float64)
        self.CLOSE = data["close"].to_numpy(dtype=np.float64)
        self.VOLUME = data["vol"].to_numpy(dtype=np.float64)
        self.VWAP = data["vwap"].to_numpy(dtype=np.float64)

    def __call__(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _install_fake_alpha(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``alpha`` package exposing the sanity marker + ExecContext."""
    fake = types.ModuleType("alpha")
    fake.MA = lambda data, periods: data  # sanity marker for the availability probe

    ctx_mod = types.ModuleType("alpha.context")
    ctx_mod.ExecContext = _FakeExecContext
    fake.context = ctx_mod
    fake.ExecContext = _FakeExecContext

    monkeypatch.setitem(sys.modules, "alpha", fake)
    monkeypatch.setitem(sys.modules, "alpha.context", ctx_mod)
    availability.reset_probe()
    return fake


def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    def _raise(name: str, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"No module named {name!r} (test)")

    monkeypatch.setattr(importlib, "import_module", _raise)
    availability.reset_probe()


@pytest.fixture(autouse=True)
def _fresh_probe_and_store(monkeypatch: pytest.MonkeyPatch) -> None:
    availability.reset_probe()
    snapshot.reset_snapshot_store()
    yield
    availability.reset_probe()
    snapshot.reset_snapshot_store()


def _store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(root=tmp_path / "factors")


def _register_close_factor(store: SnapshotStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store.register("close", name="m")


def _make_panel(n_dates: int = 10, n_codes: int = 6) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, list[str]]:
    """Deterministic wide OHLCV panel (no vwap → synthesized typical price)."""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D")
    codes = [f"c{i}" for i in range(n_codes)]
    rng = np.random.default_rng(42)
    close = rng.uniform(8, 60, size=(n_dates, n_codes)).cumsum(axis=0)

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=codes)

    panel = {
        "open": frame(close * 0.99),
        "high": frame(close * 1.02),
        "low": frame(close * 0.98),
        "close": frame(close),
        "volume": frame(rng.uniform(1000, 5000, size=(n_dates, n_codes))),
    }
    return panel, dates, codes


def _forward_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Canonical zoo-bench forward returns (alpha_bench_tool._compute_forward_returns).
    return panel["close"].pct_change(fill_method=None).shift(-1)


# --------------------------------------------------------------------------- #
# panel_to_long / reshape                                                     #
# --------------------------------------------------------------------------- #


def test_panel_to_long_maps_columns_and_orders_rows() -> None:
    panel, dates, codes = _make_panel()
    long_df = panel_to_long(panel)

    n_dates, n_codes = len(dates), len(codes)
    assert long_df.shape[0] == n_dates * n_codes
    assert list(long_df.columns) == [
        "securityid", "tradetime", "open", "high", "low", "close", "vol", "vwap",
    ]
    # Row order: code-major, date-ascending.
    assert long_df["securityid"].tolist() == [c for c in codes for _ in range(n_dates)]
    assert long_df["tradetime"].tolist() == [d for _ in range(n_codes) for d in dates]
    # volume -> vol
    assert np.allclose(long_df["vol"].to_numpy(), panel["volume"].to_numpy().T.ravel())
    # vwap synthesized as (O+H+L+C)/4
    expected_vwap = (panel["open"] + panel["high"] + panel["low"] + panel["close"]) / 4.0
    assert np.allclose(long_df["vwap"].to_numpy(), expected_vwap.to_numpy().T.ravel())


def test_panel_to_long_uses_provided_vwap() -> None:
    panel, dates, codes = _make_panel()
    panel["vwap"] = panel["close"] * 1.01
    long_df = panel_to_long(panel)
    assert np.allclose(long_df["vwap"].to_numpy(), (panel["close"] * 1.01).to_numpy().T.ravel())


def test_reshape_factor_result_roundtrip() -> None:
    panel, dates, codes = _make_panel()
    arr = panel["close"].to_numpy(dtype=np.float64).T.ravel()  # code-major
    wide = reshape_factor_result(arr, panel["close"].index, panel["close"].columns)
    pd.testing.assert_frame_equal(wide, panel["close"])


def test_reshape_factor_result_rejects_scalar_and_misshape() -> None:
    panel, dates, codes = _make_panel()
    idx, cols = panel["close"].index, panel["close"].columns
    with pytest.raises(FactorComputeError):
        reshape_factor_result(np.array(1.0), idx, cols)
    with pytest.raises(FactorComputeError):
        reshape_factor_result(np.array([1.0, 2.0]), idx, cols)


def test_reshape_factor_result_rejects_inf() -> None:
    panel, dates, codes = _make_panel()
    idx, cols = panel["close"].index, panel["close"].columns
    arr = np.full(len(dates) * len(codes), np.nan)
    arr[0] = np.inf
    with pytest.raises(FactorComputeError):
        reshape_factor_result(arr, idx, cols)


# --------------------------------------------------------------------------- #
# compute_factor                                                               #
# --------------------------------------------------------------------------- #


def test_compute_factor_roundtrip_equals_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    store = _store(tmp_path)
    _register_close_factor(store, monkeypatch)

    panel, _, _ = _make_panel()
    frame = compute_factor(store, "m", 1, panel)

    pd.testing.assert_frame_equal(frame, panel["close"])


def test_compute_factor_unknown_snapshot_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    panel, _, _ = _make_panel()
    with pytest.raises(SnapshotNotFoundError):
        compute_factor(_store(tmp_path), "m", 1, panel)


def test_compute_factor_rejects_bad_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    store = _store(tmp_path)
    _register_close_factor(store, monkeypatch)
    with pytest.raises(FactorComputeError):
        compute_factor(store, "m", 1, {"open": _make_panel()[0]["open"]})


# --------------------------------------------------------------------------- #
# evaluate_factor                                                              #
# --------------------------------------------------------------------------- #


def test_evaluate_factor_matches_zoo_math(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    store = _store(tmp_path)
    _register_close_factor(store, monkeypatch)

    panel, _, _ = _make_panel()
    return_df = _forward_returns(panel)
    result = evaluate_factor(store, "m", 1, panel, return_df=return_df, n_groups=3)

    ic = compute_ic_series(panel["close"], return_df)
    equity = compute_group_equity(panel["close"], return_df, 3)

    assert result["factor_id"] == "m"
    assert result["version"] == 1
    assert result["source"] == "py-alpha-lib"
    assert result["shape"] == [panel["close"].shape[0], panel["close"].shape[1]]
    assert result["ic"]["count"] == int(len(ic))
    assert result["ic"]["mean"] == round(float(ic.mean()), 6)
    assert result["ic"]["ir"] == round(float(ic.mean() / ic.std()), 4)
    assert result["layered_returns"]["final_nav"] == {
        col: round(float(equity[col].iloc[-1]), 4) for col in equity.columns
    }


def test_evaluate_factor_rejects_empty_ic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-code panel yields < 5 instruments/bar → empty IC → compute error."""
    _install_fake_alpha(monkeypatch)
    store = _store(tmp_path)
    _register_close_factor(store, monkeypatch)

    panel, _, _ = _make_panel(n_dates=6, n_codes=1)
    return_df = _forward_returns(panel)
    with pytest.raises(FactorComputeError):
        evaluate_factor(store, "m", 1, panel, return_df=return_df)


# --------------------------------------------------------------------------- #
# Degradation                                                                  #
# --------------------------------------------------------------------------- #


def test_compute_evaluate_degrade_without_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_unavailable(monkeypatch)
    from src.factor_runtime import get_runtime

    runtime = get_runtime()
    panel, _, _ = _make_panel()
    for call in (lambda: runtime.compute("m", panel), lambda: runtime.evaluate("m", panel)):
        with pytest.raises(FactorRuntimeUnavailableError) as exc_info:
            call()
        assert "Docker" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# API routes                                                                   #
# --------------------------------------------------------------------------- #


def _allow() -> None:
    """Stub auth dependency — always allows."""


@pytest.fixture
def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path))
    snapshot.reset_snapshot_store()

    # Stub the network panel loader; the worker imports it lazily.
    import src.tools.alpha_bench_tool as abt

    panel, _, _ = _make_panel()
    monkeypatch.setattr(abt, "_load_universe_panel", lambda universe, period: panel)

    from fastapi import FastAPI

    from src.api.alpha_routes import register_alpha_routes

    app = FastAPI()
    register_alpha_routes(app, require_auth=_allow, require_event_stream_auth=_allow)
    return TestClient(app)


def test_compute_endpoint_returns_provenance(
    _client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_alpha(monkeypatch)
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    _store(tmp_path).register("close", name="m")

    response = _client.post(
        "/alpha/custom/m/compute", json={"universe": "csi300", "period": "2026-2026"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["factor_id"] == "m"
    assert payload["version"] == 1
    assert payload["source"] == "py-alpha-lib"
    assert payload["shape"][1] == 6
    assert "frame" in payload


def test_evaluate_endpoint_returns_ic_and_layered(
    _client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_alpha(monkeypatch)
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    _store(tmp_path).register("close", name="m")

    response = _client.post(
        "/alpha/custom/m/evaluate",
        json={"universe": "csi300", "period": "2026-2026", "n_groups": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["source"] == "py-alpha-lib"
    assert "ic" in payload and "layered_returns" in payload
    assert payload["ic"]["count"] > 0


def test_compute_endpoint_degrades_without_alpha(_client: TestClient) -> None:
    response = _client.post(
        "/alpha/custom/m/compute", json={"universe": "csi300", "period": "2026-2026"}
    )
    assert response.status_code == 503
    assert "Docker" in response.json()["detail"]


def test_compute_endpoint_unknown_factor_404(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_alpha(monkeypatch)
    response = _client.post(
        "/alpha/custom/zzz/compute", json={"universe": "csi300", "period": "2026-2026"}
    )
    assert response.status_code == 404


def test_compute_endpoint_bad_factor_id_shape(_client: TestClient) -> None:
    # "UPPER" fails the factor-id regex (lowercase only) without URL-normalization.
    response = _client.post(
        "/alpha/custom/UPPER/compute", json={"universe": "csi300", "period": "2026-2026"}
    )
    assert response.status_code == 400


def test_evaluate_endpoint_benchmarks_zoo(
    _client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_alpha(monkeypatch)
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    _store(tmp_path).register("close", name="m")

    # Stub the heavy zoo bits (registry scan + real factor compute) so the test
    # proves the ranking wiring, not the zoo math (covered elsewhere).
    import src.factors.bench_runner as bench_runner
    from src.factors import compare_runner

    class _FakeAlpha:
        zoo = "gtja191"

    class _FakeRegistry:
        def get(self, aid: str) -> _FakeAlpha:
            return _FakeAlpha()

    monkeypatch.setattr(compare_runner, "get_default_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        bench_runner,
        "_compute_single_alpha",
        lambda args: {
            "row": {
                "id": args[0],
                "ic_mean": 0.01,
                "ic_std": 0.05,
                "ir": 0.2,
                "ic_positive_ratio": 0.6,
                "ic_count": 100,
                "theme": [],
                "formula_latex": "",
            }
        },
    )

    response = _client.post(
        "/alpha/custom/m/evaluate",
        json={
            "universe": "csi300",
            "period": "2026-2026",
            "benchmark_alpha_ids": ["gtja191_alpha_001", "alpha101_001"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "benchmark" in payload
    ids = [r["id"] for r in payload["benchmark"]["ranking"]]
    assert "m" in ids
    custom = next(r for r in payload["benchmark"]["ranking"] if r["id"] == "m")
    assert custom["source"] == "py-alpha-lib"
    zoo = next(r for r in payload["benchmark"]["ranking"] if r["id"] != "m")
    assert zoo["source"] == "zoo"


def test_compare_custom_with_zoo_ranks_custom_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The custom factor (higher IR) must rank above the stub zoo factors."""
    import src.factors.bench_runner as bench_runner
    from src.factors import compare_runner

    class _FakeAlpha:
        zoo = "gtja191"

    class _FakeRegistry:
        def get(self, aid: str) -> _FakeAlpha:
            return _FakeAlpha()

    monkeypatch.setattr(compare_runner, "get_default_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        bench_runner,
        "_compute_single_alpha",
        lambda args: {
            "row": {
                "id": args[0],
                "ic_mean": 0.01,
                "ic_std": 0.05,
                "ir": 0.2,
                "ic_positive_ratio": 0.6,
                "ic_count": 100,
                "theme": [],
                "formula_latex": "",
            }
        },
    )

    panel, _, _ = _make_panel()
    custom_row = {
        "id": "m",
        "ic_mean": 0.05,
        "ic_std": 0.05,
        "ir": 1.0,
        "ic_positive_ratio": 0.9,
        "ic_count": 100,
    }
    result = compare_runner.compare_custom_with_zoo(
        custom_row, ["gtja191_alpha_001", "alpha101_001"], panel, _forward_returns(panel)
    )

    assert [r["id"] for r in result["ranking"]] == ["m", "gtja191_alpha_001", "alpha101_001"]
    assert result["ranking"][0]["source"] == "py-alpha-lib"
    assert all(r["source"] == "zoo" for r in result["ranking"][1:])
    assert result["ranking"][0]["ir"] == 1.0
