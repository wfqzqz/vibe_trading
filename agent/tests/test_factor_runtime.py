"""Tests for the py-alpha-lib factor runtime shell (DORA-144 / F-01).

Covers both sides of the degradation contract without needing py-alpha-lib
installed:

* **unavailable** (Windows host, no wheel): the probe returns False, ``alpha``
  zoo presets stay untouched, and the new-factor entry points raise
  ``FactorRuntimeUnavailableError`` carrying the Docker hint;
* **available** (container): a fake ``alpha`` module in ``sys.modules`` makes
  the probe pass, so the entry points get past the guard (``register`` is
  implemented by F-02, ``compute``/``evaluate`` by F-03 — see
  ``test_factor_snapshot.py`` / ``test_factor_compute.py``).

No network; deterministic. The probe cache is reset per test.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

import src.factor_runtime.availability as availability
from src.factor_runtime import (
    DOCKER_HINT,
    FactorRuntimeUnavailableError,
    FactorRuntime,
    SnapshotNotFoundError,
    get_runtime,
    is_available,
    py_alpha_lib_version,
    require_available,
    reset_probe,
    runtime_status,
)


@pytest.fixture(autouse=True)
def _fresh_probe():
    reset_probe()
    yield
    reset_probe()


def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import alpha`` fail, then drop the cached probe."""
    def _raise(name: str, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"No module named {name!r} (test)")

    monkeypatch.setattr(importlib, "import_module", _raise)
    reset_probe()


def _fake_alpha() -> types.ModuleType:
    fake = types.ModuleType("alpha")
    fake.MA = lambda data, periods: data  # callable marker only
    return fake


# --------------------------------------------------------------------------- #
# Unavailable (degradation) path                                              #
# --------------------------------------------------------------------------- #


def test_is_available_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_unavailable(monkeypatch)
    assert is_available() is False


def test_require_available_raises_with_docker_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_unavailable(monkeypatch)
    with pytest.raises(FactorRuntimeUnavailableError) as exc_info:
        require_available()
    assert "Docker" in str(exc_info.value)
    assert "Alpha Zoo" in str(exc_info.value)


def test_status_reports_degraded_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_unavailable(monkeypatch)
    status = runtime_status()
    assert status["available"] is False
    assert status["degraded"] is True
    assert status["expected_version"] == "0.3.0"
    assert status["version"] is None
    assert status["hint"] == DOCKER_HINT


def test_new_factor_entry_points_degrade_to_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-01 acceptance: new-factor entry prompts Docker instead of crashing."""
    _force_unavailable(monkeypatch)
    runtime = get_runtime()
    for call in (
        lambda: runtime.register("close/ref(close,1)-1"),
        lambda: runtime.compute("factor_x", {}),
        lambda: runtime.evaluate("factor_x", {}),
    ):
        with pytest.raises(FactorRuntimeUnavailableError) as exc_info:
            call()
        assert "Docker" in str(exc_info.value)


def test_py_alpha_lib_version_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(availability.importlib.metadata, "version", _missing)
    assert py_alpha_lib_version() is None


# --------------------------------------------------------------------------- #
# Available (container) path                                                  #
# --------------------------------------------------------------------------- #


def test_is_available_true_with_fake_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "alpha", _fake_alpha())
    assert is_available() is True


def test_require_available_returns_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_alpha()
    monkeypatch.setitem(sys.modules, "alpha", fake)
    assert require_available() is fake


def test_status_reports_available_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "alpha", _fake_alpha())
    monkeypatch.setattr(
        availability.importlib.metadata, "version", lambda _name: "0.3.0"
    )
    status = runtime_status()
    assert status["available"] is True
    assert status["degraded"] is False
    assert status["version"] == "0.3.0"
    assert status["hint"] is None


def test_compute_evaluate_pass_guard_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the guard passes in a healthy container (compute/evaluate now run).

    With a fake ``alpha`` (MA marker only) and no snapshot registered, the entry
    points should proceed past ``require_available()`` and fail on the missing
    snapshot — NOT degrade to the Docker hint.
    """
    monkeypatch.setitem(sys.modules, "alpha", _fake_alpha())
    runtime = get_runtime()
    with pytest.raises(SnapshotNotFoundError):
        runtime.compute("factor_x", {})
    with pytest.raises(SnapshotNotFoundError):
        runtime.evaluate("factor_x", {})


def test_sanity_marker_missing_means_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub ``alpha`` package without the Rust operators must read as unavailable."""
    fake = types.ModuleType("alpha")
    monkeypatch.setitem(sys.modules, "alpha", fake)
    assert is_available() is False


# --------------------------------------------------------------------------- #
# Singleton                                                                   #
# --------------------------------------------------------------------------- #


def test_get_runtime_is_singleton() -> None:
    assert get_runtime() is get_runtime()


def test_runtime_status_echoes_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_unavailable(monkeypatch)
    assert FactorRuntime().status()["available"] is False
