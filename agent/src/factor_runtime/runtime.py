"""New-factor entry points for the py-alpha-lib factor runtime.

F-01 delivered the runtime *shell* (availability guard + graceful degradation).
F-02 implemented ``register`` (alpha.lang translation → immutable, versioned
snapshot). F-03 implements ``compute`` / ``evaluate``: run a registered
snapshot over a wide OHLCV panel through py-alpha-lib's ``ExecContext`` and
score it with the same IC/IR + layered-return math the Alpha Zoo bench uses.

Entry points mirror DORA-124 §4.3 (``/alpha/custom``):
    register(expression) -> {factor_id, version}    (F-02)
    compute(factor_id, panel) -> pd.DataFrame       (F-03)
    evaluate(factor_id, panel, ...) -> dict         (F-03)

When py-alpha-lib is unavailable, every entry point raises
:class:`~src.factor_runtime.availability.FactorRuntimeUnavailableError`
(the "prompt Docker" degradation). ``compute``/``evaluate`` results carry
``{factor_id, version, py_alpha_lib, source}`` provenance.
"""

from __future__ import annotations

import threading
from typing import Any

from src.factor_runtime.availability import require_available, runtime_status


class FactorRuntime:
    """In-process facade over the py-alpha-lib runtime (DORA-124 module D).

    ``register`` is implemented by F-02; ``compute``/``evaluate`` land with
    F-03. Obtain the process-wide instance via :func:`get_runtime`.
    """

    def status(self) -> dict[str, Any]:
        """Return the availability snapshot (see ``availability.runtime_status``)."""
        return runtime_status()

    def register(self, expression: str, **kwargs: Any) -> dict[str, Any]:
        """Register a factor expression → translated, versioned snapshot (F-02).

        Returns ``{factor_id, version, ...}``. Degrades to
        :class:`FactorRuntimeUnavailableError` when py-alpha-lib is absent.
        """
        require_available()
        from src.factor_runtime.snapshot import get_snapshot_store

        name = kwargs.get("name") or kwargs.get("factor_id")
        return get_snapshot_store().register(expression, name=name)

    def compute(
        self,
        factor_id: str,
        panel: dict[str, Any],
        *,
        version: int | None = None,
    ) -> Any:
        """Compute a registered factor over a wide OHLCV panel (F-03).

        Loads the snapshot (offline, no re-translation) and runs it through
        py-alpha-lib's ``ExecContext``, returning a wide ``DataFrame`` aligned to
        ``panel["close"]`` — the same shape ``Registry.compute`` returns.

        Degrades to :class:`FactorRuntimeUnavailableError` when py-alpha-lib is
        absent; raises ``SnapshotNotFoundError`` for an unknown factor.
        """
        require_available()
        from src.factor_runtime.compute import compute_factor
        from src.factor_runtime.snapshot import get_snapshot_store

        store = get_snapshot_store()
        resolved = self._resolve_version(store, factor_id, version)
        return compute_factor(store, factor_id, resolved, panel)

    def evaluate(
        self,
        factor_id: str,
        panel: dict[str, Any],
        *,
        version: int | None = None,
        return_df: Any | None = None,
        n_groups: int = 5,
    ) -> dict[str, Any]:
        """Evaluate a factor's IC/IR + layered returns (F-03).

        Uses the shared zoo math (``factor_analysis_core``), so the numbers are
        directly comparable to preset factors. When ``return_df`` is omitted the
        canonical ``alpha_bench_tool._compute_forward_returns`` is used so both
        paths share one forward-returns definition.

        Returns ``{factor_id, version, source, py_alpha_lib, ic, layered_returns}``.
        """
        require_available()
        from src.factor_runtime.compute import evaluate_factor
        from src.factor_runtime.snapshot import get_snapshot_store

        store = get_snapshot_store()
        resolved = self._resolve_version(store, factor_id, version)
        if return_df is None:
            return_df = self._forward_returns(panel)
        return evaluate_factor(
            store, factor_id, resolved, panel, return_df=return_df, n_groups=n_groups
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _resolve_version(store: Any, factor_id: str, version: int | None) -> int:
        """Resolve an explicit version, or fall back to the latest snapshot."""
        if version is not None:
            return version
        latest = store.latest_version(factor_id)
        if latest is None:
            from src.factor_runtime.snapshot import SnapshotNotFoundError

            raise SnapshotNotFoundError(f"no snapshot registered for factor {factor_id!r}")
        return latest

    @staticmethod
    def _forward_returns(panel: dict[str, Any]) -> Any:
        """Derive forward returns with the canonical zoo-bench definition."""
        from src.tools.alpha_bench_tool import _compute_forward_returns  # lazy: heavy deps

        return _compute_forward_returns(panel)


# ---------------------------------------------------------------------------
# Process-wide singleton (mirrors ``src.factors.registry.get_default_registry``).
# The runtime shell holds no mutable state, so a singleton is safe; tests use
# ``reset_runtime`` or construct ``FactorRuntime()`` directly.
# ---------------------------------------------------------------------------

_runtime: FactorRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> FactorRuntime:
    """Return the process-wide :class:`FactorRuntime` instance."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = FactorRuntime()
        return _runtime


def reset_runtime() -> None:
    """Drop the cached runtime instance (test hook; do not call in production)."""
    global _runtime
    with _runtime_lock:
        _runtime = None
