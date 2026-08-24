"""New-factor entry points for the py-alpha-lib factor runtime.

F-01 delivered the runtime *shell* (availability guard + graceful degradation).
F-02 implements the ``register`` body (alpha.lang translation → immutable,
versioned snapshot). The compute/evaluate bodies (F-03) are still outstanding:

* when py-alpha-lib is unavailable, every entry point raises
  :class:`~src.factor_runtime.availability.FactorRuntimeUnavailableError`
  (the "prompt Docker" degradation);
* when it is available, ``compute`` / ``evaluate`` raise
  :class:`FactorRuntimeNotImplementedError` naming the owning task, so a healthy
  container fails loudly rather than silently returning placeholder data.

Entry points mirror DORA-124 §4.3 (``/alpha/custom``):
    register(expression) -> {factor_id, version}    (F-02)
    compute(factor_id, panel) -> pd.DataFrame       (F-03)
    evaluate(factor_id, panel) -> dict              (F-03)
"""

from __future__ import annotations

import threading
from typing import Any

from src.factor_runtime.availability import require_available, runtime_status

# Tracking references for the parts deliberately left out of this task's scope
# (Agent Identity: NotImplemented must carry a tracking-issue reference).
_COMPUTE_OWNER = "F-03 (factor compute / evaluate entry points)"


class FactorRuntimeNotImplementedError(NotImplementedError):
    """Raised when an available runtime is asked for a not-yet-built stage."""


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

    def compute(self, factor_id: str, panel: Any, **kwargs: Any) -> Any:
        """Compute a registered factor over a panel (F-03).

        Degrades to :class:`FactorRuntimeUnavailableError` when py-alpha-lib is
        absent; raises :class:`FactorRuntimeNotImplementedError` otherwise.
        """
        require_available()
        raise FactorRuntimeNotImplementedError(
            f"factor compute is implemented by {_COMPUTE_OWNER} "
            "(DORA-124 §五 Stage 4)"
        )

    def evaluate(self, factor_id: str, panel: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate a factor's IC/IR + layered returns (F-03).

        Degrades to :class:`FactorRuntimeUnavailableError` when py-alpha-lib is
        absent; raises :class:`FactorRuntimeNotImplementedError` otherwise.
        """
        require_available()
        raise FactorRuntimeNotImplementedError(
            f"factor evaluate is implemented by {_COMPUTE_OWNER} "
            "(DORA-124 §五 Stage 4)"
        )


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
