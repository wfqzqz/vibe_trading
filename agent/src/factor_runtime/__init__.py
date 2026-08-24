"""py-alpha-lib factor runtime (DORA-124 §3.4 module D — F-01).

An **in-process** module of the agent — NOT a standalone microservice
(DORA-124 revision v1.1, 附带澄清 1: "factor-runtime 为 agent 进程内模块").
It is only fully functional inside the Docker container, where
``py-alpha-lib==0.3.0`` is importable (``cp311-abi3-manylinux`` wheel ⊇
Python 3.12). On a bare Windows host the package ships only a ``cp314-abi3``
wheel, so a stock 3.11/3.12 install has no wheel and the import fails.

Degradation contract (F-01):
    * the Alpha Zoo's ~460 preset factors keep working unchanged through the
      pandas path (``src.factors.registry.Registry.compute``) — they never
      import this module;
    * only the *new* factor registration/compute entry points degrade: they
      raise :class:`FactorRuntimeUnavailableError` with an actionable hint to
      run the Docker factor service.

Submodules:
    ``availability`` — import probe, version read, degradation guard + hint.
    ``runtime`` — ``FactorRuntime`` facade with the new-factor entry points
    (``register`` / ``compute`` / ``evaluate``); F-02/F-03 fill in the
    translation/snapshot/compute bodies.
"""

from src.factor_runtime.availability import (
    DOCKER_HINT,
    EXPECTED_VERSION,
    FactorRuntimeUnavailableError,
    is_available,
    py_alpha_lib_version,
    require_available,
    reset_probe,
    runtime_status,
)
from src.factor_runtime.runtime import (
    FactorRuntime,
    FactorRuntimeNotImplementedError,
    get_runtime,
    reset_runtime,
)

__all__ = [
    "DOCKER_HINT",
    "EXPECTED_VERSION",
    "FactorRuntime",
    "FactorRuntimeNotImplementedError",
    "FactorRuntimeUnavailableError",
    "get_runtime",
    "is_available",
    "py_alpha_lib_version",
    "require_available",
    "reset_probe",
    "reset_runtime",
    "runtime_status",
]
