"""Availability probe and degradation guard for the py-alpha-lib factor runtime.

The runtime is only fully functional where ``py-alpha-lib==0.3.0`` is
importable — inside the Docker container (``cp311-abi3-manylinux`` wheel,
Python ≥ 3.12). On a bare Windows host the package ships only a
``cp314-abi3`` wheel (Python ≥ 3.14), so a stock 3.11/3.12 install has no
matching wheel and ``import alpha`` fails.

Degradation contract (DORA-124 F-01):
    * :func:`is_available` returns ``False`` when the ``alpha`` extension
      cannot be imported (missing wheel) or lacks its callable primitives.
    * Alpha Zoo presets are unaffected — they run through the pandas path in
      ``src.factors.registry.Registry.compute`` and never touch this module.
    * New-factor entry points call :func:`require_available`, which raises
      :class:`FactorRuntimeUnavailableError` carrying an actionable Docker hint.

The probe result is cached per-process (mirroring ``src.factors._backend``);
call :func:`reset_probe` in tests or after mutating ``sys.modules``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import platform
import threading
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

#: Import name of the py-alpha-lib Python package (the Rust/PyO3 binding).
IMPORT_NAME = "alpha"

#: Pinned distribution name + version (DORA-124 F-01: ``py-alpha-lib==0.3.0``).
DIST_NAME = "py-alpha-lib"
EXPECTED_VERSION = "0.3.0"

#: Callable primitive used as a sanity marker that the native extension loaded
#: (a broken/partial install may import a stub without the Rust operators).
_SANITY_ATTR = "MA"

#: Actionable hint surfaced on every degraded new-factor entry point.
DOCKER_HINT = (
    "py-alpha-lib factor runtime is unavailable in this environment. "
    "Run the Docker factor service (docker-compose builds the vibe-trading "
    "agent image with py-alpha-lib==0.3.0 baked in) and retry; the bundled "
    "Alpha Zoo factors remain available without Docker."
)


class FactorRuntimeUnavailableError(RuntimeError):
    """Raised when a new-factor entry point is used without the runtime lib."""

    def __init__(self, detail: str | None = None) -> None:
        message = DOCKER_HINT if detail is None else f"{DOCKER_HINT} ({detail})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Probe cache. A process never gains or loses the native extension at runtime,
# so the import is probed once and cached; ``reset_probe`` is the test hook.
# ---------------------------------------------------------------------------

_cached_available: bool | None = None
_cached_module: ModuleType | None = None
_probe_lock = threading.Lock()


def _probe() -> tuple[bool, ModuleType | None]:
    try:
        module = importlib.import_module(IMPORT_NAME)
    except (ImportError, OSError) as exc:
        logger.debug("py-alpha-lib import unavailable: %s", exc)
        return False, None
    if not callable(getattr(module, _SANITY_ATTR, None)):
        logger.debug("py-alpha-lib imported but %r is not callable", _SANITY_ATTR)
        return False, None
    return True, module


def is_available() -> bool:
    """Return ``True`` when py-alpha-lib's ``alpha`` extension is importable."""
    global _cached_available, _cached_module
    with _probe_lock:
        if _cached_available is None:
            _cached_available, _cached_module = _probe()
        return _cached_available


def require_available() -> ModuleType:
    """Return the ``alpha`` module, else raise :class:`FactorRuntimeUnavailableError`.

    This is the single gate every new-factor entry point calls before doing any
    work, so the "prompt Docker" degradation is uniform across register /
    compute / evaluate.
    """
    if not is_available():
        raise FactorRuntimeUnavailableError()
    assert _cached_module is not None  # noqa: S101 — guarded by is_available()
    return _cached_module


def py_alpha_lib_version() -> str | None:
    """Return the installed py-alpha-lib version, or ``None`` when not installed.

    Reads distribution metadata (``importlib.metadata``) rather than the
    package, so the version is observable even when the import probe fails for
    an ABI/load reason.
    """
    try:
        return importlib.metadata.version(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_status() -> dict[str, Any]:
    """Machine-readable availability snapshot (CLI/REST/health surfaces)."""
    available = is_available()
    return {
        "available": available,
        "degraded": not available,
        "import_name": IMPORT_NAME,
        "dist_name": DIST_NAME,
        "expected_version": EXPECTED_VERSION,
        "version": py_alpha_lib_version(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "hint": None if available else DOCKER_HINT,
    }


def reset_probe() -> None:
    """Drop the cached probe result (test hook; do not call in production)."""
    global _cached_available, _cached_module
    with _probe_lock:
        _cached_available = None
        _cached_module = None
