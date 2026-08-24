"""Versioned factor snapshot store + loader (DORA-124 §4.3 — F-02).

Storage layout (immutable, one version per directory)::

    <runtime_root>/factors/<factor_id>/<version>/snapshot.py

``<runtime_root>`` is ``~/.vibe-trading`` by default (override with the
``VIBE_TRADING_HOME`` env var — see ``src.config.paths.get_runtime_root``).

Security model — two gates run *before* any import (F-02 acceptance):

1. **Path gate.** ``factor_id`` and ``version`` are strictly validated and the
   resolved snapshot path is confirmed to stay under the factors root, so a
   caller cannot traverse outside it (``../``, absolute paths, etc.).
2. **Content whitelist gate.** The snapshot source is AST-parsed and every
   statement/expression is checked against a narrow allowlist: the only import
   is ``numpy``; the only function is ``compute(ctx)``; the returned expression
   may reference nothing but ``ctx`` / ``np`` and the alpha operator surface.
   This inherits the upstream factor "AST purity gate" idea
   (``src.factors.registry.load_alpha_meta_from_py``) and blocks code injection
   through a tampered snapshot.

The loader imports the snapshot by file path (``importlib``) and never calls the
online ``alpha.lang`` translator — offline reproducible.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from src.factor_runtime.availability import py_alpha_lib_version
from src.factor_runtime.translator import FUNCTION_NAME, translate_expression

logger = logging.getLogger(__name__)

#: Strict factor-id shape (mirrors ``src.factors.registry._ID_RE``) so the id
#: can never contain a path separator or ``..``.
_FACTOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Filename inside every version directory.
_SNAPSHOT_FILENAME = "snapshot.py"

#: Size cap for a snapshot source (mirrors ``registry._MAX_PY_BYTES``).
_MAX_SNAPSHOT_BYTES = 200_000

#: Synthetic module namespace for imported snapshots (never a real package).
_IMPORT_NAME_PREFIX = "vibe_trading.factors.snapshot"

#: Metadata literal carried by every snapshot (AST-extracted, never trusted
#: blindly — see :func:`validate_snapshot_source`).
_META_KEY = "__snapshot_meta__"

# ---------------------------------------------------------------------------
# Content whitelist constants
# ---------------------------------------------------------------------------

#: The only import the snapshot may make (and only as ``import numpy as np``).
_ALLOWED_IMPORTS = frozenset({"numpy"})

#: ``numpy`` helpers the ``alpha.lang`` translator can emit
#: (ternary -> ``where``, ``&&``/``||`` -> ``bitwise_*``, ``^`` -> ``power``).
_ALLOWED_NP_ATTRS = frozenset({"where", "bitwise_or", "bitwise_and", "power"})

#: Bare names legal inside the generated expression.
_ALLOWED_BARE_NAMES = frozenset({"ctx", "np", "True", "False", "None"})

#: Operator names on the context object (``ctx.MA(...)``) — UPPERCASE only,
#: which structurally excludes dunders (``__init__`` etc.).
_FUNC_ATTR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SnapshotValidationError(ValueError):
    """Raised when a snapshot fails the path or content whitelist gate."""


class SnapshotNotFoundError(FileNotFoundError):
    """Raised when a requested factor/version snapshot does not exist."""


class SnapshotConflictError(RuntimeError):
    """Raised when a version directory already holds a different snapshot."""


def factors_root() -> Path:
    """Return the factors root directory (``<runtime_root>/factors``)."""
    from src.config.paths import get_runtime_root  # lazy: keep module import light

    return get_runtime_root() / "factors"


def _validate_factor_id(factor_id: str) -> None:
    if not isinstance(factor_id, str) or not _FACTOR_ID_RE.fullmatch(factor_id):
        raise SnapshotValidationError(
            f"invalid factor_id {factor_id!r}; must match {_FACTOR_ID_RE.pattern}"
        )


def _validate_version(version: int) -> None:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SnapshotValidationError(f"invalid version {version!r}; must be a positive int")


def _derive_factor_id(expression: str) -> str:
    """Content-address a factor id from the expression (stable across restarts)."""
    digest = hashlib.sha256(expression.strip().encode("utf-8")).hexdigest()[:12]
    return f"f{digest}"


# ---------------------------------------------------------------------------
# Snapshot rendering + AST whitelist gate
# ---------------------------------------------------------------------------


def render_snapshot(
    factor_id: str,
    version: int,
    expression: str,
    translated_fn: str,
    py_alpha_lib: str | None,
    translated_at: str,
) -> str:
    """Render a self-contained, immutable snapshot module source."""
    meta: dict[str, Any] = {
        "factor_id": factor_id,
        "version": version,
        "expression": expression,
        "py_alpha_lib_version": py_alpha_lib,
        "translated_at": translated_at,
    }
    meta_block = ",\n".join(f"    {key!r}: {value!r}" for key, value in meta.items())
    return (
        f"# Generated immutable factor snapshot: {factor_id} v{version}\n"
        "# Do not edit — versioned snapshot of a translated alpha expression.\n"
        "# Re-translate and re-register to create a new version instead.\n"
        "import numpy as np\n"
        "\n"
        "\n"
        f"{translated_fn.rstrip()}\n"
        "\n"
        "\n"
        f"{_META_KEY} = {{\n"
        f"{meta_block},\n"
        "}\n"
    )


def validate_snapshot_source(
    source: str,
    *,
    factor_id: str | None = None,
    version: int | None = None,
) -> dict[str, Any]:
    """Validate snapshot source against the content whitelist (no import).

    Returns the ``__snapshot_meta__`` literal (parsed with ``ast.literal_eval``,
    so it cannot contain executable code). Raises :class:`SnapshotValidationError`
    on any disallowed construct.

    Args:
        source: Snapshot module source.
        factor_id: When given, the metadata must claim this factor id.
        version: When given, the metadata must claim this version.
    """
    if len(source.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise SnapshotValidationError("snapshot source exceeds size cap")
    try:
        tree = ast.parse(source, mode="exec", filename="<snapshot>")
    except SyntaxError as exc:
        raise SnapshotValidationError(f"snapshot is not valid Python: {exc}") from exc

    saw_compute = False
    meta: dict[str, Any] | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            _check_top_import(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            if stmt.name != FUNCTION_NAME:
                raise SnapshotValidationError(
                    f"only a single {FUNCTION_NAME}() definition is allowed"
                )
            _check_compute_function(stmt)
            saw_compute = True
        elif isinstance(stmt, ast.Assign):
            if meta is not None:
                raise SnapshotValidationError(
                    f"only one {_META_KEY} assignment is allowed"
                )
            meta = _check_meta_assignment(stmt)
        else:
            raise SnapshotValidationError(
                f"disallowed top-level statement: {type(stmt).__name__}"
            )

    if not saw_compute:
        raise SnapshotValidationError(f"snapshot must define {FUNCTION_NAME}(ctx)")

    if factor_id is not None and (meta is None or meta.get("factor_id") != factor_id):
        raise SnapshotValidationError("snapshot metadata does not match the requested factor_id")
    if version is not None and (meta is None or meta.get("version") != version):
        raise SnapshotValidationError("snapshot metadata does not match the requested version")
    return meta if meta is not None else {}


def _check_top_import(stmt: ast.Import) -> None:
    for alias in stmt.names:
        if alias.name not in _ALLOWED_IMPORTS:
            raise SnapshotValidationError(f"import {alias.name!r} is not allowed")
        if alias.asname not in (None, "np"):
            raise SnapshotValidationError(
                f"numpy must be imported unaliased or as 'np', got {alias.asname!r}"
            )


def _check_compute_function(fn: ast.FunctionDef) -> None:
    args = fn.args
    if fn.decorator_list:
        raise SnapshotValidationError("compute() must not be decorated")
    if args.vararg is not None or args.kwarg is not None:
        raise SnapshotValidationError("compute() must not use *args/**kwargs")
    if args.kwonlyargs or args.posonlyargs:
        raise SnapshotValidationError("compute() must take exactly one positional arg")
    positional = list(args.posonlyargs) + list(args.args)
    if len(positional) != 1 or positional[0].arg != "ctx":
        raise SnapshotValidationError("compute() must take exactly one arg named 'ctx'")
    if any(args.defaults) or args.kw_defaults:
        raise SnapshotValidationError("compute() must not declare default values")
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        raise SnapshotValidationError("compute() body must be a single return statement")
    _check_expression(fn.body[0].value)


def _check_meta_assignment(stmt: ast.Assign) -> dict[str, Any]:
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        raise SnapshotValidationError("snapshot module-level assignment must target one name")
    if stmt.targets[0].id != _META_KEY:
        raise SnapshotValidationError(
            f"only {_META_KEY} may be assigned at module level"
        )
    try:
        value = ast.literal_eval(stmt.value)
    except (ValueError, SyntaxError) as exc:
        raise SnapshotValidationError(f"{_META_KEY} must be a literal dict: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"{_META_KEY} must be a dict literal")
    return value


def _check_expression(node: ast.expr) -> None:
    """Recursively allow only the node surface the translator can emit."""
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_BARE_NAMES:
            raise SnapshotValidationError(f"disallowed name {node.id!r} in factor expression")
        return
    if isinstance(node, ast.BinOp):
        _check_expression(node.left)
        _check_expression(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise SnapshotValidationError(
                f"disallowed unary operator {type(node.op).__name__}"
            )
        _check_expression(node.operand)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check_expression(value)
        return
    if isinstance(node, ast.Compare):
        _check_expression(node.left)
        for comparator in node.comparators:
            _check_expression(comparator)
        return
    if isinstance(node, ast.Call):
        _check_call(node)
        return
    raise SnapshotValidationError(
        f"disallowed expression node {type(node).__name__} in factor expression"
    )


def _check_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name) and node.func.id == "ctx":
        # ``ctx('VARIABLE')`` — exactly one string-literal argument.
        if len(node.args) != 1 or node.keywords:
            raise SnapshotValidationError("ctx(...) variable access takes one string argument")
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            raise SnapshotValidationError("ctx(...) argument must be a string literal")
        return
    if isinstance(node.func, ast.Attribute):
        _check_attribute_call(node)
        return
    raise SnapshotValidationError(
        "only ctx(...) / ctx.OP(...) / np.OP(...) calls are allowed"
    )


def _check_attribute_call(node: ast.Call) -> None:
    func = node.func
    assert isinstance(func, ast.Attribute)  # guaranteed by caller
    if not isinstance(func.value, ast.Name):
        raise SnapshotValidationError("attribute access must be rooted at ctx or np")
    if func.value.id == "ctx":
        if not _FUNC_ATTR_RE.fullmatch(func.attr):
            raise SnapshotValidationError(f"disallowed context operator ctx.{func.attr}")
    elif func.value.id == "np":
        if func.attr not in _ALLOWED_NP_ATTRS:
            raise SnapshotValidationError(f"disallowed numpy function np.{func.attr}")
    else:
        raise SnapshotValidationError(f"disallowed attribute root {func.value.id!r}")
    if node.keywords:
        raise SnapshotValidationError("keyword arguments are not allowed in factor expressions")
    for arg in node.args:
        _check_expression(arg)


def read_snapshot_meta(path: Path) -> dict[str, Any] | None:
    """AST-extract ``__snapshot_meta__`` without importing (best-effort).

    Mirrors ``registry.load_alpha_meta_from_py``. Returns ``None`` for any
    unreadable/unparseable file so callers can treat it as "no metadata".
    """
    try:
        if path.stat().st_size > _MAX_SNAPSHOT_BYTES:
            return None
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == _META_KEY
        ):
            try:
                value = ast.literal_eval(stmt.value)
            except (ValueError, SyntaxError):
                return None
            return value if isinstance(value, dict) else None
    return None


# ---------------------------------------------------------------------------
# Snapshot handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """An imported, validated snapshot ready for compute (F-03 consumes this)."""

    factor_id: str
    version: int
    expression: str
    path: Path
    meta: dict[str, Any]
    _module: ModuleType = field(repr=False, compare=False)

    def compute(self, ctx: Any) -> Any:
        """Run the snapshot's ``compute(ctx)`` against a py-alpha-lib context."""
        return getattr(self._module, FUNCTION_NAME)(ctx)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SnapshotStore:
    """Write and load immutable, versioned factor snapshots."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or factors_root()).resolve()
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    # -- registration ---------------------------------------------------

    def register(self, expression: str, *, name: str | None = None) -> dict[str, Any]:
        """Translate ``expression`` and persist it as an immutable snapshot.

        Idempotent: re-registering an identical expression under the same factor
        id returns the existing version; a different expression allocates the
        next version. Returns ``{factor_id, version, expression, snapshot_path,
        py_alpha_lib_version}``.
        """
        if not expression or not expression.strip():
            raise ValueError("factor expression must not be empty")

        factor_id = name if name is not None else _derive_factor_id(expression)
        _validate_factor_id(factor_id)

        # Translation happens once, under the lock, so a healthy container never
        # races two translators for the same registration.
        with self._lock:
            translated = translate_expression(expression)
            version, existing = self._find_version(factor_id, expression)
            if existing is not None:
                return self._registration_result(
                    factor_id, version, expression, existing
                )
            path = self._write(factor_id, version, expression, translated)
        return self._registration_result(factor_id, version, expression, path)

    def _registration_result(
        self, factor_id: str, version: int, expression: str, path: Path
    ) -> dict[str, Any]:
        return {
            "factor_id": factor_id,
            "version": version,
            "expression": expression,
            "snapshot_path": str(path),
            "py_alpha_lib_version": py_alpha_lib_version(),
        }

    def _find_version(self, factor_id: str, expression: str) -> tuple[int, Path | None]:
        """Return ``(version, existing_path|None)`` for an idempotent register."""
        existing_versions = self.list_versions(factor_id)
        for version in existing_versions:
            path = self._snapshot_path(factor_id, version)
            meta = read_snapshot_meta(path)
            if meta is not None and meta.get("expression") == expression:
                return version, path
        next_version = (existing_versions[-1] + 1) if existing_versions else 1
        return next_version, None

    def _write(
        self, factor_id: str, version: int, expression: str, translated: str
    ) -> Path:
        version_dir = self._version_dir(factor_id, version)
        target = version_dir / _SNAPSHOT_FILENAME
        if target.exists():
            raise SnapshotConflictError(
                f"snapshot already exists at {target}; refusing to overwrite"
            )
        version_dir.mkdir(parents=True, exist_ok=True)
        content = render_snapshot(
            factor_id,
            version,
            expression,
            translated,
            py_alpha_lib_version(),
            _now_iso(),
        )
        tmp = version_dir / f".{_SNAPSHOT_FILENAME}.{os.getpid()}.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)  # atomic within the same directory
        try:  # best-effort immutability hint; not relied upon for security
            target.chmod(0o444)
        except OSError:
            pass
        return target

    # -- loading ---------------------------------------------------------

    def load(self, factor_id: str, version: int) -> Snapshot:
        """Validate (path + content) and import a snapshot, without re-translating."""
        _validate_factor_id(factor_id)
        _validate_version(version)
        path = self._snapshot_path(factor_id, version)
        if not self._is_within_root(path):
            raise SnapshotValidationError("snapshot path escapes the factors root")
        module = self._import_snapshot(factor_id, version, path)
        meta = getattr(module, _META_KEY, None)
        if not isinstance(meta, dict):
            raise SnapshotValidationError("snapshot is missing __snapshot_meta__")
        compute_fn = getattr(module, FUNCTION_NAME, None)
        if not callable(compute_fn):
            raise SnapshotValidationError(
                f"snapshot does not expose a callable {FUNCTION_NAME}()"
            )
        return Snapshot(
            factor_id=factor_id,
            version=version,
            expression=str(meta.get("expression", "")),
            path=path,
            meta=dict(meta),
            _module=module,
        )

    def _import_snapshot(self, factor_id: str, version: int, path: Path) -> ModuleType:
        if not path.is_file():
            raise SnapshotNotFoundError(f"snapshot not found: {path}")
        try:
            size = path.stat().st_size
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SnapshotNotFoundError(f"cannot read snapshot {path}: {exc}") from exc
        if size > _MAX_SNAPSHOT_BYTES:
            raise SnapshotValidationError("snapshot source exceeds size cap")

        # Content whitelist gate runs strictly before any import.
        validate_snapshot_source(source, factor_id=factor_id, version=version)

        module_name = f"{_IMPORT_NAME_PREFIX}.{factor_id}_v{version}"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise SnapshotValidationError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    # -- path helpers ----------------------------------------------------

    def _version_dir(self, factor_id: str, version: int) -> Path:
        _validate_factor_id(factor_id)
        _validate_version(version)
        return self._root / factor_id / str(version)

    def _snapshot_path(self, factor_id: str, version: int) -> Path:
        return self._version_dir(factor_id, version) / _SNAPSHOT_FILENAME

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._root)
        except ValueError:
            return False
        return True

    def list_versions(self, factor_id: str) -> list[int]:
        """Return the sorted, existing version numbers for ``factor_id``."""
        _validate_factor_id(factor_id)
        factor_dir = self._root / factor_id
        if not factor_dir.is_dir():
            return []
        versions: list[int] = []
        for entry in factor_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                version = int(entry.name)
            except ValueError:
                continue
            if version >= 1 and (entry / _SNAPSHOT_FILENAME).is_file():
                versions.append(version)
        return sorted(versions)

    def latest_version(self, factor_id: str) -> int | None:
        """Return the highest existing version, or ``None`` when unknown."""
        versions = self.list_versions(factor_id)
        return versions[-1] if versions else None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Process-wide store singleton (mirrors ``registry.get_default_registry``).
# ---------------------------------------------------------------------------

_store: SnapshotStore | None = None
_store_lock = threading.Lock()


def get_snapshot_store() -> SnapshotStore:
    """Return the process-wide :class:`SnapshotStore` for the runtime root."""
    global _store
    with _store_lock:
        if _store is None:
            _store = SnapshotStore()
        return _store


def reset_snapshot_store() -> None:
    """Drop the cached store (test hook; do not call in production)."""
    global _store
    with _store_lock:
        _store = None
