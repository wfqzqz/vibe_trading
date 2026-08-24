"""ExecContext operator-surface validation (DORA-188 — F-03 review P1 closure).

py-alpha-lib's documentation advertises ``REF`` / ``HHV`` / ``LLV`` /
``HHVBARS`` / ``LLVBARS`` aliases that the 0.3.0 ``alpha.context.ExecContext``
does **not** actually implement, so a translated ``ctx.REF(...)`` call only
fails at compute time with a raw ``AttributeError``. This module closes the
F-02 review condition ("按 ExecContext 实际算子面收口或上报上游") by checking
every ``ctx.OP(...)`` call in a *translated* snapshot body against the real
operator surface of the installed ``ExecContext``:

* registration (``SnapshotStore.register`` → ``translate_expression``) rejects
  unsupported operators with :class:`FactorOperatorError` — the REST API maps
  it to **400** with an explicit "operator not supported" message and the CLI
  prints the same readable one-liner;
* compute/evaluate (``compute_factor``) map the residual ``AttributeError`` (a
  snapshot that predates the check) to :class:`FactorComputeError` — the REST
  API maps it to **422** instead of a generic 500.

When the ``alpha`` package exposes no ``ExecContext`` (e.g. a minimal fake in
tests, or a future layout change) validation is skipped: the compute-time
``AttributeError`` mapping remains as the safety net, so the alias gap can
never degrade into a 500 generic error.
"""

from __future__ import annotations

import ast
import re
from typing import Any

#: Operator-name shape on the context object (mirrors ``snapshot._FUNC_ATTR_RE``):
#: UPPERCASE, which structurally excludes dunders (``__init__`` etc.).
_OPERATOR_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class FactorOperatorError(ValueError):
    """Raised when a translated expression calls an operator the installed
    ``ExecContext`` does not actually implement (doc-claimed alias gap)."""


def resolve_exec_context(alpha_module: Any) -> type | None:
    """Resolve ``ExecContext`` from the ``alpha`` package (context then top-level).

    Returns ``None`` when the package exposes none (validation is skipped then;
    the compute-time ``AttributeError`` mapping stays as the safety net).
    """
    cls = getattr(getattr(alpha_module, "context", None), "ExecContext", None)
    if cls is None:
        cls = getattr(alpha_module, "ExecContext", None)
    if cls is None or not callable(cls):
        return None
    return cls


def exec_context_operator_names(exec_context_cls: type) -> frozenset[str]:
    """Return the operator-looking attribute names on ``ExecContext``.

    Only names matching the UPPERCASE operator convention are considered. Data
    columns (``CLOSE`` etc.) are deliberately not treated as operators: in the
    generated snapshot bodies variable access goes through ``ctx('NAME')``,
    never ``ctx.NAME(...)``.
    """
    names: set[str] = set()
    for name in dir(exec_context_cls):
        if _OPERATOR_NAME_RE.fullmatch(name):
            names.add(name)
    return frozenset(names)


def collect_ctx_operator_calls(translated_body: str) -> list[str]:
    """Return the ``ctx.OP`` operator names called in a translated body.

    Only attribute calls rooted at ``ctx`` count (``ctx('NAME')`` variable
    access and ``np.*`` helper calls are not operators). First-seen order,
    deduplicated.
    """
    try:
        tree = ast.parse(translated_body, mode="exec", filename="<translated>")
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "ctx":
            continue
        if _OPERATOR_NAME_RE.fullmatch(func.attr):
            seen.append(func.attr)
    return list(dict.fromkeys(seen))


def validate_operator_surface(translated_body: str, alpha_module: Any) -> None:
    """Raise :class:`FactorOperatorError` for any unsupported ``ctx.OP`` call.

    Skips validation when ``alpha_module`` exposes no ``ExecContext``; the
    compute-time ``AttributeError`` → ``FactorComputeError`` mapping in
    ``compute.py`` remains as the safety net in that case.
    """
    exec_context_cls = resolve_exec_context(alpha_module)
    if exec_context_cls is None:
        return
    surface = exec_context_operator_names(exec_context_cls)
    unsupported = [
        op for op in collect_ctx_operator_calls(translated_body) if op not in surface
    ]
    if not unsupported:
        return
    joined = ", ".join(unsupported)
    supported = ", ".join(sorted(surface)) or "none"
    raise FactorOperatorError(
        f"operator(s) {joined} not supported by the py-alpha-lib ExecContext "
        f"(documented alias not implemented in 0.3.0); supported operator "
        f"surface: {supported}"
    )
