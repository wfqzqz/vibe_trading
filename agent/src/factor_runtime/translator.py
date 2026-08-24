"""alpha.lang expression translator (DORA-124 §3.4 module D — F-02).

Turns a py-alpha-lib factor expression into a runnable Python function body via
the ``alpha.lang`` translator (``alpha.to_python``). The generated code targets
the py-alpha-lib ``ExecContext``:

* variables become ``ctx('NAME')``,
* operator calls become ``ctx.FUNC(...)``,
* ternary / logical / power render through ``numpy`` helpers.

Only this module touches the online translator. The snapshot store
(``snapshot.py``) persists the *output*, and the loader imports it without ever
re-translating — the offline-reproducibility contract from DORA-124 §4.3.

DORA-188 (F-03 review P1): every ``ctx.OP(...)`` call in the translated body is
validated against the installed ``ExecContext``'s real operator surface
(:func:`validate_operator_surface`), so doc-claimed-but-unimplemented aliases
(``REF`` / ``HHV`` / ``LLV`` / ``HHVBARS`` / ``LLVBARS`` in 0.3.0) are rejected
at registration instead of failing at compute with a raw ``AttributeError``.
"""

from __future__ import annotations

from src.factor_runtime.availability import require_available
from src.factor_runtime.operators import validate_operator_surface

#: Function name + parameter contract shared with ``snapshot.py``.
FUNCTION_NAME = "compute"
CONTEXT_PARAM = "ctx"

#: Map expression identifiers (variables and operators alike) onto the
#: ``ExecContext``'s UPPERCASE convention: ``close`` -> ``CLOSE``,
#: ``ref`` -> ``REF``, ``ma`` -> ``MA``. ``alpha.lang`` applies this to both
#: ``NAME`` tokens and function names, so the two surfaces stay consistent.
NAME_CONVERTOR = str.upper


class FactorTranslationError(ValueError):
    """Raised when an expression cannot be parsed/translated by alpha.lang."""


def translate_expression(expression: str, *, func_name: str = FUNCTION_NAME) -> str:
    """Translate an alpha expression into a ``def compute(ctx):`` body.

    Args:
        expression: A py-alpha-lib ``alpha.lang`` expression.
        func_name: The generated function name (defaults to ``compute``).

    Returns:
        The generated function definition (4-space indented, trailing newline).

    Raises:
        FactorTranslationError: blank input, or the expression is not valid
            ``alpha.lang`` (lark parse/lex failure).
        FactorOperatorError: the translated body calls an operator the
            installed ``ExecContext`` does not implement (DORA-188).
    """
    if not expression or not expression.strip():
        raise FactorTranslationError("factor expression must not be empty")

    # The online translator lives on the ``alpha`` package returned by the
    # availability gate; on a degraded host this raises
    # ``FactorRuntimeUnavailableError`` (the "prompt Docker" contract).
    alpha = require_available()
    try:
        body = alpha.lang.to_python(
            func_name,
            expression,
            indent=0,
            indent_by="    ",
            as_function=True,
            name_convertor=NAME_CONVERTOR,
            optimize=False,
        )
    except Exception as exc:  # noqa: BLE001 — lark raises several parse/lex types
        raise FactorTranslationError(
            f"alpha.lang could not translate expression: {exc}"
        ) from exc

    if not body or not body.strip():
        raise FactorTranslationError("alpha.lang produced an empty translation")

    # F-03 review P1 (DORA-188): the doc-claimed REF/HHV/LLV/HHVBARS/LLVBARS
    # aliases are not implemented on the 0.3.0 ExecContext — reject them at
    # registration with a clear 400, not a compute-time raw AttributeError.
    validate_operator_surface(body, alpha)
    return body
