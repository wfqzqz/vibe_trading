"""Factor compute / evaluate adapter to the py-alpha-lib ``ExecContext`` (F-03).

Bridges the wide OHLCV panel the Alpha Zoo consumes (``dict[str, DataFrame]``,
index = date, columns = code) into the long-format input py-alpha-lib's
``ExecContext`` expects, runs the snapshot's ``compute(ctx)``, and reshapes the
flat result back to the wide shape so the SAME IC/IR + layered-return math
(``src.factors.factor_analysis_core``) that the zoo bench uses applies verbatim.
That shared math is the "same-口径 comparison" contract from DORA-124 §3.4:

    * new factors run through py-alpha-lib (``source="py-alpha-lib"``),
    * preset factors run through ``Registry.compute`` (pandas path),
    * both are evaluated with ``compute_ic_series`` / ``compute_group_equity``.

Flat-layout contract (verified against py-alpha-lib 0.3.0's ``ExecContext``):
``alpha.context.ExecContext`` extracts OHLCV columns as flat ``np.ndarray`` of
length ``n_codes * n_dates`` ordered *code-major, date-ascending* (row ``i`` is
``code_idx * n_dates + date_idx``). :func:`panel_to_long` therefore emits rows in
exactly that order so a dense panel round-trips via ``array.reshape(n_codes,
n_dates).T``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.factor_runtime.availability import require_available

logger = logging.getLogger(__name__)

#: ExecContext's expected long-format OHLCV column names (lowercase).
_CTX_COLS: tuple[str, ...] = ("open", "high", "low", "close", "vol", "vwap")

#: Wide panel key -> long/ExecContext column. Note ``volume`` -> ``vol``.
_PANEL_TO_CTX: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "vol",
    "vwap": "vwap",
}


class FactorComputeError(RuntimeError):
    """Raised when a factor cannot be computed/evaluated (bad panel, bad result)."""


def _validate_panel(panel: dict[str, Any]) -> pd.DataFrame:
    """Return ``panel["close"]`` after checking it is a non-empty wide frame."""
    if not isinstance(panel, dict):
        raise FactorComputeError("panel must be a dict of wide DataFrames keyed by field")
    close = panel.get("close")
    if close is None or not isinstance(close, pd.DataFrame) or close.empty:
        raise FactorComputeError("panel must contain a non-empty 'close' DataFrame")
    if close.index.empty or close.columns.empty:
        raise FactorComputeError("panel 'close' must have a non-empty index and columns")
    return close


def _field_flat(
    panel: dict[str, Any],
    dates: list[Any],
    codes: list[Any],
    key: str,
    n_dates: int,
    n_codes: int,
) -> np.ndarray:
    """Return ``key`` as a flat float array in code-major, date-ascending order.

    A missing/absent field becomes an all-NaN array of the right length so the
    panel stays dense (ExecContext uses row order for a complete panel).
    """
    df = panel.get(key)
    if df is None or not isinstance(df, pd.DataFrame):
        return np.full(n_codes * n_dates, np.nan, dtype=np.float64)
    wide = df.reindex(index=dates, columns=codes).to_numpy(dtype=np.float64)
    return wide.T.ravel()  # (n_dates, n_codes) -> code-major


def panel_to_long(panel: dict[str, Any]) -> pd.DataFrame:
    """Convert a wide OHLCV panel into py-alpha-lib's long input frame.

    Returns a DataFrame with columns ``securityid, tradetime, open, high, low,
    close, vol, vwap`` ordered code-major / date-ascending. ``volume`` is mapped
    to ``vol``; a missing (or all-NaN) ``vwap`` is synthesized as the typical
    price ``(O + H + L + C) / 4`` — matching the sp500/btc panel loaders.
    """
    close = _validate_panel(panel)
    dates = list(close.index)
    codes = list(close.columns)
    n_dates = len(dates)
    n_codes = len(codes)

    flat: dict[str, np.ndarray] = {}
    for panel_key, ctx_col in _PANEL_TO_CTX.items():
        flat[ctx_col] = _field_flat(panel, dates, codes, panel_key, n_dates, n_codes)

    # VWAP synthesis fallback. Reuse the panel's vwap when it exists and is not
    # entirely missing; otherwise fall back to the typical price.
    vwap_df = panel.get("vwap")
    if vwap_df is None or not isinstance(vwap_df, pd.DataFrame) or np.isnan(flat["vwap"]).all():
        open_arr = flat["open"].reshape(n_codes, n_dates)
        high_arr = flat["high"].reshape(n_codes, n_dates)
        low_arr = flat["low"].reshape(n_codes, n_dates)
        close_arr = flat["close"].reshape(n_codes, n_dates)
        flat["vwap"] = ((open_arr + high_arr + low_arr + close_arr) / 4.0).ravel()

    securityid = [str(code) for code in codes for _ in range(n_dates)]
    tradetime = [date for _ in range(n_codes) for date in dates]

    return pd.DataFrame(
        {
            "securityid": securityid,
            "tradetime": tradetime,
            "open": flat["open"],
            "high": flat["high"],
            "low": flat["low"],
            "close": flat["close"],
            "vol": flat["vol"],
            "vwap": flat["vwap"],
        }
    )


def reshape_factor_result(values: Any, index: Any, columns: Any) -> pd.DataFrame:
    """Reshape a flat ExecContext result into a wide DataFrame (date x code).

    The result must be a flat array of length ``n_dates * n_codes`` in
    code-major / date-ascending order; anything else (e.g. a scalar expression)
    is a compute error rather than a silent mis-shape. The output reuses the
    caller's ``index``/``columns`` so the panel's index metadata (e.g. a daily
    ``DatetimeIndex.freq``) is preserved.
    """
    n_dates = len(index)
    n_codes = len(columns)
    arr = np.asarray(values, dtype=np.float64)
    expected = n_dates * n_codes
    if arr.shape != (expected,):
        raise FactorComputeError(
            f"factor compute returned shape {arr.shape}; expected ({expected},) "
            "(flat, code-major). A scalar expression is not a valid factor."
        )
    if np.isinf(arr).any():
        raise FactorComputeError("factor compute produced +/- inf (registry contract forbids it)")
    wide = arr.reshape(n_codes, n_dates).T  # (n_dates, n_codes)
    return pd.DataFrame(wide, index=index, columns=columns)


def _exec_context_class(alpha_module: Any) -> type:
    """Resolve ``ExecContext`` from the ``alpha`` package (context then top-level)."""
    cls = getattr(getattr(alpha_module, "context", None), "ExecContext", None)
    if cls is None:
        cls = getattr(alpha_module, "ExecContext", None)
    if cls is None or not callable(cls):
        raise FactorComputeError("alpha package does not expose an ExecContext")
    return cls


def _unsupported_operator_error(exc: AttributeError) -> str:
    """Format an ExecContext ``AttributeError`` into a readable compute error.

    py-alpha-lib 0.3.0 implements fewer operators than its docs advertise
    (``REF``/``HHV``/``LLV``/``HHVBARS``/``LLVBARS`` are doc-claimed aliases
    that are not on the real ``ExecContext``). A snapshot calling one fails
    here with ``AttributeError``; surfacing it as ``FactorComputeError`` maps
    it to 422 (never a generic 500) and keeps the CLI error readable.
    """
    name = getattr(exc, "name", None)
    if isinstance(name, str) and name:
        hint = (
            f"operator {name!r} is not supported by the py-alpha-lib "
            "ExecContext (documented alias not implemented in 0.3.0)"
        )
    else:
        hint = "factor compute raised an AttributeError"
    return f"factor compute failed: {hint} ({exc})"


def compute_factor(
    store: Any,
    factor_id: str,
    version: int,
    panel: dict[str, Any],
) -> pd.DataFrame:
    """Load a snapshot, run it over ``panel`` via ExecContext, return wide values.

    Args:
        store: The :class:`~src.factor_runtime.snapshot.SnapshotStore`.
        factor_id: Registered factor id (validated by the store's path gate).
        version: Positive snapshot version (validated by the store).
        panel: Wide OHLCV panel (same shape the Alpha Zoo consumes).

    Returns:
        A ``DataFrame`` (index = date, columns = code) of factor values aligned
        to ``panel["close"]`` — the same shape ``Registry.compute`` returns.

    Raises:
        FactorRuntimeUnavailableError: py-alpha-lib absent (degradation).
        SnapshotNotFoundError / SnapshotValidationError: snapshot missing/tampered.
        FactorComputeError: bad panel, an unreshapable factor result, or the
            snapshot calls an operator the ExecContext does not implement
            (``AttributeError`` → ``FactorComputeError``, DORA-188).
    """
    snapshot = store.load(factor_id, version)
    alpha_module = require_available()
    exec_context_cls = _exec_context_class(alpha_module)

    close = _validate_panel(panel)
    long_df = panel_to_long(panel)
    ctx = exec_context_cls(long_df)
    try:
        values = snapshot.compute(ctx)
    except AttributeError as exc:
        # Doc-claimed-but-unimplemented operator (e.g. REF/HHV in 0.3.0):
        # a readable 422 FactorComputeError, never a generic 500.
        raise FactorComputeError(_unsupported_operator_error(exc)) from exc
    return reshape_factor_result(values, close.index, close.columns)


def evaluate_factor(
    store: Any,
    factor_id: str,
    version: int,
    panel: dict[str, Any],
    *,
    return_df: pd.DataFrame,
    n_groups: int = 5,
) -> dict[str, Any]:
    """Evaluate a factor: IC/IR + layered returns on the shared zoo math.

    Uses ``compute_ic_series`` / ``compute_group_equity`` from
    ``src.factors.factor_analysis_core`` — the exact functions the zoo bench
    calls — so the numbers are directly comparable to preset factors.

    Args:
        store: The snapshot store.
        factor_id: Registered factor id.
        version: Snapshot version.
        panel: Wide OHLCV panel.
        return_df: Forward returns (index = date, columns = code) — the caller
            must derive these with ``alpha_bench_tool._compute_forward_returns``
            so both paths share one forward-returns definition.
        n_groups: Number of quantile groups for the layered backtest.

    Returns:
        ``{factor_id, version, source, py_alpha_lib, shape, ic, layered_returns}``.

    Raises:
        FactorComputeError: factor not evaluable (empty IC series / no valid
            cross-section dates).
    """
    if n_groups < 1:
        raise FactorComputeError(f"n_groups must be >= 1, got {n_groups}")

    from src.factor_runtime.availability import py_alpha_lib_version
    from src.factors.factor_analysis_core import compute_group_equity, compute_ic_series

    factor_df = compute_factor(store, factor_id, version, panel)
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        raise FactorComputeError(
            "IC series empty — factor and forward returns share no evaluable "
            "dates/codes (need >= 5 valid instruments per bar)"
        )

    ic_mean = float(ic.mean())
    ic_std = float(ic.std()) if len(ic) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0

    equity = compute_group_equity(factor_df, return_df, n_groups)
    if equity.empty:
        raise FactorComputeError(
            "layered backtest empty — insufficient valid cross-section dates"
        )

    final_nav = {col: round(float(equity[col].iloc[-1]), 4) for col in equity.columns}
    long_short_spread = round(float(equity.iloc[-1, -1] - equity.iloc[-1, 0]), 4)

    return {
        "factor_id": factor_id,
        "version": version,
        "source": "py-alpha-lib",
        "py_alpha_lib": py_alpha_lib_version(),
        "shape": [factor_df.shape[0], factor_df.shape[1]],
        "ic": {
            "mean": round(ic_mean, 6),
            "std": round(ic_std, 6),
            "ir": round(ir, 4),
            "positive_ratio": round(float((ic > 0).mean()), 4),
            "count": int(len(ic)),
        },
        "layered_returns": {
            "n_groups": n_groups,
            "index": [str(d) for d in equity.index],
            "equity": {col: equity[col].round(6).tolist() for col in equity.columns},
            "final_nav": final_nav,
            "long_short_spread": long_short_spread,
        },
    }
