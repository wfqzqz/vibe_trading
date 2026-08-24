"""Kelly position-sizing optimizer: per-symbol signal-level exposure scaling.

The five other optimizers in this package answer a *portfolio* question — how to
split weight across N assets by risk / correlation. Kelly answers a *signal*
question — for a single positive-expectancy signal, what fraction of capital to
bet — and is therefore a per-symbol scaling layer on the target weights, not a
cross-asset rebalancer.

For each active symbol the causal return window is reduced to a binary
win/loss picture (win = positive return, loss = negative return), from which the
win rate ``p`` and payoff ratio ``b = avg_win / avg_loss`` are estimated. The
binary Kelly fraction ``f* = p - q/b`` is then discounted by the fractional
constant ``c``, shrunk toward no-bet by ``n/(n+k)``, capped, and used to scale
the symbol's target weight. The row is renormalised to a valid weight panel
(gross = 1), so the output satisfies the same engine contract as the other
optimizers.

Formula and parameter contract live in ``src/skills/position-sizing/SKILL.md``:
fractional Kelly ``c`` (default 0.25), sample shrinkage ``n/(n+k)`` (``k``
default 10.0), and a hard cap ``f_cap``. Kelly only ever shrinks a bet, never
amplifies it.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest.optimizers.base import BaseOptimizer

# Default hard ceiling for a single signal's exposure, as a fraction of equity:
# ``f_final = min(f_kelly, F_CAP)``. A tighter per-deployment cap comes from the
# notional / exposure / leverage / ADV-participation limits documented in the
# ``position-sizing`` SKILL.
F_CAP = 0.25

#: ADV participation ceiling from the ``execution-model`` SKILL: never deploy a
#: single position larger than 5% of average daily volume. Sizing that breaches
#: ~5% of ADV is rejected regardless of what Kelly asks for; this constant is the
#: fraction-of-equity proxy used by :func:`f_cap_from_limits`.
ADV_PARTICIPATION_CAP = 0.05


def kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    *,
    fractional_c: float = 0.25,
    n_trades: float | None = None,
    shrink_k: float = 10.0,
    f_cap: float = F_CAP,
) -> float:
    """Binary Kelly fraction with fractional discount, shrinkage and hard cap.

    ``f* = p - q / b`` where ``p`` is the win rate, ``q = 1 - p`` and ``b`` is
    the payoff ratio (avg_win / avg_loss). A non-positive ``f*`` means no edge,
    so the result is ``0`` (never a positive bet on a losing edge). The raw
    fraction is then discounted by ``fractional_c``, shrunk toward zero by
    ``n/(n+k)`` when a sample size is known, and finally clamped to
    ``[0, min(F_CAP, f_cap)]``.

    Args:
        win_rate: Probability of a win, ``p`` in ``[0, 1]``.
        payoff_ratio: Ratio of average win to average loss, ``b``; must be
            positive. ``+inf`` is allowed (no losses observed) and yields
            ``f* = p``.
        fractional_c: Fractional-Kelly constant ``c`` (default 0.25).
        n_trades: Sample size ``n`` for shrinkage; ``None`` skips shrinkage.
        shrink_k: Shrinkage prior ``k`` in ``n/(n+k)`` (default 10.0).
        f_cap: Hard ceiling for a single signal's exposure, a fraction of
            equity. The effective ceiling is ``min(F_CAP, f_cap)`` — ``F_CAP``
            (the module-level absolute ceiling) can never be raised by a
            larger ``f_cap``, so Kelly only shrinks. An invalid / non-positive
            / non-finite ``f_cap`` fails closed to ``0.0``.

    Returns:
        The sized Kelly fraction, in ``[0, min(F_CAP, f_cap)]``.

    Invalid inputs (NaN, non-finite, ``b <= 0``, ``p`` outside ``[0, 1]``,
    degenerate sample sizes, unusable ``f_cap``) return ``0.0`` — a safe
    "no bet" fallback rather than raising.
    """
    if isinstance(win_rate, (bool, np.bool_)) or isinstance(
        payoff_ratio, (bool, np.bool_)
    ):
        return 0.0
    try:
        p = float(win_rate)
        b = float(payoff_ratio)
    except (TypeError, ValueError):
        return 0.0

    # p must be a finite probability; b must be strictly positive (NaN and
    # b <= 0 fall back to no bet). b == +inf is valid: q / b == 0, so f* = p.
    if not np.isfinite(p) or not (0.0 <= p <= 1.0):
        return 0.0
    if np.isnan(b) or b <= 0.0:
        return 0.0

    q = 1.0 - p
    f_star = p - q / b
    if f_star <= 0.0:
        return 0.0  # no edge → no bet

    f = fractional_c * f_star

    if n_trades is not None:
        if isinstance(n_trades, (bool, np.bool_)):
            return 0.0
        try:
            n = float(n_trades)
            k = float(shrink_k)
        except (TypeError, ValueError):
            return 0.0
        # n <= 0 means no usable sample; a non-finite or negative shrink prior
        # would NaN or amplify, both of which violate "Kelly only shrinks".
        if not np.isfinite(n) or n <= 0.0 or not np.isfinite(k) or k < 0.0:
            return 0.0
        f *= n / (n + k)

    cap = F_CAP
    if isinstance(f_cap, (bool, np.bool_)):
        return 0.0
    try:
        cap_value = float(f_cap)
    except (TypeError, ValueError):
        return 0.0
    # f_cap must be finite and positive to form a valid ceiling; a non-finite or
    # non-positive cap cannot bound anything, so fail closed to no bet. The
    # effective ceiling is min(F_CAP, f_cap): the module absolute ceiling is
    # never raised by a larger f_cap, so Kelly only shrinks.
    if not np.isfinite(cap_value) or cap_value <= 0.0:
        return 0.0
    cap = min(F_CAP, cap_value)

    return min(max(f, 0.0), cap)


def _binary_stats(returns: np.ndarray) -> tuple[float, float, int]:
    """Estimate (win_rate, payoff_ratio, n) from a per-symbol return window.

    Each finite observation is a binary outcome: ``return > 0`` is a win,
    ``return < 0`` is a loss. ``win_rate = wins / n`` and ``payoff_ratio =
    avg_win / avg_loss`` mirror ``backtest/metrics.py::win_rate_and_stats``.

    Args:
        returns: 1-D array of per-period returns (may contain NaN).

    Returns:
        ``(win_rate, payoff_ratio, n)``. A window with no finite observations
        yields ``(0.0, 0.0, 0)``. A window with wins but no losses yields
        ``payoff_ratio = inf`` (Kelly then gives ``f* = win_rate``).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n == 0:
        return 0.0, 0.0, 0

    wins = r[r > 0.0]
    losses = r[r < 0.0]
    win_rate = float(wins.size) / n

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = abs(float(losses.mean())) if losses.size else 0.0

    if avg_loss > 0.0:
        payoff_ratio = avg_win / avg_loss
    elif wins.size:
        payoff_ratio = float("inf")  # all (non-zero) returns are wins
    else:
        payoff_ratio = 0.0  # no wins and no losses → no edge

    return win_rate, payoff_ratio, n


def portfolio_returns(pos: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Portfolio return per bar implied by a weight panel (no lookahead).

    ``pos[t-1]`` weights are applied to ``ret[t]`` — the same next-bar-open
    convention the engine uses, so the first bar is ``0``. Only columns present
    in both frames contribute. This is the portfolio-level return stream used
    for vol-targeting: it is a single portfolio ``sigma`` derived from the
    combined weights, never a per-symbol sigma.

    Args:
        pos: Position-weight panel (index=timestamp, columns=codes).
        ret: Per-asset return panel (index=timestamp, columns=codes).

    Returns:
        Per-bar portfolio return series indexed like ``pos``.
    """
    prev = pos.shift(1).fillna(0.0)
    common = [c for c in pos.columns if c in ret.columns]
    aligned = ret.reindex(pos.index).fillna(0.0)
    return (prev[common] * aligned[common]).sum(axis=1)


def vol_target_scale(
    portfolio_returns: pd.Series,
    *,
    target_vol: float,
    periods_per_year: int = 252,
) -> float:
    """De-levering scale to bring realized portfolio vol to ``target_vol``.

    ``scale = clip(target_vol / realized_vol, 0, 1)``. The cap at ``1.0``
    encodes "vol-targeting only de-risks": a portfolio that is already below
    target is left alone rather than levered up (leverage is a separate,
    mandate-gated decision — see the ``position-sizing`` SKILL). A missing /
    non-positive / non-finite ``target_vol``, or a window whose volatility is
    unmeasurable, yields ``1.0`` (no change) rather than an invented factor.

    Args:
        portfolio_returns: Trailing portfolio return series (per-bar returns).
        target_vol: Annualized portfolio volatility target, as a decimal
            fraction (e.g. ``0.15``). Must be positive and finite.
        periods_per_year: Bars per year for annualisation (default 252).

    Returns:
        A scale factor in ``[0, 1]``.
    """
    if isinstance(target_vol, (bool, np.bool_)):
        return 1.0
    try:
        target = float(target_vol)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(target) or target <= 0.0:
        return 1.0

    values = portfolio_returns.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1.0
    realized = float(np.std(values, ddof=1))
    if not np.isfinite(realized) or realized <= 0.0:
        return 1.0
    annual = realized * np.sqrt(max(1, int(periods_per_year)))
    if not np.isfinite(annual) or annual <= 0.0:
        return 1.0
    return float(np.clip(target / annual, 0.0, 1.0))


def apply_vol_target(
    pos: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    target_vol: float,
    periods_per_year: int = 252,
    lookback: int = 60,
) -> pd.DataFrame:
    """Scale a weight panel so trailing portfolio vol targets ``target_vol``.

    For each bar ``t >= lookback`` the portfolio return implied by the panel's
    own weights over ``[t-lookback, t)`` is annualised and the row is multiplied
    by :func:`vol_target_scale`. Rows before ``lookback`` are left unchanged
    (no usable history). Because the scale factor is ``<= 1`` and applied to the
    whole row, gross exposure can only shrink, and the engine's later
    ``gross <= 1`` normalisation is unaffected. The result is a pure function of
    ``pos`` and ``ret`` — no lookahead into the decision bar.

    Args:
        pos: Position-weight panel.
        ret: Per-asset return panel (dates x codes) aligned to ``pos``.
        target_vol: Annualized portfolio volatility target (decimal fraction).
        periods_per_year: Bars per year for annualisation.
        lookback: Trailing window (in bars) over which realized vol is measured.

    Returns:
        The vol-targeted weight panel (same shape/columns as ``pos``).
    """
    if isinstance(target_vol, (bool, np.bool_)):
        return pos
    try:
        target = float(target_vol)
    except (TypeError, ValueError):
        return pos
    if not np.isfinite(target) or target <= 0.0:
        return pos

    result = pos.copy()
    prev = pos.shift(1).fillna(0.0)
    common = [c for c in pos.columns if c in ret.columns]
    aligned = ret.reindex(pos.index).fillna(0.0)
    port_ret = (prev[common] * aligned[common]).sum(axis=1)

    for i in range(lookback, len(result.index)):
        scale = vol_target_scale(
            port_ret.iloc[i - lookback : i],
            target_vol=target,
            periods_per_year=periods_per_year,
        )
        if scale < 1.0:
            result.iloc[i] = result.iloc[i] * scale
    return result


class KellyOptimizer(BaseOptimizer):
    """Per-symbol Kelly exposure scaling on top of the raw target weights.

    Overrides :meth:`optimize` because the scaling semantics differ from the
    other optimizers: instead of replacing the target weight with a normalized
    risk-based weight, each target weight keeps its magnitude and sign and is
    scaled by its own Kelly fraction, then the row is renormalised to gross 1.
    """

    def __init__(
        self,
        lookback: int = 60,
        fractional_c: float = 0.25,
        shrink_k: float = 10.0,
        f_cap: float = F_CAP,
        vol_target: float | None = None,
        periods_per_year: int = 252,
        **kwargs: Any,
    ) -> None:
        super().__init__(lookback=lookback, **kwargs)
        self.fractional_c = fractional_c
        self.shrink_k = shrink_k
        self.f_cap = f_cap
        self.vol_target = vol_target
        self.periods_per_year = periods_per_year

    def _build_context(
        self, window: pd.DataFrame, active: List[str]
    ) -> "Dict[str, Any] | None":
        """Per-symbol binary Kelly inputs from the causal return window.

        Args:
            window: Return window (dates x active codes), already causally
                sliced by :meth:`optimize`.
            active: Active codes.

        Returns:
            Dict with aligned arrays ``win_rate``, ``payoff_ratio`` and
            ``n_trades`` (one entry per active code), or None if the window is
            empty.
        """
        if window.empty:
            return None
        win_rates = np.empty(len(active), dtype=float)
        payoffs = np.empty(len(active), dtype=float)
        n_trades = np.empty(len(active), dtype=float)
        for j, code in enumerate(active):
            win_rates[j], payoffs[j], n_trades[j] = _binary_stats(
                window[code].to_numpy(dtype=float)
            )
        return {
            "win_rate": win_rates,
            "payoff_ratio": payoffs,
            "n_trades": n_trades,
        }

    def _calc_weights(self, ctx: Dict[str, Any]) -> np.ndarray:
        """Per-symbol Kelly fractions (not yet normalized to a simplex).

        Returns:
            Array of fractions in ``[0, F_CAP]``, aligned with ``ctx`` order.
        """
        win_rates = ctx["win_rate"]
        payoffs = ctx["payoff_ratio"]
        n_trades = ctx["n_trades"]
        fractions = np.empty(len(win_rates), dtype=float)
        for j in range(len(win_rates)):
            n = int(n_trades[j]) if n_trades[j] > 0 else 0
            fractions[j] = kelly_fraction(
                win_rates[j],
                payoffs[j],
                fractional_c=self.fractional_c,
                n_trades=n,
                shrink_k=self.shrink_k,
                f_cap=self.f_cap,
            )
        return fractions

    def optimize(
        self,
        ret: pd.DataFrame,
        pos: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Scale target weights by per-symbol Kelly fractions, then normalize.

        Mirrors :meth:`BaseOptimizer.optimize`'s causal-window discipline
        (only returns strictly before the decision bar are visible) but, for
        each date, multiplies the target weight by the symbol's Kelly fraction
        and renormalizes the row to gross 1 instead of replacing the weight
        with ``sign * risk_weight``.
        """
        codes = pos.columns.tolist()
        if len(codes) <= 1:
            return pos

        result = pos.copy()
        for i, dt in enumerate(dates):
            active = [c for c in codes if abs(pos.at[dt, c]) > 1e-9]
            if not active or i < self.lookback:
                continue

            # Same no-look-ahead rule as BaseOptimizer.optimize: ret[dt] is a
            # close-to-close return not observable until that bar closes.
            history = ret.loc[ret.index < dt, active]
            window = history.tail(self.lookback)
            if len(window) < max(self.lookback // 2, 5):
                continue

            ctx = self._build_context(window, active)
            if ctx is None:
                continue

            fractions = self._calc_weights(ctx)
            if fractions is None or len(fractions) != len(active):
                continue

            scaled = np.empty(len(active), dtype=float)
            for j, c in enumerate(active):
                scaled[j] = pos.at[dt, c] * fractions[j]

            total = float(np.abs(scaled).sum())
            if total > 1e-12:
                scaled = scaled / total

            for j, c in enumerate(active):
                result.at[dt, c] = scaled[j]

        if self.vol_target is not None:
            # Portfolio-level vol targeting: de-lever the whole panel toward
            # ``vol_target`` using the panel's own trailing portfolio sigma
            # (combined weights), never a per-symbol sigma. Applied after the
            # per-symbol Kelly scaling + row renormalisation, so it shapes total
            # exposure without disturbing the relative edge allocation.
            result = apply_vol_target(
                result,
                ret,
                target_vol=self.vol_target,
                periods_per_year=self.periods_per_year,
                lookback=self.lookback,
            )

        return result


def optimize(
    ret: pd.DataFrame,
    pos: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = 60,
    **params: Any,
) -> pd.DataFrame:
    """Module-level entry: per-symbol Kelly-scaled positions.

    Select via ``optimizer: "kelly"`` in ``config.json``; extra keyword args
    (``fractional_c``, ``shrink_k``, ``f_cap``, ``vol_target``,
    ``periods_per_year``) flow through ``optimizer_params``.
    """
    return KellyOptimizer(lookback=lookback, **params).optimize(ret, pos, dates)


# --------------------------------------------------------------------------- #
# K-04 — research/simulation notional link
# --------------------------------------------------------------------------- #
# The optimizer above scales a *weight panel*. The research/simulation chain
# (Shadow Account / paper) needs a *dollar notional* per position, so the same
# Kelly recipe is lifted to the notional layer here:
#
#   f_final = min(f_kelly, f_cap)          # f_cap from the four risk limits
#   notional_usd = f_final * equity
#   risk_budget_usd = notional_usd * strategy_vol
#
# These are pure functions: no I/O, no order path. They take the backtest
# engine's sizing weights (``target_positions.csv`` from
# ``backtest/engines/base.py::_write_artifacts``) plus equity and a per-strategy
# p/b pair and return a notional *intent*. Real broker execution stays out of
# scope (DORA-122 decision point 3); the reserved ``src.live.kelly_sizing_hook``
# is the future gate that would route these intents through the live mandate
# checks.


def resolve_kelly_inputs(
    win_rate: float | None,
    payoff_ratio: float | None,
) -> tuple[float, float] | None:
    """Reconcile the evidence layer's p/b with :func:`kelly_fraction`'s contract.

    ``strategy_discovery/models.py::win_rate_and_payoff`` (a faithful mirror of
    ``backtest/metrics.py::win_rate_and_stats``) reports an all-win regime as
    ``payoff_ratio = 0.0`` — its ``1e-10`` sentinel collapses to zero. But
    :func:`kelly_fraction` requires ``b > 0`` (``b = +inf`` for "no losses"), so
    feeding that ``0.0`` straight through would silently size an all-win regime
    as "no edge → no bet". This helper closes that contract gap (the K-03 → K-04
    handoff):

    * ``win_rate is None`` or ``payoff_ratio is None`` (no usable P&L) → ``None``
      (callers treat this as no edge and never guess a number).
    * all-win regime (``win_rate == 1.0`` and ``payoff_ratio == 0.0``) →
      ``(1.0, +inf)`` so Kelly yields ``f* = p`` instead of ``0``.
    * anything else → passed through unchanged; :func:`kelly_fraction` remains
      the single validator for the remaining degenerate inputs.

    Args:
        win_rate: Decimal win probability ``p`` (``None`` when unknown).
        payoff_ratio: Payoff ratio ``b = avg_win / avg_loss`` (``None`` when
            unknown).

    Returns:
        ``(win_rate, payoff_ratio)`` ready for :func:`kelly_fraction`, or
        ``None`` when there is no usable p/b pair.
    """
    if win_rate is None or payoff_ratio is None:
        return None
    try:
        p = float(win_rate)
        b = float(payoff_ratio)
    except (TypeError, ValueError):
        return None
    # The evidence-layer all-win sentinel (payoff 0.0 with a perfect win rate)
    # must map to b = +inf BEFORE kelly_fraction, which reads b <= 0 as no edge.
    if p == 1.0 and b == 0.0:
        return p, float("inf")
    return p, b


def _no_bet() -> dict[str, float]:
    """The fail-closed sizing result: no fraction, no notional, no risk budget."""
    return {"f_final": 0.0, "notional_usd": 0.0, "risk_budget_usd": 0.0}


def _positive_float(value: object) -> float | None:
    """Coerce ``value`` to a finite strictly-positive float, else ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _risk_budget(notional_usd: float, strategy_vol: float) -> float:
    """Dollar risk budget for a notional at ``strategy_vol``, fail-closed.

    ``notional_usd * strategy_vol`` is the one-standard-deviation dollar risk of
    the position. A missing / non-finite / negative ``strategy_vol`` yields
    ``0.0`` rather than a fabricated or negative budget.
    """
    vol = _positive_float(strategy_vol) if strategy_vol is not None else None
    if vol is None:
        # A strategy with unknown or non-positive volatility has no measurable
        # risk budget; report 0.0 instead of inventing one. (This is the only
        # place a negative is rejected without fail-closing the whole sizing —
        # the notional itself is still valid.)
        return 0.0
    return notional_usd * vol


def f_cap_from_limits(
    equity: float,
    *,
    single_trade_notional_cap: float | None = None,
    total_exposure_cap: float | None = None,
    leverage_cap: float | None = None,
    adv_participation_cap: float | None = ADV_PARTICIPATION_CAP,
) -> float:
    """Compute the hard-cap fraction ``f_cap`` from the four risk limits.

    Mirrors the ``position-sizing`` SKILL Stage 4 recipe verbatim::

        f_cap = min(
            single_trade_notional_cap / equity,   # USD → fraction of equity
            total_exposure_cap / equity,          # USD → fraction of equity
            leverage_cap,                         # already a fraction (1.0 = no leverage)
            adv_participation_cap,                # 0.05 from execution-model
        )

    Each limit that is ``None``, non-finite, or non-positive is treated as "no
    limit" (skipped) rather than binding at zero. A non-positive / non-finite
    ``equity`` returns ``0.0`` (no bet possible). The default
    ``adv_participation_cap`` keeps a 5%-of-equity ceiling in place even when
    every other limit is absent, matching the SKILL's worked example where the
    ADV cap is the binding constraint.

    Args:
        equity: Account equity in USD (must be finite and positive).
        single_trade_notional_cap: Max USD for a single position, or ``None``.
        total_exposure_cap: Max aggregate gross exposure in USD, or ``None``.
        leverage_cap: Max allowed leverage as a fraction (``1.0`` = no
            leverage), or ``None``.
        adv_participation_cap: ADV participation ceiling as a fraction of
            equity (defaults to :data:`ADV_PARTICIPATION_CAP` = 0.05).

    Returns:
        The minimum of the applicable limits (each expressed as a fraction of
        equity), or ``0.0`` when equity is unusable or no positive limit is
        supplied. In practice the default ADV ceiling keeps it at or below
        ``0.05``; :func:`kelly_notional` then bounds ``f_final`` to
        ``min(f_kelly, f_cap)`` so Kelly never amplifies.
    """
    if equity is None:
        return 0.0
    eq = _positive_float(equity)
    if eq is None:
        return 0.0

    limits: list[float] = []
    for usd_cap in (single_trade_notional_cap, total_exposure_cap):
        value = _positive_float(usd_cap)
        if value is not None:
            limits.append(value / eq)
    for frac_cap in (leverage_cap, adv_participation_cap):
        value = _positive_float(frac_cap)
        if value is not None:
            limits.append(value)

    if not limits:
        return 0.0
    return min(limits)


def kelly_notional(
    equity: float,
    win_rate: float | None,
    payoff_ratio: float | None,
    *,
    strategy_vol: float,
    fractional_c: float = 0.25,
    shrink_n: float | None = None,
    f_cap: float = F_CAP,
) -> dict[str, float]:
    """Kelly-sized notional for a single signal from equity and a p/b pair.

    The K-04 counterpart to :func:`kelly_fraction`: the same four-stage recipe
    (binary Kelly → fractional discount → shrinkage → hard cap) lifted to
    dollars, producing the notional *intent* the research/simulation chain
    consumes.

        f_final = min(f_kelly, f_cap)          # f_cap binds, never amplifies
        notional_usd = f_final * equity
        risk_budget_usd = notional_usd * strategy_vol

    ``f_final <= f_cap`` holds by construction (``min``), and ``f_final`` is
    additionally bounded above by :data:`F_CAP` through :func:`kelly_fraction`.
    Every invalid input degrades to :func:`_no_bet` — a zero result, never a
    raise — matching the fail-closed contract of the rest of this module.

    Args:
        equity: Account equity in USD (finite and positive).
        win_rate: Decimal win probability ``p`` (``None`` → no edge).
        payoff_ratio: Payoff ratio ``b`` (``None`` → no edge; an all-win
            ``(1.0, 0.0)`` pair is mapped to ``b = +inf`` via
            :func:`resolve_kelly_inputs`).
        strategy_vol: Annualized strategy volatility as a decimal fraction
            (e.g. ``0.20``); used only for the risk-budget output.
        fractional_c: Fractional-Kelly constant ``c`` (default 0.25).
        shrink_n: Sample size ``n`` for ``n/(n+k)`` shrinkage (default ``None``
            skips shrinkage; the shrinkage prior stays at its ``10.0`` default).
        f_cap: Hard ceiling from :func:`f_cap_from_limits` (default
            :data:`F_CAP`).

    Returns:
        ``{"f_final", "notional_usd", "risk_budget_usd"}`` with
        ``f_final in [0, min(f_cap, F_CAP)]``.
    """
    inputs = resolve_kelly_inputs(win_rate, payoff_ratio)
    if inputs is None:
        return _no_bet()
    p, b = inputs

    eq = _positive_float(equity)
    if eq is None:
        return _no_bet()

    f_kelly = kelly_fraction(
        p,
        b,
        fractional_c=fractional_c,
        n_trades=shrink_n,
        shrink_k=10.0,
    )

    cap = _positive_float(f_cap) if f_cap is not None else None
    if cap is None:
        return _no_bet()
    f_final = min(f_kelly, cap)

    notional_usd = f_final * eq
    return {
        "f_final": f_final,
        "notional_usd": notional_usd,
        "risk_budget_usd": _risk_budget(notional_usd, strategy_vol),
    }


def kelly_position_intents(
    weights: dict[str, float],
    equity: float,
    win_rate: float | None,
    payoff_ratio: float | None,
    *,
    strategy_vol: float,
    fractional_c: float = 0.25,
    shrink_n: float | None = None,
    f_cap: float = F_CAP,
) -> dict[str, Any]:
    """Convert backtest sizing weights into Kelly-scaled notional intents.

    The research/simulation integration point (K-04): the backtest engine's
    sizing weights (its ``target_positions.csv`` weight panel, one row per
    active symbol) are scaled by the strategy-level ``f_final`` from
    :func:`kelly_notional` and expressed as a per-symbol notional.

        notional(symbol) = weight(symbol) * f_final * equity

    Weights are kept signed (a short carries a negative notional); the aggregate
    ``notional_usd`` / ``risk_budget_usd`` are the gross (absolute) sums so a
    long/short book is not understated. Non-finite or zero weights are skipped.
    This is a pure function — it produces intents only, never an order.

    Args:
        weights: Mapping of symbol → target weight from the backtest sizing
            layer (``target_positions.csv``). Accepts any mapping with
            ``.items()`` (``dict``, ``pandas.Series``, a ``DataFrame`` row).
        equity: Account equity in USD.
        win_rate / payoff_ratio: Per-strategy p/b pair (see
            :func:`kelly_notional`).
        strategy_vol: Annualized strategy volatility (risk-budget output).
        fractional_c / shrink_n / f_cap: Kelly recipe parameters (see
            :func:`kelly_notional`).

    Returns:
        ``{"f_final", "notional_usd", "risk_budget_usd", "positions"}`` where
        ``positions`` is a list of ``{"symbol", "weight", "notional_usd",
        "risk_budget_usd"}``, one per active symbol.
    """
    sizing = kelly_notional(
        equity,
        win_rate,
        payoff_ratio,
        strategy_vol=strategy_vol,
        fractional_c=fractional_c,
        shrink_n=shrink_n,
        f_cap=f_cap,
    )
    f_final = sizing["f_final"]
    # f_final * equity == sizing["notional_usd"], so per-symbol notional is just
    # the weight scaled by the strategy's total Kelly notional.
    unit_notional = sizing["notional_usd"]
    unit_risk = sizing["risk_budget_usd"]

    positions: list[dict[str, Any]] = []
    gross = 0.0
    for symbol, weight in weights.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(w) or w == 0.0:
            continue
        positions.append(
            {
                "symbol": str(symbol),
                "weight": w,
                "notional_usd": w * unit_notional,
                "risk_budget_usd": abs(w) * unit_risk,
            }
        )
        gross += abs(w)

    return {
        "f_final": f_final,
        "notional_usd": gross * unit_notional,
        "risk_budget_usd": gross * unit_risk,
        "positions": positions,
    }
