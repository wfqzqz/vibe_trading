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


def kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    *,
    fractional_c: float = 0.25,
    n_trades: float | None = None,
    shrink_k: float = 10.0,
) -> float:
    """Binary Kelly fraction with fractional discount, shrinkage and hard cap.

    ``f* = p - q / b`` where ``p`` is the win rate, ``q = 1 - p`` and ``b`` is
    the payoff ratio (avg_win / avg_loss). A non-positive ``f*`` means no edge,
    so the result is ``0`` (never a positive bet on a losing edge). The raw
    fraction is then discounted by ``fractional_c``, shrunk toward zero by
    ``n/(n+k)`` when a sample size is known, and finally clamped to
    ``[0, F_CAP]``.

    Args:
        win_rate: Probability of a win, ``p`` in ``[0, 1]``.
        payoff_ratio: Ratio of average win to average loss, ``b``; must be
            positive. ``+inf`` is allowed (no losses observed) and yields
            ``f* = p``.
        fractional_c: Fractional-Kelly constant ``c`` (default 0.25).
        n_trades: Sample size ``n`` for shrinkage; ``None`` skips shrinkage.
        shrink_k: Shrinkage prior ``k`` in ``n/(n+k)`` (default 10.0).

    Returns:
        The sized Kelly fraction, in ``[0, F_CAP]``.

    Invalid inputs (NaN, non-finite, ``b <= 0``, ``p`` outside ``[0, 1]``,
    degenerate sample sizes) return ``0.0`` — a safe "no bet" fallback rather
    than raising.
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

    return min(max(f, 0.0), F_CAP)


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
        **kwargs: Any,
    ) -> None:
        super().__init__(lookback=lookback, **kwargs)
        self.fractional_c = fractional_c
        self.shrink_k = shrink_k

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
    (``fractional_c``, ``shrink_k``) flow through ``optimizer_params``.
    """
    return KellyOptimizer(lookback=lookback, **params).optimize(ret, pos, dates)
