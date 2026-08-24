"""Reserved live-broker hook for Kelly position sizing (K-04, DORA-162).

This module is a deliberate no-op boundary. Kelly sizing today feeds ONLY the
research / simulation chain (Shadow Account / paper): the backtest engine's
sizing weights are Kelly-scaled into a notional *intent* by
``backtest/optimizers/kelly.py::kelly_position_intents`` and consumed by
paper/reporting paths — never by a broker.

Live broker execution is out of scope by design (DORA-122 decision point 3:
research + simulation only; no new live order path). When that decision is
revisited, wire :func:`kelly_sizing_hook` between intent generation and the
existing mandate gate (:func:`src.live.sdk_order_gate.execute_live_order`) so
that a Kelly-sized notional flows through the SAME pre-trade checks every live
order must already pass (mandate, expiry, halt, notional/exposure/leverage
caps, daily count). Until then this function must never be imported by any
runtime path, and it raises to make accidental wiring loud rather than silently
sizing a live order.

No business logic lives here. ``backtest/optimizers/kelly.py`` is the single
source of truth for the sizing math; this module is the future *gate*, not the
recipe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only, keeps the module import cheap
    from src.live.enforcement import OrderIntent


def kelly_sizing_hook(
    *,
    intent: "OrderIntent",
    equity: float,
    win_rate: float | None,
    payoff_ratio: float | None,
    strategy_vol: float,
    fractional_c: float = 0.25,
    shrink_n: float | None = None,
    f_cap: float = 0.25,
) -> dict[str, Any]:
    """Reserved: apply Kelly sizing to a live intent before the mandate gate.

    Intended contract (for when live trading is enabled, DORA-122 decision
    point 3 is revisited):

    1. Compute ``kelly_notional(equity, win_rate, payoff_ratio,
       strategy_vol=strategy_vol, fractional_c=fractional_c,
       shrink_n=shrink_n, f_cap=f_cap)``.
    2. Replace ``intent.notional_usd`` with the resulting hard-cap-constrained
       notional (``f_final <= f_cap`` by construction).
    3. Hand the sized intent to ``execute_live_order`` so it passes the same
       mandate gate as every other live order.

    Args:
        intent: The normalized :class:`~src.live.enforcement.OrderIntent` to size.
        equity: Account equity in USD.
        win_rate / payoff_ratio: Per-strategy p/b pair.
        strategy_vol: Annualized strategy volatility.
        fractional_c / shrink_n / f_cap: Kelly recipe parameters.

    Raises:
        NotImplementedError: always — the live gate is deliberately unwired.
            Wire it behind a live-trading decision before any use; do not catch
            this and proceed to order.

    Returns:
        Never returns; the live path is disabled until an explicit decision
        enables it.
    """
    raise NotImplementedError(
        "kelly_sizing_hook is reserved (DORA-162 / DORA-122 decision point 3): "
        "Kelly sizing feeds the research/simulation chain only; live broker "
        "execution is not enabled yet."
    )


__all__ = ["kelly_sizing_hook"]
