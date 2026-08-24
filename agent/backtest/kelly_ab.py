"""Reproducible A/B backtest of Kelly vs. baseline sizing (K-05).

This is the empirical half of the Kelly epic's final stage. It does NOT touch
the network: it generates a deterministic synthetic market, runs the production
optimizer layer (the exact code the engine loads from ``optimizer: "..."``) on
one shared raw signal, simulates the portfolio that results, and reports the
four headline numbers the epic asks for — ``sharpe`` / ``max_drawdown`` /
``turnover`` / cost drag — across Kelly, the 1/N baseline and the
equal-volatility baseline, plus a ``fractional_c`` sensitivity sweep.

Two distinct layers are compared, because the K-02/K-04 split makes them
different questions and they must not be conflated:

1. **Relative-allocation A/B** (gross = 1). The backtest engine's optimizer
   path answers "which assets, in what relative weight". Kelly renormalises to
   gross 1 (K-02), so here it competes with 1/N and equal-volatility purely on
   allocation quality. This is the headline A/B table.
2. **``fractional_c`` sensitivity** (exposure layer). K-02's renormalisation
   makes ``fractional_c`` invariant at the relative-weight layer; ``c`` acts on
   TOTAL exposure through the K-04 notional layer
   (:func:`backtest.optimizers.kelly.kelly_notional`). The sensitivity sweep
   reports ``f_final(c)`` and the resulting full-pipeline metrics so the effect
   of ``c`` is measured where it actually lives.

Scope is the *optimizer / signal-exposure* layer, which is exactly where Kelly
lives. Execution-level effects (lot rounding, slippage, partial fills) are
orthogonal to Kelly and deliberately excluded; costs are modeled as a
per-unit-turnover drag so the comparison is not flattered by ignoring friction.

Reproducibility: the market, the signal and every run are functions of a single
``seed``; there is no clock, no I/O and no randomness outside ``numpy``'s seeded
generator. Re-running ``python -m backtest.kelly_ab`` reproduces the tables
bit-for-bit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import calc_metrics, calc_turnover_series
from backtest.optimizers.equal_volatility import EqualVolatilityOptimizer
from backtest.optimizers.kelly import F_CAP, KellyOptimizer, kelly_notional

#: Transaction cost per unit of turnover, in basis points (round-trip).
DEFAULT_COST_BPS = 10.0

#: Default synthetic universe: annual drift / vol per asset (decimals). Assets
#: 0-4 carry positive edge, assets 5-7 are flat-to-negative — a market where a
#: signal-level sizer has something real to act on, and 1/N does not.
DEFAULT_DRIFT = [0.15, 0.12, 0.09, 0.05, 0.02, 0.00, -0.02, -0.05]
DEFAULT_VOL = [0.25, 0.22, 0.20, 0.28, 0.30, 0.32, 0.35, 0.24]

#: Fractional-Kelly sensitivity grid required by the epic.
FRACTIONAL_GRID = (0.1, 0.25, 0.5)


def synthetic_market(
    seed: int = 7,
    n_bars: int = 2520,
    drift: list[float] | None = None,
    vol: list[float] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Deterministic synthetic per-asset daily returns (no lookahead issues).

    Each asset follows independent Gaussian daily returns ``N(mu_d, sigma_d^2)``
    with ``mu_d = drift / periods_per_year`` and ``sigma_d = vol / sqrt(ppy)``.
    Columns are ``A0..A(n-1)``; the index is a business-day range. ``drift`` /
    ``vol`` default to :data:`DEFAULT_DRIFT` / :data:`DEFAULT_VOL` when ``None``
    (lengths must match).
    """
    drift = DEFAULT_DRIFT if drift is None else list(drift)
    vol = DEFAULT_VOL if vol is None else list(vol)
    if len(drift) != len(vol):
        raise ValueError("drift and vol must have the same length")

    rng = np.random.default_rng(seed)
    n_assets = len(drift)
    daily_drift = np.asarray(drift, dtype=float) / periods_per_year
    daily_vol = np.asarray(vol, dtype=float) / math.sqrt(periods_per_year)
    returns = rng.normal(daily_drift, daily_vol, size=(n_bars, n_assets))
    dates = pd.bdate_range("2021-01-04", periods=n_bars)
    return pd.DataFrame(
        returns, index=dates, columns=[f"A{i}" for i in range(n_assets)]
    )


def raw_signal(ret: pd.DataFrame) -> pd.DataFrame:
    """The shared raw signal: hold every asset at equal target weight.

    Constant 1.0 per asset — the optimizer layer is the only difference between
    the strategies, so the comparison isolates exactly what Kelly changes.
    """
    return pd.DataFrame(1.0, index=ret.index, columns=ret.columns)


def finalize_weights(pos: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the engine's post-optimizer normalisation (gross <= 1)."""
    scale = pos.abs().sum(axis=1).clip(lower=1.0)
    return pos.div(scale, axis=0)


def _optimize_weights(
    ret: pd.DataFrame,
    pos: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    optimizer: str,
    lookback: int,
    fractional_c: float,
    vol_target: float | None,
) -> pd.DataFrame:
    """Run the named optimizer exactly as the engine's ``_load_optimizer`` would."""
    if optimizer == "1n":
        return finalize_weights(pos)
    if optimizer == "equal_volatility":
        out = EqualVolatilityOptimizer(lookback=lookback).optimize(ret, pos, dates)
        return finalize_weights(out)
    if optimizer == "kelly":
        out = KellyOptimizer(
            lookback=lookback,
            fractional_c=fractional_c,
            vol_target=vol_target,
        ).optimize(ret, pos, dates)
        return finalize_weights(out)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def pooled_pb(returns: pd.DataFrame, edge_columns: list[str]) -> tuple[float, float, int]:
    """Strategy-level (win_rate, payoff_ratio, n) from pooled edge returns.

    Mirrors ``backtest/metrics.py::win_rate_and_stats``: wins are positive
    returns, losses negative, ``payoff = avg_win / avg_loss`` with the ``1e-10``
    sentinel collapsing to ``0.0`` when there are no losses.
    """
    pooled = returns[edge_columns].to_numpy(dtype=float).ravel()
    pooled = pooled[np.isfinite(pooled)]
    n = int(pooled.size)
    if n == 0:
        return 0.0, 0.0, 0
    wins = pooled[pooled > 0.0]
    losses = pooled[pooled < 0.0]
    win_rate = float(wins.size) / n
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = abs(float(losses.mean())) if losses.size else 1e-10
    payoff = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
    return win_rate, payoff, n


def kelly_exposure(
    win_rate: float,
    payoff_ratio: float,
    *,
    fractional_c: float,
    f_cap: float = F_CAP,
) -> float:
    """Total-exposure fraction ``f_final`` from the K-04 notional layer."""
    return kelly_notional(
        1.0,
        win_rate,
        payoff_ratio,
        strategy_vol=0.15,  # risk-budget only; does not affect f_final
        fractional_c=fractional_c,
        f_cap=f_cap,
    )["f_final"]


@dataclass
class StrategyResult:
    """One strategy's headline metrics (all reproducible, no I/O)."""

    name: str
    exposure: float
    sharpe: float
    max_drawdown: float
    total_return: float
    annual_return: float
    avg_turnover: float
    total_turnover: float
    cost_drag_return: float
    final_value: float


def simulate(
    weights: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    cost_bps: float,
    periods_per_year: int,
) -> dict[str, Any]:
    """Simulate a portfolio from a weight panel and per-asset returns.

    ``gross_ret[t] = sum_i weight[t-1, i] * ret[t, i]`` (no lookahead). Turnover
    uses :func:`backtest.metrics.calc_turnover_series`; each unit of turnover
    incurs ``cost_bps`` (round-trip) of drag. ``equity`` compounds the net
    return from 1.0. Metrics reuse :func:`backtest.metrics.calc_metrics`.
    """
    prev = weights.shift(1).fillna(0.0)
    common = [c for c in weights.columns if c in ret.columns]
    aligned = ret.reindex(weights.index).fillna(0.0)
    gross_ret = (prev[common] * aligned[common]).sum(axis=1)

    turnover = calc_turnover_series(weights)
    cost_fraction = turnover * (cost_bps / 1e4)
    net_ret = gross_ret - cost_fraction

    equity = (1.0 + net_ret).cumprod()
    metrics = calc_metrics(equity, [], 1.0, periods_per_year, positions=weights)

    gross_total = float((1.0 + gross_ret).prod() - 1.0)
    net_total = metrics["total_return"]
    return {
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": net_total,
        "annual_return": metrics["annual_return"],
        "avg_turnover": metrics["avg_turnover"],
        "total_turnover": metrics["total_turnover"],
        "cost_drag_return": gross_total - net_total,
        "final_value": metrics["final_value"],
    }


def run_strategy(
    name: str,
    ret: pd.DataFrame,
    pos: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    optimizer: str,
    exposure: float,
    lookback: int,
    fractional_c: float,
    vol_target: float | None,
    cost_bps: float,
    periods_per_year: int,
) -> StrategyResult:
    """Run one strategy end-to-end and return its :class:`StrategyResult`.

    ``exposure`` scales the (gross-1) relative-weight panel to total gross
    exposure — ``1.0`` for the fully-invested baselines and the relative-weight
    A/B, ``f_final(c)`` for the exposure-layer sensitivity sweep.
    """
    relative = _optimize_weights(
        ret,
        pos,
        dates,
        optimizer=optimizer,
        lookback=lookback,
        fractional_c=fractional_c,
        vol_target=vol_target,
    )
    weights = relative * exposure
    stats = simulate(weights, ret, cost_bps=cost_bps, periods_per_year=periods_per_year)
    return StrategyResult(name=name, exposure=exposure, **stats)


def ab_compare(
    *,
    seed: int = 7,
    n_bars: int = 2520,
    lookback: int = 60,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: int = 252,
    vol_target: float | None = None,
    drift: list[float] | None = None,
    vol: list[float] | None = None,
) -> dict[str, Any]:
    """Run the full Kelly-vs-baseline comparison.

    Returns a dict with ``"meta"``, ``"ab"`` (relative-allocation A/B) and
    ``"sensitivity"`` (``fractional_c`` sweep at the exposure layer), each row a
    plain dict.
    """
    drift = DEFAULT_DRIFT if drift is None else list(drift)
    vol = DEFAULT_VOL if vol is None else list(vol)
    ret = synthetic_market(
        seed=seed, n_bars=n_bars, drift=drift, vol=vol, periods_per_year=periods_per_year
    )
    pos = raw_signal(ret)
    dates = ret.index

    def run(name: str, optimizer: str, exposure: float, fc: float = 0.25, vt: float | None = None) -> dict:
        result = run_strategy(
            name,
            ret,
            pos,
            dates,
            optimizer=optimizer,
            exposure=exposure,
            lookback=lookback,
            fractional_c=fc,
            vol_target=vt,
            cost_bps=cost_bps,
            periods_per_year=periods_per_year,
        )
        return {
            "strategy": name,
            "exposure": round(result.exposure, 6),
            "sharpe": round(result.sharpe, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "total_return": round(result.total_return, 4),
            "annual_return": round(result.annual_return, 4),
            "avg_turnover": round(result.avg_turnover, 6),
            "total_turnover": round(result.total_turnover, 4),
            "cost_drag_return": round(result.cost_drag_return, 4),
            "final_value": round(result.final_value, 6),
        }

    ab = [
        run("1/N (baseline)", "1n", 1.0),
        run("equal_volatility (baseline)", "equal_volatility", 1.0),
        run("kelly c=0.25", "kelly", 1.0),
    ]
    if vol_target is not None:
        ab.append(run(f"kelly c=0.25 + vol_target={vol_target}", "kelly", 1.0, 0.25, vol_target))

    # Exposure-layer sensitivity: p/b pooled over the positive-edge assets.
    edge_columns = [f"A{i}" for i, d in enumerate(drift) if d > 0]
    p, b, n = pooled_pb(ret, edge_columns)
    sensitivity = []
    for c in FRACTIONAL_GRID:
        exposure = kelly_exposure(p, b, fractional_c=c)
        sensitivity.append(run(f"kelly c={c}", "kelly", exposure, c))

    return {
        "meta": {
            "seed": seed,
            "n_bars": n_bars,
            "lookback": lookback,
            "cost_bps": cost_bps,
            "periods_per_year": periods_per_year,
            "vol_target": vol_target,
            "f_cap": F_CAP,
            "drift": drift,
            "vol": vol,
            "edge_win_rate": round(p, 4),
            "edge_payoff_ratio": round(b, 4),
            "edge_samples": n,
        },
        "ab": ab,
        "sensitivity": sensitivity,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render the comparison result as a self-contained Markdown report."""
    meta = result["meta"]
    lines: list[str] = []
    lines.append("# Kelly vs. baseline — reproducible A/B backtest (K-05)")
    lines.append("")
    lines.append(
        "Deterministic synthetic market (seeded Gaussian daily returns), "
        f"{meta['n_bars']} bars / {meta['periods_per_year']} bars-per-year, "
        f"lookback {meta['lookback']}, round-trip cost {meta['cost_bps']} bps per unit turnover, "
        f"seed {meta['seed']}. `f_cap` = {meta['f_cap']}."
    )
    lines.append("")
    lines.append("## A/B comparison — relative allocation (gross = 1)")
    lines.append("")
    lines.append(
        "| strategy | sharpe | max_drawdown | total_return | annual_return | avg_turnover | total_turnover | cost_drag |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in result["ab"]:
        lines.append(
            f"| {row['strategy']} | {row['sharpe']} | {row['max_drawdown']} | {row['total_return']} | "
            f"{row['annual_return']} | {row['avg_turnover']} | {row['total_turnover']} | {row['cost_drag_return']} |"
        )
    lines.append("")
    lines.append("## Sensitivity — fractional_c sweep (exposure layer)")
    lines.append("")
    lines.append(
        f"Edge evidence pooled over positive-drift assets: win_rate = {meta['edge_win_rate']}, "
        f"payoff_ratio = {meta['edge_payoff_ratio']}, samples = {meta['edge_samples']}."
    )
    lines.append("")
    lines.append(
        "| fractional_c | f_final (exposure) | sharpe | max_drawdown | total_return | avg_turnover | cost_drag |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in result["sensitivity"]:
        lines.append(
            f"| {row['strategy'].removeprefix('kelly c=')} | {row['exposure']} | {row['sharpe']} | "
            f"{row['max_drawdown']} | {row['total_return']} | {row['avg_turnover']} | {row['cost_drag_return']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the report; optionally write ``--output-json`` / ``--output-md``."""
    import argparse

    parser = argparse.ArgumentParser(description="Kelly vs. baseline A/B backtest (K-05)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-bars", type=int, default=2520)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--vol-target", type=float, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args(argv)

    result = ab_compare(
        seed=args.seed,
        n_bars=args.n_bars,
        lookback=args.lookback,
        cost_bps=args.cost_bps,
        vol_target=args.vol_target,
    )
    markdown = render_markdown(result)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as handle:
            handle.write(markdown + "\n")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
