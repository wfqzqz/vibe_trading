---
name: position-sizing
description: "Position sizing recipe: compute the Kelly fraction (f*) from win rate and payoff ratio, apply fractional Kelly and sample shrinkage, clamp with a hard risk cap, and write the result to Artifact.position_sizing."
category: strategy
---

# Position Sizing (Kelly Criterion)

## Purpose

This skill turns "how much should I bet on this strategy?" from a judgement
call into a reproducible recipe. It answers a **signal-level exposure** question
— for a single positive-expectancy trade/signal, what fraction of capital to
allocate — which is orthogonal to portfolio weight allocation across *N* assets
(handled by the `backtest/optimizers/` schemes: `equal_volatility`,
`risk_parity`, `mean_variance`, `max_diversification`, `turnover_aware`).

The recipe is a chain with four stages, each of which only ever shrinks the
bet, never grows it:

```
f*  (Kelly)  →  fractional discount  →  shrinkage  →  hard cap  →  f_final
```

- `f*` — the raw Kelly optimum (formula below).
- fractional — `f = c · f*`, `c` discounts for estimation error and log-growth asymmetry.
- shrinkage — `f · n/(n+k)`, pulls low-sample strategies toward no-bet.
- hard cap — `f_final = min(f_kelly, f_cap)`, where `f_cap` comes from notional /
  exposure / leverage / ADV-participation limits.

Scope: this is a **knowledge-layer skill only**. It defines the formula and
parameter contract that downstream work (K-02 optimizer, K-04 research/sim
notional link) implements. It does not itself place orders or write business
code.

## When to Use

- A strategy (or Strategy Development Manager output) has a positive edge and
  you need a per-signal capital fraction instead of an equal-weight default.
- You are documenting or persisting a strategy and must fill
  `Artifact.position_sizing`.
- You are reviewing a proposed bet size and need the four-stage chain applied
  consistently.

Do **not** use this for: cross-asset weight allocation (use the portfolio
optimizers), or live broker execution (out of current scope — see K-04; the
`src/live` gate is reserved, not implemented).

## Stage 1 — Kelly formula (two forms)

### Discrete binary form (default for round-trip trade records)

```
f* = p − q / b
```

- `p` = win rate — probability a trade is a win.
- `q` = `1 − p` — probability a trade is a loss.
- `b` = payoff ratio = `avg_win / avg_loss` (the profit-loss ratio).

`f* ≤ 0` means **no edge → no bet**; return `0`. This is the correct,
always-safe fallback — do not clamp a negative `f*` up to a small positive bet.

### Continuous Gaussian form (for per-period return series)

```
f* = μ / σ²
```

- `μ` = mean per-period return.
- `σ²` = variance of per-period returns.

Use this when the payoff is not a clean binary win/loss (e.g. a daily P&L
series, or a strategy whose per-trade outcome is a continuous return). Use the
binary form when trades resolve to a win/loss against a roughly stable payoff
ratio.

### Data sources (取数路径 — real, not invented)

| Input | Symbol | Source |
|-------|--------|--------|
| win rate | `p` | `backtest/metrics.py::win_rate_and_stats(trades)["win_rate"]` |
| payoff ratio | `b` | `backtest/metrics.py::win_rate_and_stats(trades)["profit_loss_ratio"]` |
| trade records | — | `backtest/models.py::TradeRecord` (`pnl`, `symbol`, `holding_bars`, `exit_reason`) |
| regime | `bull_market` / `bear_market` / `structural` | `src/strategy_discovery/models.py::EvidenceRow.regime` (canonical set `REGIMES`) |

`win_rate_and_stats` already computes `win_rate` (p) and `profit_loss_ratio`
(b = `avg_win / avg_loss`) from a `List[TradeRecord]`, so p and b are
available with zero new dependencies. Note: `EvidenceRow` does **not** yet
carry `win_rate` / `payoff_ratio` fields — that is the K-03 change. Until it
lands, compute p/b directly from `TradeRecord` via `win_rate_and_stats`; the
`regime` dimension comes from `EvidenceRow.regime`.

Always size per `(strategy, regime)` — a strategy's edge is not constant
across `bull_market` / `bear_market` / `structural`, and its p/b should be
measured within the regime it will trade in.

## Stage 2 — Fractional Kelly

```
f = c · f*
```

- `c` = fractional Kelly constant, **default `0.25`**, allowed range `0.1 – 0.5`.

Why the discount is mandatory, not optional:

1. **Edge-estimate error.** `p` and `b` are sample estimates. Sampling noise in
   either pushes the estimated `f*` away from the true optimum, and the
   optimizer is maximally sensitive exactly at `f*`.
2. **Log-growth asymmetry.** Expected log growth `E[log(1 + f·r)]` is concave in
   `f`, peaking at `f*`, but it falls **faster to the right of `f*` than to the
   left**. Over-betting past `f*` first destroys edge and then turns growth
   negative; under-betting only leaves some growth on the table. A deliberate
   discount keeps you on the safe (left) side of the peak.

The asymmetry is why the whole recipe is one-directional: every later stage
only reduces `f`, never raises it.

## Stage 3 — Shrinkage (sample-size shrinkage)

```
f_shrunk = f_kelly · n / (n + k)
```

- `n` = number of completed trades in the `(strategy, regime)` cell (the sample).
- `k` = shrinkage prior constant, **default `10.0`** (`shrink_k`).

As `n → 0` the multiplier `n/(n+k) → 0`, so a strategy with almost no sample
automatically approaches **no-bet** rather than trusting an unreliable p/b.
This mirrors the evidence-quality floor in
`src/strategy_discovery/models.py` (`MIN_TRADES = 10`): below that, evidence is
`insufficient`. With `k = 10`, `n = 10` already shrinks the fraction to half
(`10/20 = 0.5`), and `n = 40` recovers to `0.8` — a smooth handshake between
"not enough data" and "trustworthy sample".

Pass `n_trades=None` to skip shrinkage (only when the sample size is genuinely
unknown and you are documenting a formula, not sizing a real bet).

## Stage 4 — Hard cap (风控兜底)

```
f_final = min(f_kelly, f_cap)
```

`f_cap` is the risk ceiling, not a target. It is the minimum of four
independent limits, each expressed as a fraction of equity:

1. **Single-trade notional cap** — max dollars any one position may take, ÷ equity.
2. **Total exposure cap** — max aggregate gross exposure, ÷ equity.
3. **Leverage cap** — max allowed leverage (a cap of 1.0 ⇒ no leverage ⇒ `f_cap ≤ 1`).
4. **ADV participation cap** — do not deploy a size that moves the market. The
   `execution-model` skill prescribes `max_participation = 0.05` (5% of average
   daily volume) as the ceiling above which square-root impact is mandatory;
   `src/quantlib/impact.py` implements the participation-rate bands (fixed
   slippage < 0.5% ADV, linear impact 0.5–5% ADV, square-root impact > 5% ADV).
   Sizing that breaches ~5% of ADV is rejected regardless of what Kelly asks for.

```python
f_cap = min(
    single_trade_notional_cap / equity,
    total_exposure_cap / equity,
    leverage_cap,
    adv_participation_cap,   # 0.05 from execution-model
)
```

Kelly only shrinks, never amplifies: `f_final = min(f_kelly, f_cap)` is the
last step, and no stage before it may raise the fraction above `f*`.

## Stage 5 — Output spec (write the result)

Persist the conclusion in two places:

1. **`Artifact.position_sizing`** — `src/strategy_store/models.py` field
   `position_sizing: str | None = None`. Write a compact, human-readable recipe
   string so the decision is traceable and re-runnable:

```
position_sizing = "kelly: p=0.60 b=1.50 f*=0.3333 c=0.25 n=42 k=10.0 f_kelly=0.0673 f_cap=0.05 f_final=0.05 (ADV 5% cap binds)"
```

2. **Config** — record the tunable parameters so downstream stages (K-02
   optimizer, K-05 A/B) read them instead of hardcoding:

```json
{
  "position_sizing": {
    "method": "kelly",
    "fractional_c": 0.25,
    "shrink_k": 10.0,
    "f_cap": { "single_trade_notional": 0.10, "total_exposure": 0.50,
               "leverage": 1.0, "adv_participation": 0.05 }
  }
}
```

## Worked example

A `(strategy, regime)` cell has `p = 0.60`, `b = 1.50`, `n = 42` trades, and an
ADV participation cap of 5%:

```python
def kelly_fraction(win_rate, payoff_ratio, *, fractional_c=0.25,
                   n_trades=None, shrink_k=10.0, f_cap=1.0):
    q = 1.0 - win_rate
    if payoff_ratio <= 0 or not win_rate:
        return 0.0
    f_star = win_rate - q / payoff_ratio          # binary Kelly
    if f_star <= 0:
        return 0.0                                # no edge → no bet
    f = fractional_c * f_star                     # fractional discount
    if n_trades is not None:
        f *= n_trades / (n_trades + shrink_k)     # shrinkage
    return min(f, f_cap)                          # hard cap

kelly_fraction(0.60, 1.50, n_trades=42, f_cap=0.05)
# f* = 0.60 − 0.40/1.50 = 0.3333
# fractional (c=0.25)   → 0.0833
# shrinkage (42/52)     → 0.0673
# cap (0.05) binds      → 0.05
```

Gaussian form, per-period `μ = 0.001`, `σ = 0.02`:

```python
mu, sigma = 0.001, 0.02
f_star = mu / sigma**2          # 0.001 / 0.0004 = 2.5  (>1 ⇒ leverage)
f = 0.25 * f_star               # 0.625, still leverage
f_final = min(f, 1.0)           # no-leverage cap → 1.0 (or lower via ADV/notional)
```

This is why fractional + cap are load-bearing: raw Gaussian Kelly routinely
asks for `f* > 1`, which is only safe under leverage the mandate forbids.

## Parameter contract (downstream)

Canonical names that K-02 / K-04 implement against. Do not rename these.

| Parameter | Meaning | Default | Range |
|-----------|---------|---------|-------|
| `fractional_c` | fractional-Kelly constant `c` | `0.25` | `0.1 – 0.5` |
| `shrink_k` | shrinkage prior `k` in `n/(n+k)` | `10.0` | `> 0` |
| `n_trades` | sample size `n` (skip shrinkage if `None`) | `None` | `≥ 0` |
| `f_cap` | hard ceiling from notional/exposure/leverage/ADV | risk-set | `(0, 1]` |
| `f_final` | `min(f_kelly, f_cap)` — the output | — | `[0, f_cap]` |

K-02 (`backtest/optimizers/kelly.py`) exposes
`kelly_fraction(win_rate, payoff_ratio, *, fractional_c=0.25, n_trades=None, shrink_k=10.0)`
and clamps to `[0, f_cap]`. K-04 (`kelly_notional(...)`) returns
`{f_final, notional_usd, risk_budget_usd}` with `notional_usd = f_final × equity`.

## References (upstream code paths)

- `agent/backtest/metrics.py::win_rate_and_stats` — p (`win_rate`) and b
  (`profit_loss_ratio`) source.
- `agent/backtest/models.py::TradeRecord` — trade records (`pnl`, `symbol`,
  `holding_bars`, `exit_reason`).
- `agent/src/strategy_discovery/models.py::EvidenceRow` / `REGIMES` — regime
  vocabulary (`bull_market` / `bear_market` / `structural`) and
  `MIN_TRADES = 10` evidence-quality floor.
- `agent/src/strategy_store/models.py::Artifact.position_sizing` — output field.
- `agent/src/skills/execution-model/SKILL.md` + `agent/src/quantlib/impact.py` —
  `max_participation = 0.05` ADV ceiling and the participation-rate impact bands
  that justify it (fixed < 0.5% / linear 0.5–5% / sqrt > 5%).
