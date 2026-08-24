# K-05 验证报告：regime-conditional + shrinkage + vol-targeting 参数化 + A/B 回测

**结论先行**：不建议「裸」默认启用凯利；建议**有条件默认启用**——以 regime-conditional 为前提（`evidence_quality == adequate` 且样本数达标才启用，否则回落 baseline），默认 `fractional_c = 0.25`，并叠加**组合级 vol-targeting**（目标 σ ≈ 0.10）以把回撤压回优于 baseline 的水平。

本报告的证据来自一个**可复现的合成市场 A/B 回测**（无网络、单一 `seed`，重跑结果 bit-for-bit 一致），以及一组固化该结论的回归单测。真实数据的下一步验证已由 regime-conditional 路径接好（K-03 证据库 → K-05 选择器）。

---

## 1. 参数化（config 可调）

`optimizer: "kelly"` + `optimizer_params` 现可调以下参数（`agent/backtest/optimizers/kelly.py`）：

| 参数 | 默认 | 范围 / 语义 |
|---|---|---|
| `fractional_c` | `0.25` | 0.1 ~ 0.5（fractional-Kelly 折扣） |
| `shrink_k` | `10.0` | 样本收缩先验 `n/(n+k)` |
| `f_cap` | `0.25` | 单信号暴露硬顶；`min(F_CAP, f_cap)`，凯利只缩小不放大 |
| `vol_target` | `None`（关闭） | 组合级年化 σ 目标（非单标的 σ），只降杠杆、不升杠杆 |
| `periods_per_year` | `252` | vol 年化因子 |

`f_cap` 此前是模块常量（`F_CAP`），本次升为参数；`vol_target` 为新增，作用于整个组合的暴露水平（`apply_vol_target` / `vol_target_scale`），与 `equal_volatility` 的「单标的逆波动」正交。

## 2. regime-conditional（证据不足回落 baseline）

新增 `agent/backtest/optimizers/kelly_regime.py`。对 `(strategy_id, regime)` 的 K-03 证据行，仅在以下条件**全部**满足时启用凯利：

1. 存在该 `(strategy_id, regime)` 的证据行；
2. `evidence_quality == adequate`（`marginal` / `insufficient` 一律回落）；
3. `trades_in_regime >= min_trades`（默认 `MIN_TRADES = 10`）；
4. `win_rate` 与 `payoff_ratio` 均非 `None`。

否则返回基线权重（1/N 或 equal_volatility），并携带稳定 `reason` 令牌（`no_evidence` / `quality_below_adequate` / `insufficient_trades` / `missing_pb`），报告可追溯原因。边界（含 all-win 哨兵 `(1.0, 0.0)` → `b=+inf`）由既有 `resolve_kelly_inputs` 收口。

## 3. A/B 回测（可复现）

方法：`agent/backtest/kelly_ab.py`，确定性合成市场（8 资产，0-4 为正 edge、5-7 持平/负 edge，Gaussian 日收益），2520 根日线、lookback 60、单边往返 10 bps/单位换手，seed 7。复用生产 optimizer 与 `backtest.metrics`。

### 3.1 相对配置 A/B（gross = 1）

| strategy | sharpe | max_drawdown | total_return | annual_return | avg_turnover | total_turnover | cost_drag |
|---|---|---|---|---|---|---|---|
| 1/N (baseline) | 0.1375 | -0.2433 | 0.0901 | 0.0087 | 0.000198 | 0.50 | 0.0005 |
| equal_volatility (baseline) | 0.2020 | -0.2239 | 0.1568 | 0.0147 | 0.005684 | 14.32 | 0.0167 |
| kelly c=0.25 | 0.5293 | -0.2774 | 1.0376 | 0.0738 | 0.086755 | 218.62 | 0.4977 |
| kelly c=0.25 + vol_target=0.10 | 0.4062 | **-0.1997** | 0.4397 | 0.0371 | 0.056341 | 141.98 | 0.2196 |

解读：凯利（相对配置）把 Sharpe 从 0.14/0.20 抬到 0.53、总收益从 9%/16% 抬到 104%，代价是**回撤最差**（-27.7%）且换手/成本拖累最高（累计成本拖累 ~50%）。叠加组合级 vol-targeting（σ 目标 0.10）后，回撤收至 **-20%（优于两个 baseline）**，换手与成本拖累减半，Sharpe 仍为 baseline 的 ~2 倍（0.41）。

### 3.2 敏感性：fractional_c ∈ {0.1, 0.25, 0.5}（暴露层）

| fractional_c | f_final (暴露) | sharpe | max_drawdown | total_return | avg_turnover | cost_drag |
|---|---|---|---|---|---|---|
| 0.1 | 0.001733 | 0.5293 | -0.0005 | 0.0015 | 0.000150 | 0.0004 |
| 0.25 | 0.004333 | 0.5293 | -0.0013 | 0.0036 | 0.000376 | 0.0010 |
| 0.5 | 0.008666 | 0.5293 | -0.0026 | 0.0073 | 0.000752 | 0.0019 |

重要发现：**`fractional_c` 在「相对配置层」是不变的**——K-02 的 optimizer 会把行归一化到 gross=1，`c` 被约掉；`c` 真正作用在**暴露层**（K-04 `kelly_notional` 的 `f_final = min(c·f*, f_cap)`），按比例线性缩放总暴露、收益、回撤与换手，而不改变 Sharpe（杠杆中性）。合成市场 edge 较小（win_rate 0.5045 / payoff 1.017），故单笔 f* 仅 ~1.7%，`f_final` 在 0.17%~0.87% 之间——这如实反映了「日频二值凯利分数天然很小」。

## 4. 结论与参数推荐

- **不建议裸默认启用**：raw Kelly 的相对配置收益最差回撤 + 最高换手/成本，若在证据不足时也强上凯利，风险无补偿。
- **建议有条件默认启用**：
  1. **regime-conditional 为硬前提**（证据 adequate + 样本达标才启用，否则回落 baseline）；
  2. 默认 `fractional_c = 0.25`（区间 0.1~0.5，保守方取 0.1）；
  3. 叠加**组合级 vol-targeting**，σ 目标按策略自然波动校准（本报告演示 0.10），以约束回撤；
  4. `f_cap` 保持 0.25 绝对硬顶（凯利只缩小）。

## 5. 证据可追溯

- 实现：`agent/backtest/optimizers/kelly.py`（f_cap / vol-target）、`agent/backtest/optimizers/kelly_regime.py`（regime-conditional）、`agent/backtest/kelly_ab.py`（A/B harness）。
- 单测：`agent/tests/test_kelly_params.py`、`agent/tests/test_kelly_regime.py`、`agent/tests/test_kelly_ab.py`（连同既有 `test_kelly_optimizer.py` / `test_kelly_notional.py`，共 107 项通过）。
- 复现：`python -m backtest.kelly_ab`（仓库根，`PYTHONPATH=agent`）；测试：`python -m pytest agent/tests/test_kelly_*.py`。

## 6. 已知边界（本报告不证明什么）

合成市场为独立 Gaussian、无相关结构、无肥尾/regime 切换；成本为换手线性模型，未建模滑点/撮合。这些是「凯利优化器层」的受控对比，不是实盘承诺。生产启用前，应以 K-03 证据库真实数据走 regime-conditional 路径复核。
