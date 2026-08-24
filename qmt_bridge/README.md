# QMT Bridge — 只读行情桥（D-01）

独立只读服务（Windows 单机），与 FastAPI agent 进程完全解耦。仅 `import
xtquant.xtdata`，**结构性不引入 `xttrader`**（无任何交易/写接口）。按需/定时把
xtdata 行情落盘 parquet 到 `~/.vibe-trading/cache/loaders/miniqmt/`，并写入复权
（前复权为主）+ 停牌/涨跌停/除权除息元数据列，供 `china_a.py` 消费。

## 运行

```powershell
# 启动只读 HTTP 服务（loopback + token）
python -m qmt_bridge serve

# 按需落盘单只标的日线（默认 qfq）
python -m qmt_bridge cache --symbol 600519.SH --start 2024-01-01 --end 2024-06-30

# 只读能力 manifest
python -m qmt_bridge manifest

# token 管理
python -m qmt_bridge token generate
python -m qmt_bridge token show
```

## HTTP 契约（只读）

| 端点 | 说明 |
|---|---|
| `GET /health` | 桥 + xtdata 客户端状态（运行时探测） |
| `GET /v1/manifest` | 只读能力 manifest（`write_capabilities: false`） |
| `GET /v1/quotes/daily?symbol=&start=&end=&adjust=qfq` | 日线 OHLCV + 元数据列 |
| `GET /v1/quotes/minute?symbol=&period=1m&...` | 分钟线 |
| `GET /v1/quotes/tick?symbol=...` | tick（可选，分级保留） |
| `GET /v1/meta?symbol=` | 复权/停牌/涨跌停/除权除息 |

- 无任何 write/order 端点；非 GET/HEAD/OPTIONS 一律 405（中间件结构性拒绝）。
- 本地 loopback + token：`Authorization: Bearer <token>` 或 `X-API-Token`。
- 统一 provenance：`{source, symbol, timeframe, adjust, volume_unit, is_final}`。
- xtdata 不可用时返回 `503 {"unavailable": true, ...}`，不崩溃。

## 凭据（无明文落盘）

`~/.vibe-trading/qmt-bridge/secrets.v1.json` 只存 base64 密文（Windows DPAPI，
`CryptProtectData`，与桌面端 `safeStorage` 同一机制）；日志不落明文。字段：
`api_token`（loopback token）、`qmt_account_id`（可选，tick/元数据订阅用）。

## 缓存契约（供 D-02 `miniqmt` loader 消费）

- 源名 `miniqmt`，目录 `~/.vibe-trading/cache/loaders/miniqmt/<key>.parquet`
  + `<key>.parquet.json` 元数据侧车。
- key 与 `backtest/loaders/base.py` **字节一致**（`fields=None`、仅“已结算日”
  可缓存、`VIBE_TRADING_DATA_CACHE` / `VIBE_TRADING_DATA_CACHE_ROOT` 开关）。
- 仅 `adjust=qfq` 落盘（前复权为主）；侧车额外记 `adjust` / `source`。
- 列：`open/high/low/close/volume/amount` + `pre_close/suspended/limit_up/
  limit_down/ex_dividend/adj_factor`。
