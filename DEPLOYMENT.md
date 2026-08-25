# 部署指南（本地 + Docker 双路径）· P-03

> 依据 DORA-124 §1.2 部署拓扑与 §3.4 容器策略落地。目标：本地 Windows 与 Docker
> 两条路径「一键可起」，凭据分层清晰，因子在 Docker 路径完整可用。
>
> 范围：只覆盖部署/配置/文档，不引入任何业务代码变更。相关代码位置以
> `path/to/file` 内联标注。

## 1. 双路径总览

```
                ┌───────────────────────────────────────────────┐
                │  Windows 宿主机（必须：xtdata Windows-only）        │
                │                                                 │
                │  国金 QMT/miniQMT 客户端 ──(xtquant.xtdata)──▶    │
                │  QMT Bridge 独立只读进程（127.0.0.1:8100，凭据加密） │
                │      │ 落盘 parquet → cache/loaders/miniqmt/       │
                └──────┼──────────────────────────────────────────┘
                       │  共享目录 ~/.vibe-trading/cache/loaders/
            ┌──────────┴────────────────┐
            │   本地 Windows 路径          │    Docker 路径（因子工作必选）│
            │  agent(FastAPI) + CLI + Web │  docker compose up --build    │
            │  factor-runtime 优雅降级      │  agent + web（factor-runtime 为 │
            │  （提示需 Docker 因子服务）    │   agent 进程内模块，py-alpha-lib │
            └────────────────────────────┘  ==0.3.0 已打进镜像）            │
                                           └─────────────────────────────┘
```

关键点（与 DORA-124 一致）：

- **QMT Bridge 是唯一必须跑在 Windows 宿主机且进程隔离的组件**：`xtdata` 与 QMT
  客户端常驻、非线程安全，不能进 FastAPI 进程（上游回测在沙箱子进程跑、agent 主
  进程是 FastAPI）。它只 `import xtquant.xtdata`，结构性不引入 `xttrader`。
- **factor-runtime 是 agent 进程内模块，不是独立微服务**（DORA-124 修订 v1.1 附带
  澄清 1）：compose 里没有单独的 `factor-runtime` 服务，`vibe-trading` 镜像已把
  `py-alpha-lib==0.3.0` 打进镜像（`Dockerfile` 第 48–57 行），因此因子能力只在容器
  内完整可用。
- **共享缓存**：桥主动写 `~/.vibe-trading/cache/loaders/miniqmt/`，本地 agent 与
  Docker 容器读同一目录（`docker-compose.yml` 的 bind mount + 容器内
  `VIBE_TRADING_DATA_CACHE_ROOT=/home/vibe/.vibe-trading/cache/loaders`）。

## 2. 前置条件

- Windows 10/11（QMT Bridge 需要）、Docker Desktop（Docker Engine ≥ 20.10 /
  Compose v2，随 Docker Desktop 自带）。
- 本地路径另需 Python 3.11+；Web 前端 dev server 需 Node ≥ 22.22（仅开发模式）。
- 一个 LLM API Key（OpenRouter/DeepSeek/Ollama 等，见 `agent/.env.example`）。

## 3. Docker 路径（推荐，因子工作必选）

### 3.1 一键启动

```powershell
git clone https://github.com/wfqzqz/vibe_trading.git
cd vibe_trading
copy agent\.env.example agent\.env          # 编辑：解注释 LLM provider 并填 API Key
docker compose up --build                   # 构建并启动 agent（含前端静态资源）
```

> 顶层的 `.env.example` 可复制为 `.env` 覆盖端口/缓存路径（见 §6）；不复制也能用
> 默认值一键启动。

> **构建代理说明（DORA-251）**：Docker Desktop 会自动向构建注入指向
> `http.docker.internal:3128` 的 `HTTP(S)_PROXY` 构建参数；该代理对 PyPI
> 间歇性故障——表现为构建时随机报 `Could not find a version ... (from
> versions: none)`（python-telegram-bot / pycryptodome 先后中招；容器直连
> PyPI 正常）。`docker-compose.yml` 已为 `vibe-trading` 服务固化空代理构建参数
> （构建直连 PyPI），无需额外操作；若你的网络确实需经代理访问 PyPI，用
> `docker compose build --build-arg HTTP_PROXY=<url> --build-arg HTTPS_PROXY=<url>`
> 覆盖。仍偶遇上述报错时重跑一次构建即可（与锁文件无关，DORA-251 实证）。

### 3.2 验证

```powershell
# 存活探针（镜像内 HEALTHCHECK 也打这个端点；/health 为旧别名）
curl http://localhost:8899/live
# 打开 Web UI（agent 同时托管 frontend/dist）
start http://localhost:8899
```

- 前端 dev server 可选（改前端源码时）：`docker compose --profile frontend up`，
  打开 `http://localhost:5899`。
- 数据持久化在命名卷里（`vibe-home` 等），`docker compose down -v` 才会清空；
  日常 `git pull && docker compose up --build` 保留全部状态。

### 3.3 因子能力验证（容器内 py-alpha-lib 可用）

```powershell
docker compose exec vibe-trading python -c "import src.factor_runtime as fr; print(fr.runtime_status())"
```

输出应含 `py-alpha-lib==0.3.0`（`available`）。本地 Windows 上同样的调用会抛出
`FactorRuntimeUnavailableError`（§4.3），这是设计内的优雅降级。

## 4. 本地 Windows 路径（agent + CLI + Web）

### 4.1 安装

```powershell
cd vibe_trading
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .                            # 基础依赖（不含 py-alpha-lib，Windows 无 3.11/3.12 wheel）
copy agent\.env.example agent\.env          # 编辑：解注释 LLM provider 并填 API Key
```

> 若 PowerShell 拒绝执行 `Activate.ps1`，先运行
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`。

### 4.2 启动（一键脚本或手动）

**一键（推荐，等价于 Linux/macOS 的 `scripts/dev`）：**

```powershell
.\scripts\dev.ps1 up       # 后台启动 backend(8899) + frontend(5899)
.\scripts\dev.ps1 status   # 查看状态与 URL
.\scripts\dev.ps1 stop     # 停止
```

**手动：**

```powershell
# 终端 1：CLI（交互式 TUI）
vibe-trading
# 或单条研究任务
vibe-trading run -p "Backtest 600519.SH 2024 年 20/50 均线策略"

# 终端 2：API/Web（若已 npm run build，FastAPI 会直接托管 frontend/dist）
vibe-trading serve --port 8899
# 开发模式前端（改前端源码时）
cd frontend; npm install; npm run dev
```

### 4.3 因子优雅降级（本机无 py-alpha-lib）

本地 Windows 基础安装不含 `py-alpha-lib`（它只发布 `cp314-abi3` Windows wheel，
`pyproject.toml` 的 `factor_runtime` extra 因此是可选、懒加载）。效果：

- **Alpha Zoo 462 预设因子照常**：走现有 `src/factors/registry.py` 的 pandas 路径，
  不受影响。
- **「新因子注册/计算/评估」入口**会抛出 `FactorRuntimeUnavailableError`，错误信息
  直接提示「运行 Docker 因子服务并重试」。可用
  `python -c "import src.factor_runtime as fr; print(fr.runtime_status())"` 观察
  状态（`available=false`）。

结论：需要新因子开发/计算时走 Docker 路径（§3），只做预设因子/回测研究时本地路径
即可。

## 5. QMT Bridge 集成（Windows 宿主机进程隔离）

见 `qmt_bridge/README.md`（D-01 交付）。核心命令：

```powershell
# 启动只读 HTTP 服务（默认 127.0.0.1:8100，loopback + token）
python -m qmt_bridge serve

# 按需落盘单只标的日线（默认 qfq，写共享缓存 cache/loaders/miniqmt/）
python -m qmt_bridge cache --symbol 600519.SH --start 2024-01-01 --end 2024-06-30

# 只读能力 manifest（write_capabilities: false）
python -m qmt_bridge manifest

# token 管理（存 DPAPI 加密 vault，无明文落盘）
python -m qmt_bridge token generate
python -m qmt_bridge token show
```

### 5.1 与两条路径的连接

| 消费方 | 连接方式 | 需要 |
|---|---|---|
| 本地 Windows agent | 读共享缓存 + loopback 冷读 `http://127.0.0.1:8100` | 缓存启用 `VIBE_TRADING_DATA_CACHE=1`；冷读默认即可 |
| Docker agent | 读共享缓存（bind mount，主路径） | `docker-compose.yml` 已内置挂载与 `VIBE_TRADING_DATA_CACHE=1` |
| Docker agent（冷读，可选） | 经 `host.docker.internal:8100` | 桥需改绑 `QMT_BRIDGE_HOST=0.0.0.0`（或局域网 IP）+ 设 `QMT_BRIDGE_TOKEN` |

> **进程隔离与安全**：桥默认只监听 `127.0.0.1`、非 GET/HEAD/OPTIONS 一律 405、
> 凭据走 Windows DPAPI 加密（`qmt_bridge/credentials.py`）。让容器冷读回源属于可选
> 放宽（桥暴露到宿主网关），默认不开启——共享缓存已是主路径，冷读只是缓存未命中的
> 兜底。三级降级语义：`miniqmt → 免费源（baostock 日线主链 / mootdx、eastmoney 分钟
> 兜底）→ local`。

### 5.2 桥缓存与 loader 的一致性

桥与 agent 用同一套 key（`backtest/loaders/base.py` 的 `make_loader_cache_key`，
`source="miniqmt"`、`fields=None`、仅「已结算日」可缓存）与同一
`VIBE_TRADING_DATA_CACHE` / `VIBE_TRADING_DATA_CACHE_ROOT` 开关，所以三方（桥、本地
agent、容器）读写的目录与文件必须一致。改缓存根目录时三处同名环境变量要同步。

## 6. 端口与配置（.env 驱动，不硬编码）

### 6.1 端口

| 变量（宿主 shell / 顶层 `.env`） | 默认 | 作用 |
|---|---|---|
| `VIBE_WEB_PORT` | 8899 | Docker 宿主机对外绑定端口（容器内固定 8899） |
| `VIBE_FRONTEND_PORT` | 5899 | 前端 dev server 宿主机对外端口 |

本地路径端口由 `scripts/dev.ps1` 的 `VIBE_BACKEND_PORT` / `VIBE_FRONTEND_PORT`
（默认 8899 / 5899）驱动。

### 6.2 数据缓存 / 桥连接

| 变量 | 默认 | 说明 |
|---|---|---|
| `VIBE_TRADING_DATA_CACHE` | `1`（容器内） | 开 loader 缓存；本地按需 `1` |
| `VIBE_TRADING_DATA_CACHE_ROOT` | `%USERPROFILE%\.vibe-trading\cache\loaders` | 三方共享的缓存根 |
| `QMT_BRIDGE_HOST` / `QMT_BRIDGE_PORT` | `127.0.0.1` / `8100`（本地）；`host.docker.internal` / `8100`（容器） | 桥地址 |
| `QMT_BRIDGE_TOKEN` | 空 | loopback bearer token（可选；桥默认自生成并存入 DPAPI vault） |

其余 agent 配置见 `agent/.env.example`。

## 7. 凭据分层（LLM / 数据源 / 券商）

| 层 | 存放 | 机制 |
|---|---|---|
| **LLM** | `agent/.env`（本机；Docker 由 `env_file` 注入） | `LANGCHAIN_PROVIDER` + 对应 `*_API_KEY`，明文存于用户自己的 `agent/.env`（不进镜像，`.dockerignore` 已排除）。 |
| **数据源（QMT/miniQMT）** | `~/.vibe-trading/qmt-bridge/secrets.v1.json` | Windows DPAPI 加密（`CryptProtectData`，与桌面端 `safeStorage` 同机制），只存密文 base64；字段 `api_token`、`qmt_account_id`。日志不落明文。 |
| **数据源（免费源）** | `agent/.env` | `TUSHARE_TOKEN`（可选，免费链默认无需）；其余免费源（baostock/mootdx/eastmoney 等）无需凭据。 |
| **券商** | 无实盘下单路径 | 本项目为「研究 + 模拟」边界：不接真实下单通道，QMT Bridge manifest `write_capabilities=false`、无 `xttrader`。券商只读连接器暂缓（DORA-130 Q5），Shadow Account 以同花顺/东财/富途对账单 CSV 覆盖；上游 13 个连接器配置存 `~/.vibe-trading`（不落仓库）。可选 TAP 模式（`agent/.env.example`）将 Alpaca 订单密钥隔离到 TAP 代理并需人工审批。 |

> 原则：**凭据绝不进仓库**（`agent/.env`、`secrets.v1.json` 均被 `.gitignore` /
> `.dockerignore` 排除）、**日志不落明文**、**桥结构性无写端点**。

## 8. 回滚与清理

- **Docker**：回滚镜像 `docker compose down && docker compose build --no-cache` 前，
  用 `docker compose down` 停容器（不加 `-v` 保留数据卷）。彻底清理（含数据）：
  `docker compose down -v`。
- **本地**：`.\scripts\dev.ps1 stop` 停后台服务；`Remove-Item -Recurse -Force .venv`
  后重装即可回滚到纯源码状态。
- **缓存**：随时 `Remove-Item -Recurse -Force ~\.vibe-trading\cache` 清 loader 缓存
  （下次取数重新回源，不影响其余状态）。
- **配置回滚**：compose 与 `.env.example` 均为声明式、幂等，重跑 `docker compose up`
  即恢复到当前配置；端口/路径全部走变量，无硬编码，改回默认值即可复现原始行为。
