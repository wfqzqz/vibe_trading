# 上游同步策略（fork 不僵死）

> 本文件是 vibe_trading（fork 自 [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading)）
> 的**上游同步策略落地文档**，对应架构设计 DORA-124 §六「上游同步策略（P0，避免 fork 僵死）」。
> 落地负责人：DevOps 工程师（主）+ 技术架构师（冲突评审）。
>
> 原则：**加法式改造，不重写上游**。我们的增量全部落在新增文件/新模块
> （`miniqmt_loader.py`、`factor_runtime/`、`qmt_bridge/`、`backtest/optimizers/kelly*.py` 等）
> 与配置（registry 回退链、settings tier），对上游核心做「配置/适配/验证」而非深改，使
> merge 冲突面最小化。

## 1. fork 点固定（同步基准）

| 项 | 值 |
| --- | --- |
| 上游仓库 | `https://github.com/HKUDS/Vibe-Trading.git` |
| fork 仓库 | `https://github.com/wfqzqz/vibe_trading.git` |
| fork 基线 tag | `v0.1.14` |
| fork 基线 commit | `67e562a2f85080bcd5416f751061acba1885e57c` |
| fork 基线提交信息 | `docs(readme): mirror #1180's Docker data-survival sentence in README_es`（2026-08-23 01:27 +0800） |

同步以「上游 tag/release + 迁移说明」为单位，**不逐 commit 盲追**。每次同步的起点是上一个
已合流的同步点（当前为 `67e562a2`），终点是上游的最新 tag 或 `upstream/main` 的某个经评审
确认的 commit。

## 2. 远程跟踪

每个 clone 只需执行一次（幂等），由 `scripts/upstream-sync setup` 完成：

```bash
git remote add upstream https://github.com/HKUDS/Vibe-Trading.git || true
git fetch upstream '+refs/heads/*:refs/remotes/upstream/*' '+refs/tags/*:refs/tags/upstream/*'
git branch -f upstream/main upstream/main   # 本地 main 保留改造，upstream/main 专用于跟踪
```

约定：

- `origin` → 我们的 fork（`wfqzqz/vibe_trading`），`origin/main` 承载全部改造。
- `upstream` → 上游（`HKUDS/Vibe-Trading`），`upstream/main` 仅用于跟踪，**永不直接推送**。
- 上游 tag 命名空间化到 `upstream/` 前缀（`refs/tags/upstream/*`），避免与本地 tag 冲突。

## 3. 同步流程

每次同步走独立 `sync/upstream-YYYYMM` 分支 → 冲突评审（架构师审核）→ 合流 `main`。
节奏：**按月或随上游 release 触发**，不逐 commit 盲追。

```text
upstream/main ──(fetch)──▶ sync/upstream-YYYYMM ──(冲突评审, 架构师)──▶ main
                              ▲
                              └─ 由 origin/main 切出，merge upstream/main
```

操作步骤（`scripts/upstream-sync sync 202608` 自动化，手动作等价）：

```bash
# 1. 确保跟踪最新
scripts/upstream-sync setup

# 2. 预评估：commits 落后数 / 改动文件 / 双方同时改动的冲突面
scripts/upstream-sync check

# 3. 切出同步分支并合流
git switch -c sync/upstream-202608 origin/main
git merge upstream/main          # 或 --no-ff 保留同步边界
#   有冲突：逐文件评审解决；解决不了的升架构师；解决后 git add && git commit

# 4. 锁文件重生成（上游改动过 requirements 时；见 §5）
scripts/upstream-sync regen-locks

# 5. 验证（见 §7）
# 6. push 后发起评审（架构师审核冲突）→ 评审通过 → 合流 main
```

冲突评审要点（架构师负责）：冲突文件是否涉及我们的核心改造（`miniqmt` loader、因子运行时、
QMT Bridge、Kelly 优化器、registry 链），上游改动是否改变了我们依赖的契约（如
`market_data_tool.py` 的 source allow-list、loader registry 语义）。跨模块契约变更以
DORA-124 §四 为准，变更须回写 DORA-124。

## 4. 迁移说明（每次同步必附）

每次同步以一次 commit 合流 `main`，其 message 采用统一格式，并在 PR 描述中附迁移说明：

```text
sync(upstream): merge HKUDS/Vibe-Trading <tag-or-commit> (upstream-YYYYMM)

上游区间: v0.1.14 (67e562a2) .. <本次合入的上游 commit>
上游提交: N 个 (git rev-list --count <base>..upstream/main)

迁移说明:
- 新增/变更功能: <上游 release note 或 commit 摘要>
- 冲突面: <双方同时改动的文件清单，无则写「无文本冲突」>
- 契约影响: <loader registry / market_data_tool / 因子 / 通道 等受影响项，无则写「无」>
- 锁文件: <重生成 or 无变化>
- 验证: <见 §7，给出命令与结果>
- 回滚: <见 §9>
```

## 5. 锁文件管理

三个锁文件均由 `uv pip compile` 生成，源文件是 `agent/requirements*.txt`：

| 锁文件 | 生成命令 |
| --- | --- |
| `requirements-lock.txt` | `uv pip compile --universal --python-version 3.11 --generate-hashes --output-file requirements-lock.txt agent/requirements.txt` |
| `requirements-channels-lock.txt` | `uv pip compile --universal --python-version 3.11 --generate-hashes --constraint requirements-lock.txt --output-file requirements-channels-lock.txt agent/requirements-channels.txt` |
| `requirements-factor-lock.txt` | `uv pip compile --universal --python-version 3.11 --generate-hashes --constraint requirements-lock.txt --output-file requirements-factor-lock.txt agent/requirements-factor.txt` |

规则：

- **同步后重生成**：上游改动了 `agent/requirements*.txt`（或 `pyproject.toml` 依赖）时，
  必须重跑上面三条命令；`requirements-channels-lock.txt` / `requirements-factor-lock.txt`
  以 `requirements-lock.txt` 为 `--constraint`，**先重生成 base 锁，再重生成其余两个**。
- `py-alpha-lib==0.3.0` 与 `numpy>=2` 的兼容声明单列于
  `agent/src/factor_runtime/COMPATIBILITY.md`；任何 `py-alpha-lib` / `numpy` 升级都必须重跑
  因子对账回归（`agent/tests/test_factor_reconciliation.py`，`pytest.mark.integration`）。
- 完整性自检（CI `hash-lock` job 已固化）：两个锁分别独立
  `pip install --dry-run --ignore-installed --require-hashes -r <lock>` 必须通过。

## 6. CI drift 检查（周度）

`.github/workflows/upstream-drift.yml` 每周（周一 03:00 UTC）+ 手动触发，对比
`upstream/main` 与 `main` 的 diff 面：

- 度量：`commits_behind`（落后上游的 commit 数）、上游改动文件数、**冲突面**
  （fork 点以来双方同时改动的文件交集）、`git merge-tree` 实际冲突。
- **冲突预评估**：每次运行都产出冲突面清单（双方同时改动的文件），写入 job summary。
- **超阈值告警**：`commits_behind >= 100` 或 `git merge-tree` 实际冲突文件数 `>= 1` 时，用
  `gh issue create` 开告警 issue（label `upstream-drift`），附完整冲突预评估，防止长期不合并导致僵死。
- 阈值可在 `workflow_dispatch` 手动触发时覆盖。

## 7. 验证口径（首次同步无回归）

同步合流前必须给出验证证据：

1. `git merge-tree --write-tree origin/main upstream/main` 无文本冲突（或冲突已解决）。
2. 锁文件与 `agent/requirements*.txt` 一致：`git status` 无脏锁文件；若重生成，跑 CI
   `hash-lock` 自检通过。
3. 受契约影响的路径做针对性回归（如 `market_data_tool.py` 的 source allow-list 从 loader
   registry 派生 → 跑 `agent/tests/test_registry.py`、`agent/tests/test_market_data_tool.py`）。
4. 全量回归由 Stage 6 V-02 承接（CI `test` job：`pytest` + `frontend` 构建）。

## 8. 首次同步记录（upstream-202608）

| 项 | 值 |
| --- | --- |
| 上游区间 | `v0.1.14 (67e562a2)` → `upstream/main @ 99e84aba`（`chore(mcp): drop the fetch_market_data_json import #1185 orphaned`） |
| 上游提交数 | 23（`git rev-list --count 67e562a2..upstream/main`） |
| 冲突面 | 双方同时改动仅 `README.md`（`git merge-tree --write-tree` 无文本冲突） |
| 锁文件 | 上游未改 `agent/requirements*.txt` / `pyproject.toml` → 无变化（重生成步骤为 no-op） |
| 验证 | 见本次同步的 issue 评论与 `sync/upstream-202608` 分支 |

## 9. 回滚

- **合流前**：直接删除 `sync/upstream-YYYYMM` 分支即可，`main` 不受影响。
  `git branch -D sync/upstream-YYYYMM && git push origin --delete sync/upstream-YYYYMM`。
- **合流后**：若 `main` 需回退，`git revert -m 1 <merge-commit>`（保留历史），或
  `git reset --hard <合流前的 main commit>` 后强制推送（需确认无他人提交，慎用）。
  回退后锁文件与 `requirements-lock.txt` 一并回到同步前状态。
