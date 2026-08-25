# py-alpha-lib factor runtime — dependency compatibility assessment (F-01)

> **Owner / purpose.** DORA-144 (F-01) deliverable; the durable record that
> DORA-124 Stage 4 task **F-04** ("因子对账回归 + numpy≥2 兼容评估") reconciles
> against. Everything below is traceable to pinned files: `requirements-lock.txt`,
> `requirements-factor-lock.txt`, `pyproject.toml`, `Dockerfile`.

## 1. Conclusion (TL;DR)

| Question | Answer |
| --- | --- |
| Is `py-alpha-lib==0.3.0` compatible with `numpy>=2`? | **Yes — it requires it.** The wheel declares `Requires-Dist: numpy>=2` (Python `>=3.11`). |
| Compatible with the upstream stack `pandas>=2,<3` / `bottleneck` / `scikit-learn`? | **Yes.** The base lock already pins `numpy==2.4.6`, `pandas==2.3.3`, `bottleneck==1.6.0`, `scikit-learn==1.9.0`, `scipy==1.17.1` — all numpy-2-native releases — and `py-alpha-lib` adds no new transitive constraints beyond `numpy>=2`. |
| Any version conflict? | **None.** `py-alpha-lib`'s only dependency is `numpy>=2`, satisfied by the already-locked `numpy==2.4.6` (no second numpy, no pin change). |

`py-alpha-lib` is a Rust/PyO3 extension built against the **abi3** stable ABI, so its
native code does not re-enter numpy's C API at build time — the only runtime
requirement is a numpy 2.x ABI at the Python level, which the base stack already
ships.

## 2. Wheel availability (why "container-only + graceful degradation")

Verified against PyPI (`py-alpha-lib` 0.3.0 file list, 2026-08-24):

| Platform | Wheel tag | Importable on |
| --- | --- | --- |
| Linux x86_64 | `cp311-abi3-manylinux_2_17_x86_64` | Python ≥ 3.12 (the Docker image is `python:3.14-slim` — 0.3.0's Python source uses PEP 695, DORA-251) |
| Linux musl | `cp311-abi3-musllinux_1_2_x86_64` | Python ≥ 3.11 |
| macOS arm64 | `cp311-abi3-macosx_11_0_arm64` | Python ≥ 3.11 |
| **Windows x86_64** | **`cp314-abi3-win_amd64` only** | **Python ≥ 3.14 only** |
| free-threaded | `cp314-cp314t` / `cp315-cp315t` | CPython 3.14t / 3.15t |

**Consequence (the F-01 degradation contract).** The project's Windows host runs
Python 3.11/3.12, for which py-alpha-lib publishes **no** Windows wheel (and no
sdist install is expected — it needs the Rust toolchain). Therefore:

- **Docker path (recommended for factor work):** the agent image installs
  `requirements-factor-lock.txt` and `import alpha` succeeds (`cp311-abi3`).
- **Local Windows path:** py-alpha-lib is *not* installed (it is an optional
  `factor_runtime` extra, not a base dependency), so the base install stays lean.
  The Alpha Zoo's ~460 preset factors are unaffected (they run on the pandas path
  in `src.factors.registry.Registry.compute`); only the new-factor entry points
  (`src.factor_runtime.runtime.FactorRuntime.register/compute/evaluate`) raise
  `FactorRuntimeUnavailableError` with an actionable Docker hint.

## 3. Empirical verification (isolated venv)

Command (Python 3.14, the only local interpreter with a Windows py-alpha-lib wheel):

```text
python -m venv compat_venv
compat_venv/Scripts/python -m pip install \
    numpy==2.4.6 pandas==2.3.3 bottleneck==1.6.0 scikit-learn==1.9.0 py-alpha-lib==0.3.0
compat_venv/Scripts/python compat_smoke.py
```

`compat_smoke.py` imports `numpy`/`pandas`/`bottleneck`/`sklearn`/`alpha` in the
agent's real import order and checks:

1. all five packages co-import with the pinned versions;
2. `alpha.MA` returns the documented rolling-window values (both default
   partial-warmup mode and `FLAG_STRICTLY_CYCLE` NaN-warmup mode), matching a
   pandas/`rolling` reference.

Result (isolated venv, Python 3.14.6 — the local interpreter with a Windows
py-alpha-lib wheel):

```text
[ok] import numpy 2.4.6
[ok] import pandas 2.3.3
[ok] import bottleneck 1.6.0
[ok] import scikit-learn 1.9.0
[ok] py-alpha-lib dist version 0.3.0
[ok] alpha.MA shape (6,)
[ok] alpha.MA warmup values [1.  1.5]
[ok] alpha.MA steady-state values [2. 3. 4. 5.]
[ok] alpha.MA strict-cycle (NaN warmup) [nan nan  2.]
----
numpy=2.4.6 pandas=2.3.3 bottleneck=1.6.0 sklearn=1.9.0 py-alpha-lib=0.3.0
FAILURES: 0
```

All five packages co-import at the pinned versions and `alpha.MA` matches the
documented rolling-window semantics, so the `numpy>=2` conclusion is confirmed
empirically, not just by metadata.

## 4. Risk notes (handed to F-04)

- **Re-verify on upgrade.** `requirements-factor-lock.txt` pins
  `py-alpha-lib==0.3.0`; any bump must re-run the factor reconciliation
  regression (F-04) against same-口径 Alpha Zoo factors before landing.
- **Windows remains degraded by design** until py-alpha-lib ships a
  `cp311-abi3`/`cp312-abi3` Windows wheel; do not "fix" this by building from
  source in the base install (Rust toolchain + long builds) — the Docker path is
  the supported route.
- **numpy 2.x is the floor**, not a ceiling: `py-alpha-lib` requires `numpy>=2`,
  so any future numpy 1.x downgrade of the base stack would break the runtime.
  Keep the base lock at `numpy>=2`.

## 5. Traceable findings for F-04 reconciliation (DORA-188 closure)

DORA-188 (F-03 review P1: "按 ExecContext 实际算子面收口或上报上游") pins three
runtime findings here so F-04 (因子对账回归, DORA-147) reconciles the same
口径 (DORA-124 §3.4 "same-口径 comparison" contract) instead of rediscovering
them:

### 5.1 REF/HHV/LLV/HHVBARS/LLVBARS alias gap (doc-claimed, not in 0.3.0)

py-alpha-lib's documentation advertises `REF` / `HHV` / `LLV` / `HHVBARS` /
`LLVBARS` aliases; the 0.3.0 `alpha.context.ExecContext` does **not** implement
them, so a translated `ctx.REF(...)` call only failed at compute time with a
raw `AttributeError` (previously a generic 500). Closure behavior:

| Surface | Before (F-03) | After (DORA-188) |
| --- | --- | --- |
| `POST /alpha/custom` (register) | translated OK → error only at compute | **400** `unsupported factor operator: ...` (`FactorOperatorError`, from `validate_operator_surface`) |
| `compute` / `evaluate` | `AttributeError` → **500** generic | **422** `FactorComputeError` ("operator ... not supported") |
| CLI `alpha custom compute|evaluate` | raw traceback | same readable one-line error via `_handle_exception` |

Evidence: `src/factor_runtime/operators.py` (ExecContext surface introspection),
`src/factor_runtime/translator.py` (registration-time check),
`src/factor_runtime/compute.py` (`AttributeError` → `FactorComputeError`),
`src/api/alpha_routes.py` (400/422 mapping),
`agent/tests/test_factor_snapshot.py` / `agent/tests/test_factor_compute.py`
(error-path tests: register 400 / compute+evaluate 422 / CLI readable error).

### 5.2 ExecContext flat-array order (code-major, date-ascending)

`alpha.context.ExecContext` consumes OHLCV as a flat `np.ndarray` of length
`n_codes * n_dates`, ordered **code-major / date-ascending** (row `i` is
`code_idx * n_dates + date_idx`). The adapter (`panel_to_long` /
`reshape_factor_result` in `src/factor_runtime/compute.py`) emits rows in
exactly that order and round-trips a dense wide panel via
`reshape(n_codes, n_dates).T`. F-04 must compare same-layout (a long-format
zoo trace must be re-flattened code-major before it can be diffed against a
py-alpha-lib result).

Evidence: `src/factor_runtime/compute.py` (docstring + implementation) and
`agent/tests/test_factor_compute.py::test_panel_to_long_maps_columns_and_orders_rows`.

### 5.3 MA rolling warmup 口径 difference (partial-warmup vs `min_periods=n`)

`alpha.MA` (py-alpha-lib 0.3.0) defaults to **partial warmup**: during the
first `n-1` bars it returns the mean over the observations available so far
(window=2 smoke run → warmup `[1., 1.5]`, §3 above). The zoo pandas path
(`ts_mean` and the inline `rolling(window=n, min_periods=n).mean()` used
across the zoo, e.g. `src/factors/base.py`, qlib158/gtja191/alpha101 factor
files) yields **NaN warmup** (all-NaN until `n` valid bars). py-alpha-lib's
`FLAG_STRICTLY_CYCLE` reproduces the NaN-warmup semantics (smoke: `[nan nan
2.]`, §3 above), but the **default is partial-warmup and differs from the zoo
path**. F-04 must either force the strict-cycle flag or restrict the
reconciliation to the steady-state region — do not diff the two paths' warmup
values directly.

Evidence: §3 empirical smoke output above, `src/factors/base.py::ts_mean`,
zoo factor files (`min_periods=n` rolling calls).

## 6. F-04 closure — upgrade re-run + conflict list + lock hash (DORA-147)

> Recorded by DORA-147 (F-04, 测试工程师). Everything below was re-verified on
> the **exact pinned stack** (`numpy==2.4.6 pandas==2.3.3 bottleneck==1.6.0
> scikit-learn==1.9.0 scipy==1.17.1 py-alpha-lib==0.3.0`) in an isolated venv,
> plus a real `py-alpha-lib==0.3.0` reconciliation run.

### 6.1 `numpy>=2` conflict list — **closed (empty)**

`pip` resolves the whole stack with no conflicts, and `pip check` reports
**"No broken requirements found"**. Per-dependency status:

| Upstream dep | Pinned | numpy>=2 status |
| --- | --- | --- |
| `pandas` | 2.3.3 | compatible (numpy-2-native release) |
| `bottleneck` | 1.6.0 | compatible (numpy-2-native release) |
| `scikit-learn` | 1.9.0 | compatible (numpy-2-native release) |
| `scipy` | 1.17.1 | compatible (numpy-2-native release) |
| `py-alpha-lib` | 0.3.0 | **requires** `numpy>=2` |

Co-import + numeric smoke (11/11 checks, `FAILURES: 0`): all five packages
co-import at the pinned versions; `alpha.MA` == pandas rolling mean,
`bottleneck.move_max` == pandas rolling max, `sklearn.LinearRegression` and
`scipy.stats.pearsonr` both run on numpy 2.4.6. **No version conflict, no
second numpy, no pin change.**

### 6.2 Reconciliation regression (upgrade re-run target)

New test `agent/tests/test_factor_reconciliation.py` (9 tests, `pytest.mark.integration`,
auto-skipped on a degraded host via `pytest.importorskip("alpha")`) reconciles the
py-alpha-lib path against the Alpha Zoo's **own** `src.factors.base` operators on a
shared panel. Re-run it after any `py-alpha-lib` / `numpy>=2` / factor-dependency bump.

Empirical warmup finding (refines the F-01 note): py-alpha-lib's rolling family is
**not** uniformly strict-warmup —

| Operator family | py-alpha-lib warmup | zoo equivalent | reconcile |
| --- | --- | --- | --- |
| `stddev`/`std`/`ts_std_dev` | strict (`min_periods=n`) | `ts_std` | exact |
| `corr` / `cov` | strict (`min_periods=n`) | `ts_corr` / `ts_cov` | exact |
| `ma`/`sma`/`mean`/`ts_mean` | **partial** (`min_periods=1`) | `ts_mean` (strict) | equal after warmup |
| `sum` / `ts_max` / `ts_min` | **partial** (`min_periods=1`) | (n/a) / `ts_max` / `ts_min` (strict) | equal after warmup |

The partial-vs-strict divergence is confined to the first `n-1` bars; from bar
`n` onward both paths agree, and the shared IC/IR + layered math
(`factor_analysis_core`) is unaffected — confirming the F-03 handoff note and
extending it from `MA` to the whole mean/max/min/sum rolling family.

### 6.3 Lock-file hash verification

- `requirements-factor-lock.txt`: `numpy==2.4.6` and `py-alpha-lib==0.3.0`
  hashes verified against the actual `cp314` Windows wheels (SHA256 match);
  `pip install --dry-run --require-hashes -r requirements-factor-lock.txt`
  accepted (no missing/mismatched hash).
- `requirements-lock.txt`: `bottleneck==1.6.0`, `pandas==2.3.3`,
  `scikit-learn==1.9.0`, `scipy==1.17.1` hashes verified against the actual
  wheels (SHA256 match).
- Any upgrade to `py-alpha-lib` / `numpy` **must** re-generate the lock
  (`uv pip compile --generate-hashes …`) and re-run §6.2's reconciliation.
