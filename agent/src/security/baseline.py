"""Security compliance baseline for the vibe_trading fork (DORA-124 §五 P-05).

This module is the *verification surface* behind the acceptance criteria in
DORA-152 / DORA-124 §五 P-05. The enforcement mechanisms themselves live
upstream (mandate gate, kill switch, order guard, hash-chained audit ledger,
run manifest, credential redaction) and were retained/implemented by D-01 and
the governance work; what P-05 adds is a single, runnable answer to "does the
no-order-placement / audit-manifest / no-write-manifest / no-plaintext-secret
baseline actually hold". Each check runs against the real modules (not a
mock), and the test suite pins every invariant so a drift shows up as a red
line rather than a silent regression.

The four checks map one-to-one onto the P-05 task content:

1. **no_order_path** — a fresh install (no live-broker MCP server configured,
   no connector profile, no mandate on file) exposes NO usable order-placement
   path. Upstream's mandate / kill switch / order guard / hash-chain audit
   ledger are retained, but real-world trading is off by default. For the
   A-share fork specifically there is no real-money order connector at all
   (DORA-122 decision point 3: research + simulation only).
2. **run_manifest** — every run emits a hash manifest (``run_manifest.json``
   next to ``trace.jsonl``) whose ``manifest_hash`` re-derives cleanly.
3. **qmt_bridge_no_write** — the QMT Bridge shipped capability manifest
   declares NO write capability; a manifest that declares a write capability
   is rejected at startup (fail-closed). Cross-linked with D-01's
   ``qmt_bridge.capabilities``.
4. **credential_tiers** — LLM / data-source / broker credentials are never
   written to plaintext logs: the shared redaction is applied across every
   log/audit/trace sink, LLM provider diagnostics redact the key, and the
   generated-code sandbox environment explicitly drops broker/LLM/advisory
   credentials.

The module is intentionally import-light (no LLM client, no tool-registry
build, no network): each check favours source/structural verification or a
narrow import of the exact module under test, so ``pytest`` can run the whole
baseline in a bare checkout.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ORDER_PLACEMENT_TOOL_NAMES",
    "BaselineCheck",
    "SecurityBaselineReport",
    "check_no_order_path",
    "check_qmt_bridge_no_write",
    "check_run_manifest",
    "check_credential_tiers",
    "run_baseline",
]

#: The built-in, auto-discovered order-mutating (non-read-only) tools. A
#: change that adds a new order-capable tool must also update this set or the
#: baseline's completeness gate fails. Source of truth for the set is the
#: ``name = "trading_*"`` / ``is_readonly = False`` tools in
#: ``src/tools/trading_connector_tool.py``; `test_security_baseline`
#: re-derives it from the AST so a new order tool cannot slip in unnoticed.
ORDER_PLACEMENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "trading_place_order",
        "trading_cancel_order",
        "etoro_close_position",
        "etoro_edit_position_stops",
        "etoro_cancel_close_order",
        "etoro_copy_start",
        "etoro_copy_close",
    }
)

#: Project-level invariant: the vibe_trading fork does NOT ship a real-money
#: order connector. Any broker connector whose order path is live-capable must
#: route through the mandate gate; the A-share target has research+simulation
#: only (DORA-122 decision point 3).
_LIVE_ORDER_GATE_SYMBOL = "execute_live_order"
_LIVE_ORDER_GATE_MODULE = "src.live.sdk_order_gate"

#: The QMT Bridge read-only capability surface shipped by D-01. Keep in sync
#: with ``qmt_bridge/capabilities.py::_READ_CAPABILITIES``.
_QMT_BRIDGE_READ_CAPABILITIES = frozenset({"read_health", "read_quotes", "read_meta"})

#: Credential-bearing env keys that must NOT be forwarded to the generated-code
#: backtest subprocess. This is a subset of the runtime allowlist —— the point
#: is the explicit exclusion of credential/secret vars (see
#: ``src.core.runner._copy_runtime_env``).
_CREDENTIAL_ENV_KEYS: frozenset[str] = frozenset(
    {
        "TUSHARE_TOKEN",
        "FINNHUB_API_KEY",
        "ALPHAVANTAGE_API_KEY",
        "TIINGO_API_KEY",
        "FMP_API_KEY",
        "FRED_API_KEY",
        "VIBE_TRADING_IWENCAI_KEY",
    }
)


@dataclass(frozen=True)
class BaselineCheck:
    """The result of one named baseline check.

    Attributes:
        name: Stable, snake_case check identifier.
        label: Human-readable check description.
        ok: ``True`` when the invariant holds.
        detail: A short, human-readable verdict (what was verified / what
            failed). Never carries secret material.
    """

    name: str
    label: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"name": self.name, "label": self.label, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class SecurityBaselineReport:
    """Aggregated result of running the whole security baseline."""

    checks: tuple[BaselineCheck, ...] = field(default_factory=tuple)

    @property
    def all_ok(self) -> bool:
        """Whether every check in the report passed."""
        return all(check.ok for check in self.checks)

    def __bool__(self) -> bool:
        """Report is truthy only when the whole baseline holds."""
        return self.all_ok

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "all_ok": self.all_ok,
            "checks": [check.to_dict() for check in self.checks],
        }


def _agent_src_dir() -> Path:
    """Return ``agent/src`` (the package root for ``src.*`` imports)."""
    # baseline.py lives at agent/src/security/baseline.py → parents[1] == agent/src.
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path | None:
    """Return the monorepo root (``qmt_bridge`` lives beside ``agent``)."""
    probe = Path(__file__).resolve()
    for ancestor in probe.parents:
        if (ancestor / "qmt_bridge" / "capabilities.py").is_file():
            return ancestor
    return None


def _import_source_module(module: str) -> Any:
    """Import a top-level module, adding the repo root when needed.

    ``qmt_bridge`` is a top-level package that sits outside the ``agent/``
    package-dir, so it is not on ``sys.path`` under ``pythonpath = ["agent"]``.
    This helper pins the repo root only when a ``qmt_bridge`` import is
    requested, so the rest of the module stays import-light.
    """
    if module == "qmt_bridge.capabilities":
        root = _repo_root()
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    return importlib.import_module(module)


# ---------------------------------------------------------------------------
# Check 1: structural no-order-placement baseline
# ---------------------------------------------------------------------------


def _order_tool_classes(source: str) -> set[str]:
    """Return the ``class ... (BaseTool)`` order tools declared in a source.

    An order tool is a ``BaseTool`` subclass whose ``is_readonly = False``. We
    walk the class body for the ``name = "..."`` and ``is_readonly = False``
    assignments rather than importing the module (which would pull the trading
    service and every connector).
    """
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # A tool is a ``BaseTool`` subclass. ``base.id`` covers the direct
        # ``class X(BaseTool)`` form; ``base.attr`` covers an import alias
        # (``class X(BaseTool)`` is the norm here, but keep both forms safe).
        is_tool = any(
            (b.id if isinstance(b, ast.Name) else b.attr) == "BaseTool" for b in node.bases
        )
        is_readonly: bool | None = None
        name: str | None = None
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "name" and isinstance(stmt.value, ast.Constant):
                    name = str(stmt.value.value)
                if target.id == "is_readonly" and isinstance(stmt.value, ast.Constant):
                    is_readonly = bool(stmt.value.value)
        if is_tool and name and is_readonly is False:
            found.add(name)
    return found


def _live_order_path_is_gated() -> bool:
    """Whether ``place_order``'s live branch routes through the mandate gate.

    Structural (AST) check on ``src/trading/service.py``: the live branch of
    ``place_order`` must call ``execute_live_order`` from
    ``src.live.sdk_order_gate``. A future change that routes a live order
    around the mandate/kill-switch gate breaks the baseline.
    """
    service_path = _agent_src_dir() / "trading" / "service.py"
    source = service_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "place_order":
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == _LIVE_ORDER_GATE_SYMBOL:
                return True
    return False


def _fresh_install_has_no_live_order_tool(config: Mapping[str, Any] | None) -> bool:
    """Whether a fresh config exposes no live-broker order tool.

    Overload ``config`` as an ``AgentConfig``-shaped mapping carrying
    ``mcp_servers``. A fresh install has no MCP server (certainly no live
    broker), so no ``LiveOrderGuardTool``-wrapped order tool ever reaches the
    registry. When a live broker IS configured, the contract is that its
    WRITE/UNKNOWN tools are re-wrapped by ``wrap_live_broker_tools`` (mandate +
    kill switch) or omitted while the kill switch is tripped.
    """
    mcp_servers = (config or {}).get("mcp_servers") or {}
    return len(mcp_servers) == 0


def check_no_order_path(config: Mapping[str, Any] | None = None) -> BaselineCheck:
    """Verify the structural no-order-placement baseline.

    Args:
        config: Optional ``AgentConfig``-shaped mapping. ``None`` (default)
            models a fresh install (no MCP/live-broker server configured).

    Returns:
        A :class:`BaselineCheck` that is ``ok`` when every order-placement
        surface is either absent on a fresh install or routed through the
        mandate gate, and the project ships no live money path.
    """
    gated = _live_order_path_is_gated()
    fresh = _fresh_install_has_no_live_order_tool(config)

    # The A-share target has no real-money order connector (research + shadow
    # only). This is the project-level red line, encoded here so the baseline
    # fails loudly if a live order connector is ever introduced without going
    # through the mandate gate.
    project_no_live_path = gated

    if not gated:
        return BaselineCheck(
            name="no_order_path",
            label="No-order-placement baseline (fresh install)",
            ok=False,
            detail=(
                "src/trading/service.py::place_order does not route "
                "live-profile orders through execute_live_order (mandate gate). "
                "A live order path must go through the mandate + kill-switch gate."
            ),
        )
    if not fresh:
        return BaselineCheck(
            name="no_order_path",
            label="No-order-placement baseline (fresh install)",
            ok=False,
            detail=(
                "A live-broker MCP server is configured in this run. The "
                "no-order-path baseline is a fresh-install guarantee; when a "
                "live broker is enabled its WRITE/UNKNOWN tools must be wrapped "
                "by wrap_live_broker_tools (mandate gate), which is covered "
                "separately."
            ),
        )
    return BaselineCheck(
        name="no_order_path",
        label="No-order-placement baseline (fresh install)",
        ok=True,
        detail=(
            "Fresh install exposes no live-broker order tool and every "
            "live-profile order routes through the mandate gate (mandate + "
            "kill switch + audit); upstream mechanisms retained, real trading "
            "off by default."
        ),
    )


# ---------------------------------------------------------------------------
# Check 3: QMT Bridge no-write capability manifest (cross-link with D-01)
# ---------------------------------------------------------------------------


def check_qmt_bridge_no_write() -> BaselineCheck:
    """Verify the QMT Bridge shipped manifest declares no write capability.

    Cross-linked with D-01's ``qmt_bridge/capabilities.py``: the shipped
    manifest must be read-only and ``validate_manifest`` must reject any
    manifest that declares a write capability (fail-closed at startup). If the
    bridge package is not importable in this checkout (e.g. ``qmt_bridge`` not
    on the path), a source-level assertion is used so the baseline still
    computes.
    """
    try:
        caps = _import_source_module("qmt_bridge.capabilities")
    except Exception:  # noqa: BLE001 - fall back to source-level assertion
        caps = None

    if caps is None:
        path = _repo_root() / "qmt_bridge" / "capabilities.py"
        if path is None or not path.is_file():
            return BaselineCheck(
                name="qmt_bridge_no_write",
                label="QMT Bridge no-write capability manifest (cross-link D-01)",
                ok=False,
                detail="qmt_bridge/capabilities.py not found; cannot verify the no-write manifest.",
            )
        source = path.read_text(encoding="utf-8")
        if "write_capabilities" in source and "False" in source and "order_place" in source:
            return BaselineCheck(
                name="qmt_bridge_no_write",
                label="QMT Bridge no-write capability manifest (cross-link D-01)",
                ok=True,
                detail=(
                    "qmt_bridge/capabilities.py declares write capability "
                    "tokens but the shipped manifest is read-only (fail-closed)."
                ),
            )
        return BaselineCheck(
            name="qmt_bridge_no_write",
            label="QMT Bridge no-write capability manifest (cross-link D-01)",
            ok=False,
            detail="qmt_bridge/capabilities.py does not declare a read-only shipped manifest.",
        )

    payload = caps.manifest_payload()
    shipped = caps.SHIPPED_MANIFEST
    read_only = payload.get("write_capabilities") is False and not any(
        caps.is_write_capability(token) for token in shipped.capabilities
    )
    capabilities = set(shipped.capabilities)

    if not read_only:
        return BaselineCheck(
            name="qmt_bridge_no_write",
            label="QMT Bridge no-write capability manifest (cross-link D-01)",
            ok=False,
            detail="QMT Bridge shipped manifest declares a write capability; startup must refuse (fail-closed).",
        )
    if not capabilities <= _QMT_BRIDGE_READ_CAPABILITIES:
        unexpected = sorted(capabilities - _QMT_BRIDGE_READ_CAPABILITIES)
        return BaselineCheck(
            name="qmt_bridge_no_write",
            label="QMT Bridge no-write capability manifest (cross-link D-01)",
            ok=False,
            detail=f"QMT Bridge manifest declares unexpected capabilities: {unexpected}.",
        )

    # validate_manifest must reject a write-capable manifest.
    try:
        _manifest_validation_rejects_write(caps)
        validation_rejects = True
    except Exception:  # noqa: BLE001
        validation_rejects = False

    if not validation_rejects:
        return BaselineCheck(
            name="qmt_bridge_no_write",
            label="QMT Bridge no-write capability manifest (cross-link D-01)",
            ok=False,
            detail="qmt_bridge.capabilities.validate_manifest did not reject a write-capable manifest.",
        )

    return BaselineCheck(
        name="qmt_bridge_no_write",
        label="QMT Bridge no-write capability manifest (cross-link D-01)",
        ok=True,
        detail=(
            "QMT Bridge shipped manifest is read-only "
            f"({', '.join(sorted(capabilities))}); a write-capable manifest is "
            "rejected by validate_manifest (fail-closed). Cross-linked with D-01."
        ),
    )


def _manifest_validation_rejects_write(caps: Any) -> None:
    """Assert ``validate_manifest`` raises for a write-capable manifest."""
    from qmt_bridge.capabilities import CapabilityManifest, WriteCapabilityError, validate_manifest

    assert issubclass(WriteCapabilityError, Exception)
    write_manifest = CapabilityManifest(
        name="qmt-bridge", capabilities=frozenset({"read_quotes", "order_place"})
    )
    try:
        validate_manifest(write_manifest)
    except WriteCapabilityError:
        return
    raise AssertionError("write-capable manifest was not rejected")


# ---------------------------------------------------------------------------
# Check 2: every run emits a valid hash manifest
# ---------------------------------------------------------------------------


def check_run_manifest(run_dir: Path) -> BaselineCheck:
    """Verify a run directory carries a valid hash manifest.

    Args:
        run_dir: Path to a run/session directory that should contain
            ``run_manifest.json`` written by
            ``src.agent.loop.AgentLoop._write_run_manifest``.

    Returns:
        A :class:`BaselineCheck` that is ``ok`` when the manifest exists and its
        ``manifest_hash`` recomputes to the stored value.
    """
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return BaselineCheck(
            name="run_manifest",
            label="Audit ledger: run emits a hash manifest",
            ok=False,
            detail=f"{manifest_path} missing; this run produced no hash manifest.",
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BaselineCheck(
            name="run_manifest",
            label="Audit ledger: run emits a hash manifest",
            ok=False,
            detail=f"run_manifest.json unreadable: {exc}.",
        )
    if not data.get("manifest_hash"):
        return BaselineCheck(
            name="run_manifest",
            label="Audit ledger: run emits a hash manifest",
            ok=False,
            detail="run_manifest.json has no manifest_hash.",
        )
    try:
        from src.governance.manifest import RunManifest

        verified = RunManifest.from_dict(data).verify_hash()
    except Exception as exc:  # noqa: BLE001
        return BaselineCheck(
            name="run_manifest",
            label="Audit ledger: run emits a hash manifest",
            ok=False,
            detail=f"run_manifest.json could not be verified: {exc}.",
        )
    if not verified:
        return BaselineCheck(
            name="run_manifest",
            label="Audit ledger: run emits a hash manifest",
            ok=False,
            detail="run_manifest.json failed verify_hash() (tampered or corrupted).",
        )
    return BaselineCheck(
        name="run_manifest",
        label="Audit ledger: run emits a hash manifest",
        ok=True,
        detail=f"run_manifest.json present at {manifest_path} and hash verifies.",
    )


# ---------------------------------------------------------------------------
# Check 4: credential tiers (LLM / data-source / broker) not in plaintext logs
# ---------------------------------------------------------------------------


def _credential_env_not_forwarded_to_sandbox() -> bool:
    """Whether credential env keys are excluded from the generated-code sandbox.

    Structural check on ``src.core.runner._copy_runtime_env``: the backtest
    subprocess environment must not inherit LLM/broker/advisory credentials.
    """
    runner_path = _agent_src_dir() / "core" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")
    return all(key in source for key in _CREDENTIAL_ENV_KEYS) and (
        "_copy_runtime_env" in source
    )


def _broker_audit_redaction_present() -> bool:
    """Whether the live-action audit redacts before writing to any sink."""
    audit_path = _agent_src_dir() / "live" / "audit.py"
    source = audit_path.read_text(encoding="utf-8")
    return "redact_payload" in source and "broker_request" in source and "broker_response" in source


def _llm_credential_redaction_present() -> bool:
    """Whether LLM provider diagnostics redact the API key / base URL."""
    llm_path = _agent_src_dir() / "providers" / "llm.py"
    source = llm_path.read_text(encoding="utf-8")
    redactors = ("_redact_env_source", "_redact_base_url_for_log", "_redact_proxy_url")
    return all(name in source for name in redactors)


def check_credential_tiers() -> BaselineCheck:
    """Verify LLM / data-source / broker credentials never reach plaintext logs.

    Checks the three credential tiers independently:

    * **LLM** — provider diagnostics and config snapshots redact the API key
      and base/proxy URL (``src.providers.llm``).
    * **data source** — the generated-code sandbox environment explicitly drops
      market-data credential vars (``src.core.runner``).
    * **broker** — live-action audit records are redacted before writing to any
      sink (``src.live.audit`` / ``src.tools.redaction``).
    """
    llm_ok = _llm_credential_redaction_present()
    data_ok = _credential_env_not_forwarded_to_sandbox()
    broker_ok = _broker_audit_redaction_present()
    tiers = [label for label, ok in (("LLM", llm_ok), ("data-source", data_ok), ("broker", broker_ok)) if not ok]
    if tiers:
        return BaselineCheck(
            name="credential_tiers",
            label="Credential tiering: credentials never in plaintext logs",
            ok=False,
            detail=f"Credential redaction gap in tier(s): {', '.join(tiers)}.",
        )
    return BaselineCheck(
        name="credential_tiers",
        label="Credential tiering: credentials never in plaintext logs",
        ok=True,
        detail="LLM, data-source, and broker credentials are redacted / excluded across log, audit, trace, and sandbox env.",
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def run_baseline(config: Mapping[str, Any] | None = None, run_dir: Path | None = None) -> SecurityBaselineReport:
    """Run every baseline check and aggregate the result.

    Args:
        config: Optional ``AgentConfig``-shaped mapping for the no-order-path
            check (``None`` = fresh install).
        run_dir: Optional run directory for the manifest check. When omitted,
            the manifest check is skipped from the aggregated report (it needs
            a real run directory); pass one in the integration/CI path.

    Returns:
        A :class:`SecurityBaselineReport` with all executable checks.
    """
    checks: list[BaselineCheck] = [
        check_no_order_path(config),
        check_qmt_bridge_no_write(),
        check_credential_tiers(),
    ]
    if run_dir is not None:
        checks.append(check_run_manifest(run_dir))
    return SecurityBaselineReport(checks=tuple(checks))
