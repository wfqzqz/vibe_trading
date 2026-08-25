"""Security compliance baseline tests for DORA-124 §五 P-05.

These pin the four P-05 acceptance items as runnable, drift-sensitive checks:

1. **No-order-path baseline** — a fresh install exposes no order-placement
   path; every live-profile order routes through the mandate gate.
2. **Run hash manifest** — every run emits a ``run_manifest.json`` whose
   ``manifest_hash`` re-derives cleanly (tampering is caught).
3. **QMT Bridge no-write manifest (cross-link D-01)** — the shipped manifest is
   read-only and a write-capable manifest is rejected (fail-closed).
4. **Credential tiering** — LLM / data-source / broker credentials never reach
   plaintext logs.

The checks verify against the real modules (the mandate gate, the shipped
bridge manifest, the governance manifest, the redaction helpers) rather than
mocks, so a regression anywhere in those layers fails this suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.manifest import build_run_manifest
from src.security import baseline as baseline_mod
from src.security.baseline import (
    ORDER_PLACEMENT_TOOL_NAMES,
    SecurityBaselineReport,
    check_credential_tiers,
    check_no_order_path,
    check_qmt_bridge_no_write,
    check_run_manifest,
    run_baseline,
)

#: ``trading_select_connection`` is a non-read-only tool but not an order
#: placement tool (it only persists the selected profile). The completeness
#: gate treats it as the sole non-order write tool.
_NON_ORDER_WRITE_TOOLS = frozenset({"trading_select_connection"})


def _tool_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "tools" / "trading_connector_tool.py"


def _write_manifest(dirpath: Path, *, system_prompt: str = "sys", tool_names: tuple = ("a", "b")) -> Path:
    """Write a valid ``run_manifest.json`` into ``dirpath`` and return its path."""
    manifest = build_run_manifest(
        run_id="run-1",
        timestamp="2026-01-01T00:00:00+00:00",
        system_prompt=system_prompt,
        tool_names=tool_names,
        package_versions={"numpy": "2.2.0"},
        extra={"skill_coverage": "stated"},
    )
    path = Path(dirpath) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Check 1: no-order-path baseline
# ---------------------------------------------------------------------------


def test_fresh_install_has_no_live_order_path() -> None:
    """A fresh install (no MCP server) must pass the no-order-path check."""
    result = check_no_order_path(config=None)
    assert result.ok, result.detail
    assert "mandate" in result.detail.lower()


def test_live_order_path_is_mandate_gated() -> None:
    """The live-broker order branch must route through the mandate gate.

    ``place_order`` must call ``execute_live_order`` (mandate + kill switch +
    fail-closed pre-trade checks + audit). A future change that bypasses the
    gate breaks the baseline.
    """
    assert baseline_mod._live_order_path_is_gated()


def test_a_configured_live_broker_fails_the_fresh_install_baseline() -> None:
    """Enabling a live-broker MCP server is not self-certifying as a no-path."""
    config = {"mcp_servers": {"robinhood": {"url": "http://localhost:8000"}}}
    assert not baseline_mod._fresh_install_has_no_live_order_tool(config)


def test_no_order_path_negative_config() -> None:
    """A config that wires a live broker must not report a fresh no-path."""
    config = {"mcp_servers": {"ibkr": {"url": "https://localhost:8000"}}}
    result = check_no_order_path(config=config)
    assert not result.ok
    assert "live-broker" in result.detail


# ---------------------------------------------------------------------------
# Check 2: run manifest
# ---------------------------------------------------------------------------


def test_a_valid_run_manifest_passes(tmp_path) -> None:
    """A run directory carrying a valid hash manifest passes the check."""
    _write_manifest(tmp_path)
    assert check_run_manifest(tmp_path).ok


def test_a_missing_run_manifest_fails(tmp_path) -> None:
    """A run with no manifest does not satisfy the audit-ledger criterion."""
    result = check_run_manifest(tmp_path / "nope")
    assert not result.ok
    assert "missing" in result.detail


def test_a_tampered_manifest_fails(tmp_path) -> None:
    """A run_manifest.json whose hash no longer re-derives is a signature of tampering."""
    _write_manifest(tmp_path)
    path = tmp_path / "run_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["manifest_hash"] = "sha256:deadbeef" + data["manifest_hash"][16:]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not check_run_manifest(tmp_path).ok


def test_a_corrupt_manifest_fails(tmp_path) -> None:
    """An unreadable run_manifest.json fails safely (no false pass)."""
    (tmp_path / "run_manifest.json").write_text("{not json", encoding="utf-8")
    assert not check_run_manifest(tmp_path).ok


# ---------------------------------------------------------------------------
# Check 3: QMT Bridge no-write manifest (cross-link D-01)
# ---------------------------------------------------------------------------


def test_qmt_bridge_shipped_manifest_is_read_only() -> None:
    """The shipped bridge manifest declares no write capability (cross-link D-01)."""
    result = check_qmt_bridge_no_write()
    assert result.ok, result.detail
    assert "read_health" in result.detail and "read_quotes" in result.detail


def test_qmt_bridge_rejects_a_write_manifest() -> None:
    """A manifest declaring a write capability must be rejected (fail-closed).

    Cross-linked with D-01's ``qmt_bridge.capabilities.validate_manifest``:
    this is the exact gate that refuses to start the bridge if a write
    capability is ever (mistakenly) added.
    """
    caps = baseline_mod._import_source_module("qmt_bridge.capabilities")
    from qmt_bridge.capabilities import CapabilityManifest, WriteCapabilityError, validate_manifest

    write_manifest = CapabilityManifest(
        name="qmt-bridge", capabilities=frozenset({"read_quotes", "order_place"})
    )
    with pytest.raises(WriteCapabilityError):
        validate_manifest(write_manifest)

    # assert_read_only (used at startup) must refuse when the manifest declares write.
    try:
        caps.assert_read_only()
    except WriteCapabilityError:
        pytest.fail("assert_read_only() must accept the (read-only) shipped manifest")
    assert caps.manifest_payload()["write_capabilities"] is False


def test_order_capability_token_is_classified_write() -> None:
    """Sanity: the classifier recognises ``order_place`` as write, ``read_quotes`` not."""
    caps = baseline_mod._import_source_module("qmt_bridge.capabilities")
    assert caps.is_write_capability("order_place")
    assert not caps.is_write_capability("read_quotes")


# ---------------------------------------------------------------------------
# Check 4: credential tiering
# ---------------------------------------------------------------------------


def test_credential_tiers_are_redacted() -> None:
    """LLM / data-source / broker credentials must never reach plaintext logs."""
    result = check_credential_tiers()
    assert result.ok, result.detail


def test_credential_env_keys_are_not_forwarded_to_the_sandbox() -> None:
    """The generated-code backtest subprocess must not inherit data-source keys."""
    assert baseline_mod._credential_env_not_forwarded_to_sandbox()


def test_broker_audit_is_redacted() -> None:
    """Live-action audit records redact broker request/response before write."""
    assert baseline_mod._broker_audit_redaction_present()


def test_llm_provider_diagnostics_are_redacted() -> None:
    """LLM provider diagnostics make the API key / base URL non-recoverable."""
    assert baseline_mod._llm_credential_redaction_present()


# ---------------------------------------------------------------------------
# Aggregation + completeness
# ---------------------------------------------------------------------------


def test_baseline_aggregates_all_checks(tmp_path) -> None:
    """``run_baseline`` returns a report with every executable check."""
    _write_manifest(tmp_path)
    report = run_baseline(config=None, run_dir=tmp_path)
    assert isinstance(report, SecurityBaselineReport)
    assert report.all_ok, report.to_dict()
    names = {check.name for check in report.checks}
    assert "no_order_path" in names
    assert "qmt_bridge_no_write" in names
    assert "credential_tiers" in names
    assert "run_manifest" in names
    assert report.to_dict()["all_ok"] is True


def test_baseline_reports_a_failing_check(tmp_path) -> None:
    """A tampered manifest must make the aggregate report fail."""
    _write_manifest(tmp_path)
    path = tmp_path / "run_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["manifest_hash"] = "sha256:tampered"
    path.write_text(json.dumps(data), encoding="utf-8")
    report = run_baseline(config=None, run_dir=tmp_path)
    assert not report.all_ok
    assert not bool(report)


def test_order_tool_set_covers_every_non_readonly_write_tool() -> None:
    """Anti-drift gate: a new order tool must be added to the curated set.

    If a non-read-only tool appears in ``trading_connector_tool.py`` that is
    not one of the curated order tools (and not the profile selector), this
    fails so the security baseline does not silently miss a new order path.
    """
    declared = baseline_mod._order_tool_classes(_tool_source_path().read_text(encoding="utf-8"))
    expected = ORDER_PLACEMENT_TOOL_NAMES | _NON_ORDER_WRITE_TOOLS
    assert declared == expected, (
        f"Non-read-only tool set changed. Declared={sorted(declared)}; "
        f"expected={sorted(expected)}. Add any new order tool to "
        "ORDER_PLACEMENT_TOOL_NAMES and re-review its live path."
    )


def test_every_curated_order_tool_is_really_a_write_tool() -> None:
    """Every curated order tool must be a declared non-read-only BaseTool."""
    declared = baseline_mod._order_tool_classes(_tool_source_path().read_text(encoding="utf-8"))
    assert ORDER_PLACEMENT_TOOL_NAMES <= declared
