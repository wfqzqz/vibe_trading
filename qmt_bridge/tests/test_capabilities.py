"""Tests for the capability manifest and read-only enforcement."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qmt_bridge.capabilities import (
    CapabilityManifest,
    SHIPPED_MANIFEST,
    WriteCapabilityError,
    assert_read_only,
    is_write_capability,
    manifest_payload,
    validate_manifest,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def test_shipped_manifest_is_read_only() -> None:
    assert_read_only()  # must not raise
    payload = manifest_payload()
    assert payload["write_capabilities"] is False
    assert payload["name"] == "qmt-bridge"
    for token in SHIPPED_MANIFEST.capabilities:
        assert not is_write_capability(token)


def test_explicit_write_capability_is_rejected() -> None:
    manifest = CapabilityManifest(name="bad", capabilities=frozenset({"order_place"}))
    with pytest.raises(WriteCapabilityError, match="order_place"):
        validate_manifest(manifest)


def test_write_prefix_capability_is_rejected() -> None:
    manifest = CapabilityManifest(name="bad", capabilities=frozenset({"trade_manual"}))
    with pytest.raises(WriteCapabilityError):
        validate_manifest(manifest)


def test_read_capabilities_are_accepted() -> None:
    manifest = CapabilityManifest(name="ok", capabilities=frozenset({"read_quotes", "read_meta"}))
    validate_manifest(manifest)  # must not raise


def test_blank_token_is_not_write() -> None:
    assert is_write_capability("") is False
    assert is_write_capability("read_health") is False


def test_no_xttrader_import_anywhere_in_package() -> None:
    """Structural gate: the bridge must never import ``xttrader``.

    Parses every module directly under ``qmt_bridge/`` and rejects any
    ``import`` / ``from ... import`` statement whose module path mentions
    ``xttrader``. This is the acceptance criterion "结构性不引入 xttrader".
    """
    offenders: list[str] = []
    for module_path in sorted(_PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if "xttrader" in name.lower():
                    offenders.append(f"{module_path.name}:{node.lineno}: {name}")
    assert not offenders, f"xttrader import found: {offenders}"
