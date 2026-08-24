"""Tests for the DPAPI-backed secret vault (no plaintext at rest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qmt_bridge.credentials import (
    SecretVault,
    SecretVaultError,
    SecretVaultUnavailableError,
)


class _ReversibleBackend:
    """Deterministic in-memory cipher: base64(plaintext) with a marker."""

    def protect(self, plaintext: bytes) -> bytes:
        return b"enc:" + plaintext

    def unprotect(self, ciphertext: bytes) -> bytes:
        if not ciphertext.startswith(b"enc:"):
            raise SecretVaultError("bad ciphertext")
        return ciphertext[len(b"enc:"):]


class _RaisingBackend:
    def protect(self, plaintext: bytes) -> bytes:
        raise SecretVaultUnavailableError("no dpapi")

    def unprotect(self, ciphertext: bytes) -> bytes:
        raise SecretVaultUnavailableError("no dpapi")


def _vault(tmp_path: Path) -> SecretVault:
    return SecretVault(path=tmp_path / "secrets.v1.json", backend=_ReversibleBackend())


def test_roundtrip(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.set("api_token", "secret-token-123")
    assert vault.get("api_token") == "secret-token-123"


def test_missing_field_returns_none(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    assert vault.get("nope") is None


def test_blank_value_removes_field(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.set("api_token", "secret-token-123")
    vault.set("api_token", "   ")
    assert vault.get("api_token") is None
    assert "api_token" not in vault.fields()


def test_file_contains_no_plaintext(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    secret = "super-secret-plaintext"
    vault.set("api_token", secret)
    raw = (tmp_path / "secrets.v1.json").read_text(encoding="utf-8")
    assert secret not in raw
    # The stored value is base64 of the ciphertext, never the raw secret.
    payload = json.loads(raw)
    assert payload["values"]["api_token"] != secret


def test_delete_removes_field(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.set("api_token", "secret-token-123")
    vault.delete("api_token")
    assert vault.get("api_token") is None


def test_corrupt_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "secrets.v1.json"
    path.write_text("{not json", encoding="utf-8")
    vault = SecretVault(path=path, backend=_ReversibleBackend())
    with pytest.raises(SecretVaultError):
        vault.get("api_token")


def test_unsupported_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "secrets.v1.json"
    path.write_text(json.dumps({"version": 99, "values": {}}), encoding="utf-8")
    vault = SecretVault(path=path, backend=_ReversibleBackend())
    with pytest.raises(SecretVaultError):
        vault.get("api_token")


def test_unavailable_backend_raises(tmp_path: Path) -> None:
    vault = SecretVault(path=tmp_path / "secrets.v1.json", backend=_RaisingBackend())
    with pytest.raises(SecretVaultUnavailableError):
        vault.set("api_token", "x")
