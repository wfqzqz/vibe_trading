"""OS-backed secret storage for the QMT Bridge (no plaintext at rest).

The bridge needs a small set of secrets: the miniQMT account id used for
tick/meta subscriptions and the loopback API token that guards the read-only
HTTP surface. Both are encrypted with the Windows Data Protection API
(``CryptProtectData``) — the same machine/user-bound primitive the desktop
Electron shell uses via ``safeStorage`` — and stored base64-encoded in a JSON
file, so no secret is ever written to disk in the clear and nothing sensitive
is logged.

The DPAPI backend is imported/used lazily and is pluggable: tests inject an
in-memory double so the round-trip can be verified on any platform, while the
real backend raises an actionable error when DPAPI is unavailable (non-Windows
or a locked user profile).
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import json
import os
import uuid
from pathlib import Path
from typing import Protocol

__all__ = [
    "SecretBackend",
    "SecretVault",
    "DpapiBackend",
    "SecretVaultError",
    "SecretVaultUnavailableError",
    "default_secrets_path",
]

#: On-disk format version. Bump on a layout change so stale files are rejected.
_VAULT_FORMAT_VERSION = 1

_SECRETS_DIR = Path.home() / ".vibe-trading" / "qmt-bridge"
_SECRETS_FILENAME = "secrets.v1.json"


def default_secrets_path() -> Path:
    """Return the default vault file path under the user's home directory."""
    return _SECRETS_DIR / _SECRETS_FILENAME


class SecretVaultError(Exception):
    """Raised when a secret cannot be read or written."""


class SecretVaultUnavailableError(SecretVaultError):
    """Raised when the encryption backend is unavailable (e.g. non-Windows)."""


class SecretBackend(Protocol):
    """Encrypt/decrypt primitive used by :class:`SecretVault`."""

    def protect(self, plaintext: bytes) -> bytes: ...

    def unprotect(self, ciphertext: bytes) -> bytes: ...


class DpapiBackend:
    """Windows DPAPI backend via ``crypt32.dll`` (CryptProtectData/CryptUnprotectData).

    The protection scope is the current user + machine, with the UI prompt
    forbidden (``CRYPTPROTECT_UI_FORBIDDEN``), so encrypt/decrypt never blocks
    on a dialog. There is no key material to manage: the OS holds it.
    """

    _CRYPTPROTECT_UI_FORBIDDEN = 0x01

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_char))]

    @staticmethod
    def _blob_from_bytes(data: bytes) -> "_DATA_BLOB":
        buffer = ctypes.create_string_buffer(data, len(data))
        return DpapiBackend._DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))

    def protect(self, plaintext: bytes) -> bytes:
        """Encrypt ``plaintext`` for the current user; return raw ciphertext bytes."""
        if os.name != "nt":
            raise SecretVaultUnavailableError(
                "DPAPI credential encryption is only available on Windows."
            )
        in_blob = self._blob_from_bytes(plaintext)
        out_blob = self._DATA_BLOB()
        try:
            crypt32 = ctypes.windll.crypt32
        except (AttributeError, OSError) as exc:
            raise SecretVaultUnavailableError(
                "crypt32.dll is unavailable; cannot encrypt credentials."
            ) from exc
        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            self._CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            raise SecretVaultError("CryptProtectData failed; credential not encrypted.")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)

    def unprotect(self, ciphertext: bytes) -> bytes:
        """Decrypt ``ciphertext`` for the current user; return plaintext bytes."""
        if os.name != "nt":
            raise SecretVaultUnavailableError(
                "DPAPI credential decryption is only available on Windows."
            )
        in_blob = self._blob_from_bytes(ciphertext)
        out_blob = self._DATA_BLOB()
        try:
            crypt32 = ctypes.windll.crypt32
        except (AttributeError, OSError) as exc:
            raise SecretVaultUnavailableError(
                "crypt32.dll is unavailable; cannot decrypt credentials."
            ) from exc
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            self._CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            raise SecretVaultError("CryptUnprotectData failed; credential could not be decrypted.")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)


class SecretVault:
    """Persist small secrets as base64 ciphertext in a versioned JSON file.

    The file holds only ciphertext — never the plaintext secret — so a disk
    snapshot cannot reveal it. Writes are atomic (tmp file + ``os.replace``)
    and never fail a fetch: read/write errors raise :class:`SecretVaultError`
    for the caller (auth/credential handling) to decide how to degrade.

    Args:
        path: Vault file path. Defaults to
            ``~/.vibe-trading/qmt-bridge/secrets.v1.json``.
        backend: Encryption primitive. Defaults to :class:`DpapiBackend`; tests
            inject an in-memory double.
    """

    def __init__(
        self,
        path: Path | None = None,
        backend: SecretBackend | None = None,
    ) -> None:
        self._path = path or default_secrets_path()
        self._backend: SecretBackend = backend or DpapiBackend()
        self._values: dict[str, str] = {}
        self._loaded = False

    # -- internal -----------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SecretVaultError(f"cannot read secret vault {self._path}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise SecretVaultError(f"secret vault {self._path} is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("version") != _VAULT_FORMAT_VERSION:
            raise SecretVaultError(f"secret vault {self._path} has an unsupported format")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise SecretVaultError(f"secret vault {self._path} is malformed")
        self._values = {str(k): v for k, v in values.items() if isinstance(v, str)}

    def _persist(self) -> None:
        payload = {"version": _VAULT_FORMAT_VERSION, "values": self._values}
        unique = f"{os.getpid()}.{uuid.uuid4().hex}"
        tmp_path = self._path.with_name(f"{self._path.name}.{unique}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._path)
        except OSError as exc:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise SecretVaultError(f"cannot write secret vault {self._path}: {exc}") from exc

    # -- public -------------------------------------------------------------

    def set(self, field: str, value: str) -> None:
        """Encrypt and store ``value`` under ``field``; blank values are dropped.

        A blank value removes the field (mirroring the desktop shell, so a
        partial form submission cannot erase an unrelated secret).

        Raises:
            SecretVaultError: On encryption or persistence failure.
        """
        self._load()
        normalized = str(value or "").strip()
        if not normalized:
            self._values.pop(field, None)
            self._persist()
            return
        ciphertext = self._backend.protect(normalized.encode("utf-8"))
        self._values[field] = base64.b64encode(ciphertext).decode("ascii")
        self._persist()

    def get(self, field: str) -> str | None:
        """Decrypt and return the stored value for ``field``, or ``None``.

        Raises:
            SecretVaultError: On load/decryption failure.
        """
        self._load()
        encoded = self._values.get(field)
        if encoded is None:
            return None
        try:
            ciphertext = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SecretVaultError(f"secret {field!r} is not valid base64") from exc
        try:
            return self._backend.unprotect(ciphertext).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SecretVaultError(f"secret {field!r} did not decrypt to text") from exc

    def delete(self, field: str) -> None:
        """Remove ``field`` from the vault (no-op when absent)."""
        self._load()
        if field in self._values:
            del self._values[field]
            self._persist()

    def fields(self) -> list[str]:
        """Return the sorted list of field names with a stored value."""
        self._load()
        return sorted(self._values)

    def has(self, field: str) -> bool:
        """Return whether ``field`` has a stored (encrypted) value."""
        self._load()
        return field in self._values
