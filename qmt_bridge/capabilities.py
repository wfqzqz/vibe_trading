"""Capability manifest for the QMT Bridge, and its read-only enforcement.

The bridge inherits the upstream governance idea of a *declared* capability
surface: the manifest is a plain data object naming exactly what the service
can do. The service is **structurally read-only** — the shipped manifest
declares only read capabilities, and :func:`validate_manifest` rejects any
manifest that declares a write capability. Startup calls
:func:`assert_read_only` and refuses to serve when the manifest would grant
write access, so a future edit that (mistakenly) adds a write capability fails
closed instead of silently enabling a trade path.

This module is standalone: no ``xtquant`` / ``xttrader`` import, no wall-clock
read, no filesystem access. It can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "WRITE_CAPABILITY_TOKENS",
    "CapabilityManifest",
    "SHIPPED_MANIFEST",
    "WriteCapabilityError",
    "is_write_capability",
    "validate_manifest",
    "assert_read_only",
    "manifest_payload",
]

#: Explicit capability tokens that constitute a write / trade path.
WRITE_CAPABILITY_TOKENS: frozenset[str] = frozenset(
    {
        "write",
        "trade",
        "order_place",
        "order_cancel",
        "order_modify",
        "order_query",
        "account_mutate",
        "position_mutate",
        "transfer",
        "withdraw",
        "deposit",
        "subscribe_mutate",
    }
)

#: Capability prefixes that also denote write access (so a future capability
#: like ``order_place_stock`` or ``account_close`` cannot slip through the
#: explicit-token allowlist).
_WRITE_PREFIXES: tuple[str, ...] = (
    "write",
    "trade",
    "order_",
    "account_",
    "position_",
    "transfer_",
)

#: The read-only capability surface this service ships with. Any addition here
#: must stay inside the read category — a write token makes startup fail.
_READ_CAPABILITIES: tuple[str, ...] = (
    "read_health",
    "read_quotes",
    "read_meta",
)


@dataclass(frozen=True)
class CapabilityManifest:
    """A named, immutable declaration of one service's capabilities.

    Attributes:
        name: Service name (e.g. ``"qmt-bridge"``).
        capabilities: The declared capability tokens.
    """

    name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)


#: The manifest shipped with this source tree — read-only by construction.
SHIPPED_MANIFEST: CapabilityManifest = CapabilityManifest(
    name="qmt-bridge",
    capabilities=frozenset(_READ_CAPABILITIES),
)


class WriteCapabilityError(Exception):
    """Raised when a capability manifest declares write access."""


def is_write_capability(token: str) -> bool:
    """Return whether ``token`` denotes a write / trade capability.

    Args:
        token: A capability token.

    Returns:
        ``True`` when the token is an explicit write token or starts with a
        write prefix; ``False`` otherwise.
    """
    normalized = str(token).strip().lower()
    if not normalized:
        return False
    if normalized in WRITE_CAPABILITY_TOKENS:
        return True
    return normalized.startswith(_WRITE_PREFIXES)


def validate_manifest(manifest: CapabilityManifest) -> None:
    """Reject a manifest that declares any write capability.

    Args:
        manifest: The manifest to validate.

    Raises:
        WriteCapabilityError: When at least one declared capability is a write
            capability. The offending tokens are named in the message.
    """
    offenders = sorted(
        token for token in manifest.capabilities if is_write_capability(token)
    )
    if offenders:
        raise WriteCapabilityError(
            f"capability manifest {manifest.name!r} declares write capabilities: "
            f"{', '.join(offenders)}"
        )


def assert_read_only() -> None:
    """Validate the shipped manifest; the process must refuse to serve on failure.

    Raises:
        WriteCapabilityError: When the shipped manifest declares write access.
    """
    validate_manifest(SHIPPED_MANIFEST)


def manifest_payload() -> dict[str, object]:
    """Return a JSON-serializable, read-only capability manifest.

    Returns:
        ``{name, capabilities, write_capabilities}`` where
        ``write_capabilities`` is always ``False`` for the shipped manifest.
    """
    return {
        "name": SHIPPED_MANIFEST.name,
        "capabilities": sorted(SHIPPED_MANIFEST.capabilities),
        "write_capabilities": False,
    }


def _coerce_manifest(name: str, capabilities: Iterable[str]) -> CapabilityManifest:
    """Build a manifest from loose inputs (used by tests and config loads)."""
    return CapabilityManifest(name=str(name), capabilities=frozenset(capabilities))
