"""Configuration for the QMT Bridge (non-secret settings only).

Secrets (the miniQMT account id and the loopback API token) are *not* kept
here — they live in :class:`qmt_bridge.credentials.SecretVault`, encrypted at
rest. This module reads only non-secret environment variables.

The cache switch/root deliberately mirror the agent's ``base.py`` convention
(``VIBE_TRADING_DATA_CACHE`` / ``VIBE_TRADING_DATA_CACHE_ROOT``) so the bridge
and the agent agree on where the shared ``miniqmt`` loader cache lives.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Callable, Mapping

__all__ = [
    "Settings",
    "load_settings",
    "resolve_api_token",
    "generate_token",
    "TOKEN_FIELD",
    "ACCOUNT_FIELD",
]

#: Default loopback bind address — the bridge never listens on a public NIC.
DEFAULT_HOST = "127.0.0.1"
#: Default port (kept off the agent's 8000 to avoid collision).
DEFAULT_PORT = 8100

#: The same truthy set the agent's loader cache uses.
_CACHE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: Vault field name for the persisted loopback API token.
TOKEN_FIELD = "api_token"
#: Vault field name for the (optional) miniQMT account id.
ACCOUNT_FIELD = "qmt_account_id"

_Get = Callable[[str], str | None]


@dataclass(frozen=True)
class Settings:
    """Immutable bridge settings.

    Attributes:
        host: Loopback bind address.
        port: TCP port.
        cache_enabled: Whether the loader cache is enabled.
        cache_root: Cache root directory (default under the user's home).
        adjust: Default adjustment for quotes (``"qfq"`` primary).
        tick_persist: Whether tick data is persisted (default off, volume
            control — tick is tiered and short-lived).
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    cache_enabled: bool = False
    cache_root: str = ""
    adjust: str = "qfq"
    tick_persist: bool = False


def _env_bool(get: _Get, name: str, default: bool) -> bool:
    raw = get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _CACHE_TRUE_VALUES


def _env_int(get: _Get, name: str, default: int) -> int:
    raw = get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw))
    except ValueError:
        return default
    return value if value > 0 else default


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from environment variables (or an explicit mapping).

    Args:
        env: Optional mapping used instead of ``os.environ`` (for tests).

    Returns:
        A :class:`Settings` instance.
    """
    get: _Get = (env or os.environ).get

    adjust = str(get("QMT_BRIDGE_ADJUST") or "qfq").strip().lower()
    if adjust not in {"qfq", "hfq", "none"}:
        adjust = "qfq"

    host = str(get("QMT_BRIDGE_HOST") or DEFAULT_HOST).strip() or DEFAULT_HOST
    return Settings(
        host=host,
        port=_env_int(get, "QMT_BRIDGE_PORT", DEFAULT_PORT),
        cache_enabled=_env_bool(get, "VIBE_TRADING_DATA_CACHE", False),
        cache_root=str(get("VIBE_TRADING_DATA_CACHE_ROOT") or "").strip(),
        adjust=adjust,
        tick_persist=_env_bool(get, "QMT_BRIDGE_TICK_PERSIST", False),
    )


def generate_token() -> str:
    """Return a fresh URL-safe API token (43 chars, ~256 bits of entropy)."""
    return secrets.token_urlsafe(32)


def resolve_api_token(vault) -> str:
    """Resolve the loopback API token for this process.

    Order of precedence:
      1. ``QMT_BRIDGE_TOKEN`` environment override (operators pin a token).
      2. The encrypted ``api_token`` already persisted in ``vault``.
      3. A freshly generated token, persisted into ``vault`` for next start.

    Args:
        vault: A :class:`qmt_bridge.credentials.SecretVault`.

    Returns:
        The active token string.
    """
    override = os.getenv("QMT_BRIDGE_TOKEN")
    if override and override.strip():
        return override.strip()

    existing = vault.get(TOKEN_FIELD)
    if existing:
        return existing

    token = generate_token()
    vault.set(TOKEN_FIELD, token)
    return token
