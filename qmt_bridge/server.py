"""Uvicorn entry point for the QMT Bridge.

``run`` performs the read-only manifest check first — the process refuses to
serve if the shipped manifest ever declares a write capability (fail closed) —
then binds to loopback with the configured token and serves.
"""

from __future__ import annotations

import logging

from qmt_bridge.api import create_app
from qmt_bridge.capabilities import assert_read_only
from qmt_bridge.config import Settings, load_settings, resolve_api_token
from qmt_bridge.credentials import SecretVault
from qmt_bridge.service import BridgeService
from qmt_bridge.xtdata_client import XtdataClient

__all__ = ["build_app", "run"]

logger = logging.getLogger(__name__)


def build_app(
    settings: Settings | None = None,
    vault: SecretVault | None = None,
    token: str | None = None,
):
    """Assemble the app from settings + a token (without starting uvicorn).

    Raises:
        WriteCapabilityError: If the shipped manifest declares write access.
    """
    assert_read_only()
    settings = settings or load_settings()
    vault = vault or SecretVault()
    active_token = token if token is not None else resolve_api_token(vault)
    service = BridgeService(XtdataClient(), settings=settings, cache_root=settings.cache_root)
    return create_app(service, token=active_token)


def run() -> None:
    """Start the bridge (blocking)."""
    settings = load_settings()
    app = build_app(settings=settings)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
