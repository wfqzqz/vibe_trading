"""Tests for the read-only FastAPI surface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from qmt_bridge.api import create_app
from qmt_bridge.service import BridgeService
from fakes import FakeProvider

TOKEN = "test-token"


def _client(available: bool = True, token: str | None = TOKEN) -> TestClient:
    service = BridgeService(FakeProvider(available=available))
    app = create_app(service, token=token)
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_requires_token() -> None:
    client = _client()
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=_auth()).status_code == 200


def test_health_payload() -> None:
    client = _client()
    body = client.get("/health", headers=_auth()).json()
    assert body["read_only"] is True
    assert body["xtdata_available"] is True


def test_manifest_is_read_only() -> None:
    client = _client()
    body = client.get("/v1/manifest", headers=_auth()).json()
    assert body["write_capabilities"] is False


def test_post_to_quotes_is_405() -> None:
    client = _client()
    resp = client.post("/v1/quotes/daily", headers=_auth(), json={})
    assert resp.status_code == 405


def test_post_to_unknown_path_is_405() -> None:
    client = _client()
    resp = client.post("/v1/orders", headers=_auth(), json={"symbol": "x"})
    assert resp.status_code == 405


def test_put_is_405() -> None:
    client = _client()
    assert client.put("/v1/meta", headers=_auth()).status_code == 405


def test_daily_returns_provenance() -> None:
    client = _client()
    resp = client.get(
        "/v1/quotes/daily",
        params={"symbol": "600519.SH", "start": "2024-01-01", "end": "2024-01-31"},
        headers=_auth(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["source"] == "miniqmt"
    assert body["provenance"]["adjust"] == "qfq"
    assert len(body["bars"]) == 3


def test_daily_unavailable_is_503() -> None:
    client = _client(available=False)
    resp = client.get(
        "/v1/quotes/daily",
        params={"symbol": "600519.SH", "start": "2024-01-01", "end": "2024-01-31"},
        headers=_auth(),
    )
    assert resp.status_code == 503
    assert resp.json()["unavailable"] is True


def test_meta_endpoint() -> None:
    client = _client()
    resp = client.get("/v1/meta", params={"symbol": "600519.SH"}, headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "600519.SH"


def test_no_token_configured_allows_access() -> None:
    client = _client(token=None)
    assert client.get("/health").status_code == 200
