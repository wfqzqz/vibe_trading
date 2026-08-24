"""FastAPI read-only HTTP surface for the QMT Bridge.

The route table registers **only** ``GET`` endpoints, and a middleware rejects
every non-``GET``/``HEAD``/``OPTIONS`` method with ``405`` before routing, so a
write endpoint cannot be added accidentally and a client can never POST a
mutation — the "structural no-write surface" acceptance criterion.

Auth: when a token is configured the bridge requires
``Authorization: Bearer <token>`` (or ``X-API-Token: <token>``) on every route
and otherwise replies ``401``. The server binds to loopback only (see
``server.py``), so the token plus the loopback bind are the access boundary.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from qmt_bridge.capabilities import manifest_payload
from qmt_bridge.service import BridgeService

__all__ = ["create_app", "ALLOWED_METHODS"]

ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer "):].strip()
    return request.headers.get("x-api-token")


def create_app(
    service: BridgeService,
    *,
    token: str | None = None,
) -> FastAPI:
    """Build the bridge FastAPI app.

    Args:
        service: The :class:`BridgeService` backing the endpoints.
        token: Optional loopback API token. When set, every route requires it.
    """
    app = FastAPI(
        title="QMT Bridge",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def enforce_read_only(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        if request.method not in ALLOWED_METHODS:
            return JSONResponse(status_code=405, content={"detail": "method not allowed (read-only)"})
        return await call_next(request)

    async def require_token(request: Request) -> None:
        if token and _bearer_token(request) != token:
            raise HTTPException(status_code=401, detail="missing or invalid API token")

    def _respond(result: dict[str, Any]) -> JSONResponse:
        if result.get("unavailable"):
            return JSONResponse(status_code=503, content=result)
        return JSONResponse(content=result)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        await require_token(request)
        return JSONResponse(content=service.health())

    @app.get("/v1/manifest")
    async def manifest(request: Request) -> JSONResponse:
        await require_token(request)
        return JSONResponse(content=manifest_payload())

    @app.get("/v1/quotes/daily")
    async def daily(
        request: Request,
        symbol: str = Query(..., description="A-share symbol, e.g. 600519.SH"),
        start: str = Query(..., description="Start date YYYY-MM-DD"),
        end: str = Query(..., description="End date YYYY-MM-DD"),
        adjust: str | None = Query(None, description="qfq | hfq | none"),
    ) -> JSONResponse:
        await require_token(request)
        return _respond(service.daily(symbol, start, end, adjust))

    @app.get("/v1/quotes/minute")
    async def minute(
        request: Request,
        symbol: str = Query(..., description="A-share symbol, e.g. 600519.SH"),
        start: str = Query(..., description="Start date YYYY-MM-DD"),
        end: str = Query(..., description="End date YYYY-MM-DD"),
        period: str = Query("1m", description="1m | 5m | 15m | 30m | 60m | 1h"),
        adjust: str | None = Query(None, description="qfq | hfq | none"),
    ) -> JSONResponse:
        await require_token(request)
        return _respond(service.minute(symbol, start, end, period, adjust))

    @app.get("/v1/quotes/tick")
    async def tick(
        request: Request,
        symbol: str = Query(..., description="A-share symbol, e.g. 600519.SH"),
        start: str = Query(..., description="Start date YYYY-MM-DD"),
        end: str = Query(..., description="End date YYYY-MM-DD"),
    ) -> JSONResponse:
        await require_token(request)
        return _respond(service.tick(symbol, start, end))

    @app.get("/v1/meta")
    async def meta(
        request: Request,
        symbol: str = Query(..., description="A-share symbol, e.g. 600519.SH"),
    ) -> JSONResponse:
        await require_token(request)
        return _respond(service.meta(symbol))

    return app
