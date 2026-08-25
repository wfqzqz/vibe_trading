"""``miniqmt`` thin loader: read the QMT Bridge cache, cold-read the bridge HTTP.

The QMT Bridge (D-01, a standalone Windows service) is the authoritative A-share
source: it pulls ``xtquant.xtdata`` and persists qfq-adjusted OHLCV plus
复权/停牌/涨跌停/除权除息 metadata into the shared loader cache at
``~/.vibe-trading/cache/loaders/miniqmt/``. This loader is a *thin* reader over
that cache with a cold-read fallback to the bridge's read-only HTTP surface:

- ``is_available()`` probes the bridge (``GET /health``) and the local cache.
  Only when *both* are unreachable does it return ``False``, so the fallback
  chain (``resolve_loader``) moves on to the free sources without stalling.
- ``fetch()`` goes through :func:`backtest.loaders.base.cached_loader_fetch`
  (same content-addressed key the bridge writes, ``source="miniqmt"``,
  ``fields=None``); on a cache miss it cold-reads the bridge over HTTP.

The bridge normalizes volume to board lots (1 lot = 100 shares) at ingestion, so
this loader passes volume through unchanged and declares
``volume_units={"a_share": "lots"}`` (DORA-124 D-04 门禁 1). The metadata columns
``pre_close`` / ``limit_up`` / ``limit_down`` are qfq-relative (前复权基准) and are
preserved verbatim for ``china_a.py`` to consume on that same 口径 (DORA-156 条件 2).

Connection settings mirror the bridge's own env (``qmt_bridge.config``) and are
read through the unified config layer (``DataConfig.qmt_bridge_*``):
``QMT_BRIDGE_HOST`` (default ``127.0.0.1``), ``QMT_BRIDGE_PORT`` (default 8100),
``QMT_BRIDGE_TOKEN`` (optional loopback bearer token; when the bridge pins its
token in the DPAPI vault rather than the env, expose it here so the loader can
authenticate — without a token the loader still reads the shared cache and only
the cold-read path degrades).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.loaders._http import throttled_get
from backtest.loaders.base import (
    cached_loader_fetch,
    loader_cache_enabled,
    loader_cache_root,
    validate_date_range,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# Bridge connection settings, mirroring ``qmt_bridge.config`` (loopback only).
# The values themselves are read through the unified config layer
# (``agent/src/config/env_schema.py`` ``DataConfig.qmt_bridge_*``); these are
# the last-line fallbacks when the config is empty.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8100

#: The only 口径 the bridge persists to the loader cache (DORA-124 "前复权为主").
_ADJUST = "qfq"

#: Project interval token → bridge timeframe token. The bridge's minute surface
#: accepts ``1m/5m/15m/30m/60m/1h``; hourly is pinned to ``60m`` so the cache
#: key never forks on ``60m`` vs ``1h`` spellings. ``4H`` has no bridge period,
#: so it is rejected (the chain then falls through to a free source).
_INTERVAL_MAP: dict[str, str] = {
    "1D": "1d", "1d": "1d", "d": "1d", "day": "1d", "daily": "1d",
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "60m", "1h": "60m", "60m": "60m",
    "tick": "tick",
}

#: Loopback health probe timeout: short, so an absent bridge fails fast and the
#: fallback chain reaches a free source without stalling a backtest.
_HEALTH_TIMEOUT_S = 2.0
#: Cold-read quotes timeout (a slow xtdata pull can take a few seconds).
_QUOTES_TIMEOUT_S = 30.0

#: Shared throttle/session bucket; loopback needs no spacing, so min_interval=0.
_HOST_KEY = "miniqmt"


def _bridge_settings() -> tuple[str, int, str]:
    """Return ``(host, port, token)`` from the unified config layer.

    The values are read through :func:`src.config.accessor.get_env_config` —
    the single place that touches ``os.environ`` (env-var gate) — with the
    ``qmt_bridge.config`` defaults as a last-line fallback when the config
    returns an empty value.
    """
    from src.config.accessor import get_env_config

    data = get_env_config().data
    host = (data.qmt_bridge_host or "").strip() or _DEFAULT_HOST
    port = data.qmt_bridge_port or _DEFAULT_PORT
    token = (data.qmt_bridge_token or "").strip()
    return host, port, token


def _bridge_base_url() -> str:
    """Return the bridge base URL (``http://host:port``), no trailing slash."""
    host, port, _ = _bridge_settings()
    return f"http://{host}:{port}"


def _bridge_token() -> str:
    """Return the loopback bearer token from the env (``""`` when unset)."""
    return _bridge_settings()[2]


def _auth_headers(token: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _to_timeframe(interval: str) -> Optional[str]:
    return _INTERVAL_MAP.get(str(interval).strip())


def _normalize_symbol(code: str) -> str:
    """Canonical upper-case symbol, matching ``qmt_bridge.metadata.normalize_symbol``."""
    return str(code).strip().upper()


def _parse_bars(payload: Any, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Convert a bridge quotes response into an ascending OHLCV frame.

    The bridge serializes a frame as ``{"bars": [{trade_date, open, ...}, ...]}``
    (``service._frame_to_records``). Daily bars also carry the metadata columns
    ``pre_close/suspended/limit_up/limit_down/ex_dividend/adj_factor``, which are
    preserved so ``china_a.py`` consumes the same qfq-relative basis.

    Returns:
        A DataFrame indexed by ``trade_date`` with float OHLCV columns (and any
        metadata columns the bridge attached), trimmed to the inclusive window,
        or ``None`` when the payload holds no usable bars.
    """
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not bars:
        return None

    rows = [bar for bar in bars if isinstance(bar, dict)]
    if not rows:
        return None
    df = pd.DataFrame(rows)

    index_col = next((c for c in ("trade_date", "date", "time", "datetime") if c in df.columns), None)
    if index_col is None:
        return None
    df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
    df = df.dropna(subset=[index_col])
    df = df.set_index(index_col).sort_index()
    df.index.name = "trade_date"

    # Cast every numeric OHLCV column to float (the shared loader contract).
    for field in ("open", "high", "low", "close", "volume", "amount"):
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce").astype(float)

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Trim defensively to the inclusive window (end inclusive of its whole day).
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    if df.empty:
        return None

    # Canonical column order: OHLCV first, then any extra bridge columns.
    ordered = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


@register
class DataLoader:
    """Thin ``miniqmt`` A-share loader (reads the bridge cache, cold-reads HTTP)."""

    name = "miniqmt"
    markets = {"a_share"}
    #: Bar periods the bridge can serve (DORA-124 §3.2). ``60m`` is the hourly
    #: token; ``tick`` is tiered/optional on the bridge side.
    intervals = {"1d", "1m", "5m", "15m", "30m", "60m", "tick"}
    #: The bridge encrypts its own credentials; the loader holds none.
    requires_auth = False
    #: The bridge normalizes volume to board lots at ingestion (1 lot = 100
    #: shares), so the served unit is ``"lots"`` — the canonical A-share unit
    #: shared with tencent/eastmoney/baostock/akshare/mootdx/tushare (#1062).
    volume_units = {"a_share": "lots"}

    def __init__(self) -> None:
        pass

    def is_available(self) -> bool:
        """Return whether the loader can serve data right now.

        True when the bridge is reachable (cold reads work) or the shared cache
        already holds miniqmt data (settled ranges can be served even while the
        bridge is down). False only when *both* are unreachable, so
        ``resolve_loader`` moves on to the free sources.
        """
        return self._bridge_reachable() or self._cache_has_data()

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV frames keyed by the original input codes.

        Each symbol goes through :func:`cached_loader_fetch` with the exact
        ``source="miniqmt"`` / ``fields=None`` identity the bridge writes, so a
        settled range is served from cache; a cache miss cold-reads the bridge.
        A transient per-symbol failure logs and skips so one symbol never poisons
        the batch (the fallback chain then serves the missing symbols).

        Args:
            codes: A-share symbols (e.g. ``["600519.SH", "000001.SZ"]``).
            start_date: Inclusive start date, ``YYYY-MM-DD``.
            end_date: Inclusive end date, ``YYYY-MM-DD``.
            interval: Bar size token; unsupported tokens (``4H``) are rejected.
            fields: Ignored — the bridge returns a fixed OHLCV + metadata schema.

        Returns:
            Mapping ``{symbol: DataFrame(trade_date, open, high, low, close,
            volume[, amount, pre_close, suspended, limit_up, limit_down,
            ex_dividend, adj_factor])}`` for every symbol with data.
        """
        validate_date_range(start_date, end_date)
        del fields  # the bridge has no extra-field surface

        timeframe = _to_timeframe(interval)
        if timeframe is None:
            logger.warning("miniqmt unsupported interval %r; rejecting", interval)
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            clean = code.strip()
            if not clean:
                continue
            norm = _normalize_symbol(clean)
            try:
                frame = cached_loader_fetch(
                    source=self.name,
                    symbol=norm,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda n=norm: self._fetch_one(n, timeframe, start_date, end_date),
                )
            except Exception as exc:  # noqa: BLE001 - one symbol never poisons the batch
                logger.warning("miniqmt failed for %s: %s", clean, exc)
                continue
            if frame is not None and not frame.empty:
                result[code] = frame
        return result

    # -- internals -----------------------------------------------------------

    def _bridge_reachable(self) -> bool:
        """Return whether the bridge answers ``GET /health`` (best-effort)."""
        token = _bridge_token()
        try:
            response = throttled_get(
                f"{_bridge_base_url()}/health",
                host_key=_HOST_KEY,
                min_interval=0.0,
                headers=_auth_headers(token),
                timeout=_HEALTH_TIMEOUT_S,
            )
            return response.status_code == 200
        except Exception as exc:  # noqa: BLE001 - any probe failure means unreachable
            logger.debug("miniqmt bridge unreachable: %s", exc)
            return False

    def _cache_has_data(self) -> bool:
        """Return whether the shared miniqmt cache holds at least one entry."""
        try:
            if not loader_cache_enabled():
                return False
            cache_dir = loader_cache_root() / "miniqmt"
            return cache_dir.is_dir() and any(cache_dir.glob("*.parquet"))
        except Exception as exc:  # noqa: BLE001 - cache probe is best-effort
            logger.debug("miniqmt cache probe failed: %s", exc)
            return False

    def _fetch_one(
        self, code: str, timeframe: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Cold-read one symbol's quotes from the bridge; ``None`` on no data."""
        token = _bridge_token()
        headers = _auth_headers(token)

        if timeframe == "1d":
            url = f"{_bridge_base_url()}/v1/quotes/daily"
            params: Dict[str, Any] = {
                "symbol": code, "start": start_date, "end": end_date, "adjust": _ADJUST,
            }
        elif timeframe == "tick":
            url = f"{_bridge_base_url()}/v1/quotes/tick"
            params = {"symbol": code, "start": start_date, "end": end_date}
        else:  # minute periods (1m/5m/15m/30m/60m)
            url = f"{_bridge_base_url()}/v1/quotes/minute"
            params = {
                "symbol": code, "start": start_date, "end": end_date,
                "period": timeframe, "adjust": _ADJUST,
            }

        try:
            response = throttled_get(
                url,
                host_key=_HOST_KEY,
                min_interval=0.0,
                params=params,
                headers=headers,
                timeout=_QUOTES_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - a failed cold read degrades to None
            logger.warning("miniqmt cold read failed for %s: %s", code, exc)
            return None

        # 503 = the bridge reports ``unavailable`` (xtdata down) — no data, not
        # an error; the fallback chain serves the symbol.
        if response.status_code == 503:
            return None
        if response.status_code != 200:
            logger.warning(
                "miniqmt bridge returned HTTP %s for %s", response.status_code, code
            )
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("miniqmt bridge returned non-JSON for %s: %s", code, exc)
            return None

        if isinstance(payload, dict) and payload.get("unavailable"):
            return None
        return _parse_bars(payload, start_date, end_date)
