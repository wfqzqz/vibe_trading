"""Loader-cache writer for the QMT Bridge (mirrors ``backtest/loaders/base.py``).

The bridge is a standalone process, so it cannot import the agent's
``base.py`` helpers (``loader_cache_root`` reaches into the agent's config
accessor). Instead it re-implements the *convention* byte-for-byte — the same
content-addressed key, the same ``~/.vibe-trading/cache/loaders/<source>/``
layout, the same ``VIBE_TRADING_DATA_CACHE`` opt-in switch, the same
"only settled days are cacheable" rule, and the same parquet + JSON-metadata
sidecar — so the ``miniqmt`` loader (D-02) reads the bridge's writes through
the agent's own ``loader_cache_get`` without any drift.

Compatibility is pinned by ``tests/test_cache.py``, which asserts the bridge's
key and path functions return byte-identical results to ``base.py`` for the
same inputs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

__all__ = [
    "LOADER_CACHE_VERSION",
    "cache_enabled",
    "loader_cache_root",
    "make_loader_cache_key",
    "loader_cache_path",
    "range_is_final",
    "write_frame",
]

logger = logging.getLogger(__name__)

#: Bump only in lockstep with ``backtest.loaders.base._LOADER_CACHE_VERSION``.
LOADER_CACHE_VERSION = 4

_LOADER_CACHE_ENV = "VIBE_TRADING_DATA_CACHE"
_LOADER_CACHE_ROOT_ENV = "VIBE_TRADING_DATA_CACHE_ROOT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: The source name under which the bridge writes (the ``miniqmt`` loader reads
#: the same name, so the source-directory segment must match).
SOURCE_NAME = "miniqmt"


def cache_enabled() -> bool:
    """Return whether the opt-in loader cache is enabled (raw env read)."""
    raw = os.getenv(_LOADER_CACHE_ENV)
    return bool(raw) and raw.strip().lower() in _TRUE_VALUES


def loader_cache_root(cache_root: str | None = None) -> Path:
    """Return the loader cache root directory.

    Honors an explicit ``cache_root`` argument, else the
    ``VIBE_TRADING_DATA_CACHE_ROOT`` environment override, else the default
    ``~/.vibe-trading/cache/loaders`` — identical to ``base.py``.
    """
    override = cache_root if cache_root is not None else os.getenv(_LOADER_CACHE_ROOT_ENV)
    if isinstance(override, str) and override.strip():
        return Path(override).expanduser()
    return Path.home() / ".vibe-trading" / "cache" / "loaders"


def _loader_cache_payload(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    fields: Sequence[str] | None,
) -> dict[str, object]:
    return {
        "version": LOADER_CACHE_VERSION,
        "source": str(source),
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "start_date": _normalize_cache_date(start_date),
        "end_date": _normalize_cache_date(end_date),
        "fields": [str(field) for field in (fields or ())],
    }


def _normalize_cache_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _sanitize_cache_segment(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower()
    )
    return cleaned or "unknown"


def make_loader_cache_key(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    fields: Sequence[str] | None = None,
) -> str:
    """Build the stable content-addressed cache key (byte-identical to base.py)."""
    payload = _loader_cache_payload(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def loader_cache_path(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    fields: Sequence[str] | None = None,
    cache_root: str | None = None,
) -> Path:
    """Return the parquet cache path for one payload (byte-identical to base.py)."""
    key = make_loader_cache_key(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
    )
    source_dir = _sanitize_cache_segment(source)
    return loader_cache_root(cache_root) / source_dir / f"{key}.parquet"


def range_is_final(end_date: str) -> bool:
    """Return whether ``end_date`` is settled enough to cache.

    Mirrors ``base.py``: only ranges whose last bar has fully elapsed (strictly
    before today) are cacheable, so a forming day is never pinned and re-served.
    """
    try:
        end = pd.Timestamp(end_date).normalize().date()
    except Exception:  # noqa: BLE001 - unparseable dates are simply not cacheable
        return False
    return end < dt.date.today()


def write_frame(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    fields: Sequence[str] | None,
    frame: pd.DataFrame | None,
    cache_root: str | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> bool:
    """Write one non-empty DataFrame to the cache; ``False`` when not cacheable.

    Skips a disabled cache, an unsettled range, and empty/non-DataFrame results.
    Write failures are swallowed (logged) so a fetch never fails because of the
    cache — the same contract as ``base.py.loader_cache_put``.

    ``extra_metadata`` is merged into the JSON metadata sidecar (e.g. the
    ``adjust`` 口径). The agent's reader ignores unknown sidecar keys, so these
    extras are non-breaking and carry the 口径 alongside the parquet.

    Returns:
        ``True`` when a frame was actually written, ``False`` otherwise.
    """
    if not cache_enabled() or not range_is_final(end_date):
        return False
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return False
    cache_path = loader_cache_path(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        cache_root=cache_root,
    )
    return _write_frame(cache_path, frame, extra_metadata=extra_metadata)


# ---------------------------------------------------------------------------
# Parquet write (duckdb) + JSON metadata sidecar, mirroring base.py.
# ---------------------------------------------------------------------------


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _duckdb_sql_string(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _frame_for_cache(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_frame = frame.copy()
    original_index_names = list(cache_frame.index.names)
    columns_name = cache_frame.columns.name
    index_dtypes = [
        str(cache_frame.index.get_level_values(level).dtype)
        for level in range(cache_frame.index.nlevels)
    ]
    index_columns = _cache_index_columns(cache_frame)
    cache_frame.index = cache_frame.index.set_names(index_columns)
    metadata: dict[str, object] = {
        "version": LOADER_CACHE_VERSION,
        "index_columns": index_columns,
        "index_names": original_index_names,
        "columns_name": None if columns_name is None else str(columns_name),
        "index_dtypes": index_dtypes,
    }
    return cache_frame.reset_index(), metadata


def _cache_index_columns(frame: pd.DataFrame) -> list[str]:
    columns = {str(column) for column in frame.columns}
    used: set[str] = set()
    index_columns: list[str] = []
    for pos, name in enumerate(frame.index.names):
        base = str(name) if name is not None else f"__vibe_loader_index_{pos}__"
        candidate = base
        suffix = 1
        while candidate in columns or candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        index_columns.append(candidate)
        used.add(candidate)
    return index_columns


def _write_frame(
    cache_path: Path,
    frame: pd.DataFrame,
    extra_metadata: Mapping[str, object] | None = None,
) -> bool:
    metadata_path = _metadata_path(cache_path)
    unique = f"{os.getpid()}.{uuid.uuid4().hex}"
    tmp_path = cache_path.with_name(f"{cache_path.name}.{unique}.tmp")
    tmp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.{unique}.tmp")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_frame, metadata = _frame_for_cache(frame)
        if extra_metadata:
            metadata.update(extra_metadata)

        import duckdb

        con = duckdb.connect(database=":memory:")
        try:
            con.register("cache_frame", cache_frame)
            con.execute(f"COPY cache_frame TO {_duckdb_sql_string(tmp_path)} (FORMAT PARQUET)")
        finally:
            con.close()

        tmp_metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, cache_path)
        os.replace(tmp_metadata_path, metadata_path)
        return True
    except Exception as exc:  # noqa: BLE001 - cache write failures must not fail fetches
        logger.warning("qmt bridge cache write failed for %s: %s", cache_path.name, exc)
        for path in (tmp_path, tmp_metadata_path):
            try:
                path.unlink()
            except (FileNotFoundError, OSError):
                pass
        return False
