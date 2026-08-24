"""QMT Bridge — a standalone, read-only market-data bridge for miniQMT.

The bridge is a Windows-hosted HTTP service that is fully decoupled from the
FastAPI agent process. It imports **only** ``xtquant.xtdata`` (never
``xttrader``), exposes a read-only HTTP surface, and writes adjusted OHLCV
plus suspension / price-limit / dividend metadata into the shared loader cache
at ``~/.vibe-trading/cache/loaders/miniqmt/`` so the ``miniqmt`` loader
(D-02) can consume it.

The service refuses to start if its capability manifest declares any write
capability, and the HTTP surface structurally rejects every non-GET method.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
