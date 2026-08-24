"""sys.path setup for qmt_bridge tests.

Mirrors ``agent/tests/conftest.py``'s import-time path insertion (but without
the home-redirect sandbox, which the agent suite needs for its config/ledger
state and the bridge suite does not): the repo root is added so ``qmt_bridge``
imports, and ``agent/`` is added so the cache-compatibility test can import
``backtest.loaders.base``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_DIR = _REPO_ROOT / "agent"

for _path in (str(_REPO_ROOT), str(_AGENT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
