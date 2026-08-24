"""Enable ``python -m qmt_bridge``."""

from __future__ import annotations

import sys

from qmt_bridge.cli import main

if __name__ == "__main__":
    sys.exit(main())
