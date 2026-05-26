"""僅供 ``python -m backtest.*``；cron 請用 ``python -m fibb_trading.*`` + PYTHONPATH。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def ensure() -> None:
    parent = str(_REPO_ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


ensure()
