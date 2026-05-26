from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_BACKTEST_ROOT = Path(__file__).resolve().parent
TEST_DATA_DIR = _BACKTEST_ROOT / "test_data"


def ensure_test_data_dir(path: PathLike = TEST_DATA_DIR) -> Path:
    target = Path(path)
    dir_path = target.parent if target.suffix else target
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
