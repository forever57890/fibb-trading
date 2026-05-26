"""Load bn_api_key / bn_api_secret from the project root .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

_PKG_DIR = Path(__file__).resolve().parent


def env_file_paths() -> List[Path]:
    return [_PKG_DIR / ".env"]


def _parse_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_fibb_env(paths: Iterable[Path] | None = None) -> List[Path]:
    """Load environment variables from .env files."""
    candidates = list(paths) if paths is not None else env_file_paths()
    existing = [p for p in candidates if p.exists()]

    try:
        import dotenv

        for path in existing:
            dotenv.load_dotenv(path, override=False)
    except ImportError:
        pass

    for path in existing:
        _parse_env_file(path)

    return existing


def require_binance_keys() -> None:
    load_fibb_env()
    if os.getenv("bn_api_key") and os.getenv("bn_api_secret"):
        return
    tried = ", ".join(str(p) for p in env_file_paths())
    raise ValueError(
        "Set bn_api_key and bn_api_secret in environment or .env. "
        f"Tried: {tried}"
    )
