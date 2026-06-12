"""Runtime directory, JSON state, and text log I/O."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

PathLike = Union[str, Path]


def ensure_runtime_dir(runtime_dir: PathLike) -> Path:
    path = Path(runtime_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _warn(msg: str, *, warn: bool) -> None:
    if warn:
        print(f"[runtime] {msg}", file=sys.stderr)


def safe_read_json(
    path: PathLike,
    *,
    default: Optional[dict] = None,
    warn: bool = True,
) -> dict:
    file_path = Path(path)
    fallback = {} if default is None else dict(default)

    if not file_path.exists():
        _warn(f"missing, using default: {file_path}", warn=warn)
        return fallback

    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        _warn(f"cannot read {file_path}: {exc}", warn=warn)
        return fallback

    if not text:
        return fallback

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        _warn(f"invalid JSON in {file_path}: {exc}", warn=warn)
        return fallback

    if not isinstance(data, dict):
        return fallback

    return data


def safe_write_json(path: PathLike, data: dict) -> None:
    """Atomic write so concurrent readers never see partial JSON."""
    file_path = Path(path)
    ensure_runtime_dir(file_path.parent)
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, file_path)


def _leg_lock_name(bar_time_iso: str, entry_id: str) -> str:
    safe_bar = bar_time_iso.replace(":", "").replace("+", "_")
    safe_id = entry_id.replace(" ", "_")
    return f"{safe_bar}__{safe_id}.lock"


def try_acquire_leg_entry_lock(
    locks_dir: PathLike,
    bar_time_iso: str,
    entry_id: str,
) -> bool:
    """
    Cross-process mutex: one open attempt per (bar, entry_id).

    Uses O_EXCL so parallel cron / websocket workers cannot double-enter.
    """
    root = ensure_runtime_dir(locks_dir)
    path = root / _leg_lock_name(bar_time_iso, entry_id)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{bar_time_iso}\n{entry_id}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_leg_entry_lock(
    locks_dir: PathLike,
    bar_time_iso: str,
    entry_id: str,
) -> None:
    path = Path(locks_dir) / _leg_lock_name(bar_time_iso, entry_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@contextmanager
def state_file_lock(
    state_path: PathLike,
    *,
    blocking: bool = True,
) -> Iterator[bool]:
    """Exclusive flock on the state JSON file for the whole read-modify-write cycle."""
    if fcntl is None:
        yield True
        return

    file_path = Path(state_path)
    ensure_runtime_dir(file_path.parent)
    if not file_path.exists():
        file_path.write_text("{}", encoding="utf-8")

    fd = os.open(str(file_path), os.O_RDWR)
    acquired = False
    try:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def safe_append_log(path: PathLike, text: str) -> None:
    file_path = Path(path)
    ensure_runtime_dir(file_path.parent)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


@contextmanager
def single_instance_lock(
    lock_path: PathLike,
    *,
    blocking: bool = False,
) -> Iterator[bool]:
    """
    Exclusive flock so only one trader process runs at a time.

    Yields True when the lock was acquired, False if another instance holds it
    (non-blocking mode only).
    """
    if fcntl is None:
        yield True
        return

    file_path = Path(lock_path)
    ensure_runtime_dir(file_path.parent)
    fd = os.open(str(file_path), os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    try:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
