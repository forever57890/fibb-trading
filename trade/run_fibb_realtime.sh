#!/usr/bin/env bash
# FiBB realtime trader (long-running WebSocket process).
#
# Run under systemd, screen, or tmux — NOT crontab.
#
#   screen -S fibb-rt
#   bash /path/to/fibb_trading/trade/run_fibb_realtime.sh
#
# Disable fibb_15m cron on the same account before starting this.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${FIBB_RT_RUNTIME_DIR:-$SCRIPT_DIR/runtime_realtime}"
CRON_LOG="${FIBB_RT_LOG:-$RUNTIME_DIR/fibb_realtime_daemon.log}"
mkdir -p "$RUNTIME_DIR"

exec >>"$CRON_LOG" 2>&1

echo "[realtime] ===== $(date -u +%Y-%m-%dT%H:%M:%SZ) start ====="

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export FIBB_RT_DRY_RUN="${FIBB_RT_DRY_RUN:-${FIBB_DRY_RUN:-0}}"
export FIBB_RT_USE_WS="${FIBB_RT_USE_WS:-1}"

resolve_python() {
  if [ -n "${FIBB_PYTHON:-}" ] && [ -x "$FIBB_PYTHON" ]; then
    echo "$FIBB_PYTHON"
    return 0
  fi
  for candidate in \
    "$REPO_ROOT/.venv/bin/python3" \
    "$REPO_ROOT/../.venv/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"
  do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    if "$candidate" -c "import pandas" 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(resolve_python)" || {
  echo "[realtime] ERROR: no python3 with pandas"
  exit 127
}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

echo "[realtime] python=$PYTHON dry_run=$FIBB_RT_DRY_RUN"

"$PYTHON" - <<'PYCODE'
import os
import runpy
import sys
import types

root = os.getcwd()
pkg = types.ModuleType("fibb_trading")
pkg.__path__ = [root]
sys.modules["fibb_trading"] = pkg
runpy.run_module("trade.fibb_realtime_trader", run_name="__main__")
PYCODE
