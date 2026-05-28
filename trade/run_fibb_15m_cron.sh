#!/usr/bin/env bash
# Run FiBB 15m trader once (crontab at :01, :16, :31, :46 UTC recommended).
#
#   TZ=UTC
#   1,16,31,46 * * * * /bin/bash /path/to/fibb_trading/trade/run_fibb_15m_cron.sh
#
# Env:
#   FIBB_PYTHON=/path/to/python3
#   FIBB_DRY_RUN=0          (set in script after .env for live)
#   FIBB_ENABLE_HEDGE_MODE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${FIBB_RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
CRON_LOG="${FIBB_CRON_LOG:-$RUNTIME_DIR/fibb_cron.log}"
mkdir -p "$RUNTIME_DIR"

exec >>"$CRON_LOG" 2>&1

echo "[cron] ===== $(date -u +%Y-%m-%dT%H:%M:%SZ) start ====="

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export FIBB_DRY_RUN="${FIBB_DRY_RUN:-0}"
export FIBB_ENABLE_HEDGE_MODE="${FIBB_ENABLE_HEDGE_MODE:-1}"

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

resolve_module() {
  # Try common layouts:
  # 1) package layout: <root>/fibb_trading/trade/fibb_15m_trader.py
  # 2) flat layout:    <root>/trade/fibb_15m_trader.py
  local roots=(
    "$REPO_ROOT"
    "$REPO_ROOT/.."
    "$(pwd)"
  )
  local root
  for root in "${roots[@]}"; do
    [ -d "$root" ] || continue
    if [ -f "$root/fibb_trading/trade/fibb_15m_trader.py" ]; then
      echo "$root|fibb_trading.trade.fibb_15m_trader|package"
      return 0
    fi
    if [ -f "$root/trade/fibb_15m_trader.py" ]; then
      echo "$root|trade.fibb_15m_trader|flat"
      return 0
    fi
  done
  return 1
}

PYTHON="$(resolve_python)" || {
  echo "[cron] ERROR: no python3 with pandas"
  exit 127
}

module_info="$(resolve_module)" || {
  echo "[cron] ERROR: cannot locate fibb_15m_trader module root"
  exit 127
}
MODULE_ROOT="${module_info%%|*}"
rest="${module_info#*|}"
ENTRY_MODULE="${rest%%|*}"
LAYOUT_MODE="${module_info##*|}"
export MODULE_ROOT

export PYTHONPATH="${MODULE_ROOT}:${PYTHONPATH:-}"
cd "$MODULE_ROOT"

echo "[cron] python=$PYTHON dry_run=$FIBB_DRY_RUN"
echo "[cron] module_root=$MODULE_ROOT entry_module=$ENTRY_MODULE layout=$LAYOUT_MODE"

if [ "$LAYOUT_MODE" = "flat" ]; then
  # Flat layout has top-level dirs (core/, trade/, ...), but code imports fibb_trading.*.
  # Inject a runtime namespace package alias so imports resolve without install step.
  "$PYTHON" - <<PYCODE
import os
import runpy
import sys
import types

root = os.environ["MODULE_ROOT"]
pkg = types.ModuleType("fibb_trading")
pkg.__path__ = [root]
sys.modules["fibb_trading"] = pkg
runpy.run_module("trade.fibb_15m_trader", run_name="__main__")
PYCODE
else
  "$PYTHON" -m "$ENTRY_MODULE"
fi
exit_code=$?
echo "[cron] ===== $(date -u +%Y-%m-%dT%H:%M:%SZ) done exit=$exit_code ====="
exit "$exit_code"
