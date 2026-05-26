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
PKG_PARENT="$(cd "$REPO_ROOT/.." && pwd)"
RUNTIME_DIR="${FIBB_RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
CRON_LOG="${FIBB_CRON_LOG:-$RUNTIME_DIR/fibb_cron.log}"
mkdir -p "$RUNTIME_DIR"

exec >>"$CRON_LOG" 2>&1

echo "[cron] ===== $(date -u +%Y-%m-%dT%H:%M:%SZ) start ====="

export PYTHONPATH="${PKG_PARENT}:${PYTHONPATH:-}"
cd "$PKG_PARENT"

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
    "$PKG_PARENT/.venv/bin/python3" \
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
  echo "[cron] ERROR: no python3 with pandas"
  exit 127
}

echo "[cron] python=$PYTHON dry_run=$FIBB_DRY_RUN"

"$PYTHON" -m fibb_trading.trade.fibb_15m_trader
exit_code=$?
echo "[cron] ===== $(date -u +%Y-%m-%dT%H:%M:%SZ) done exit=$exit_code ====="
exit "$exit_code"
