#!/usr/bin/env python3
"""
FiBB 15m live trader — runs once per invocation (cron every 15 minutes).

Strategy: deferred channel SL + fixed %% take profit (current Python default).

  python3 -m fibb_trading.trade.fibb_15m_trader
  FIBB_DRY_RUN=1 python3 -m fibb_trading.trade.fibb_15m_trader
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from fibb_trading.core.data_fetch import fetch_binance_futures_klines, interval_to_ms
from fibb_trading.core.fibb_env import (
    configure_strategy_from_env,
    indicator_history_bars,
    params_snapshot_dict,
)
from fibb_trading.core.fibb_logic import compute_fibb_channels
from fibb_trading.env_loader import load_fibb_env
from fibb_trading.trade.exchange import (
    ensure_dual_side_position_mode,
    try_create_trader,
)
from fibb_trading.trade.fibb_live_engine import latest_closed_bar_index, process_bar
from fibb_trading.trade.live_state import LiveState
from fibb_trading.trade.run_logging import (
    enrich_run_record,
    format_run_log_block,
    print_run_output,
)
from fibb_trading.trade.runtime_io import (
    ensure_runtime_dir,
    safe_append_log,
    safe_read_json,
    safe_write_json,
    single_instance_lock,
)

load_fibb_env()
_STRATEGY_PARAMS = configure_strategy_from_env(reload_env=False)

_TRADE_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("FIBB_RUNTIME_DIR", str(_TRADE_ROOT / "runtime")))
STATE_FILE = RUNTIME_DIR / "fibb_15m_state.json"
LOCK_FILE = Path(os.getenv("FIBB_TRADE_LOCK", str(RUNTIME_DIR / "fibb_trade.lock")))
LOG_FILE = RUNTIME_DIR / "fibb_15m_runs.log"

SYMBOL = os.getenv("FIBB_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("FIBB_INTERVAL", "15m")
DRY_RUN = os.getenv("FIBB_DRY_RUN", "1") == "1"
IGNORE_STATE = os.getenv("FIBB_IGNORE_STATE", "0") == "1"
FORCE_BAR_TIME = os.getenv("FIBB_FORCE_BAR_TIME")  # ISO open_time for replay
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_config_snapshot(params=None) -> Dict[str, Any]:
    p = params or configure_strategy_from_env(reload_env=True)
    return {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "dry_run": DRY_RUN,
        "tp_mode": p.tp_mode,
        "params": params_snapshot_dict(p),
        "has_api_keys": bool(os.getenv("bn_api_key") and os.getenv("bn_api_secret")),
    }


def load_state() -> LiveState:
    return LiveState.from_dict(safe_read_json(STATE_FILE))


def save_state(state: LiveState) -> None:
    safe_write_json(STATE_FILE, state.to_dict())


def append_log(record: dict) -> None:
    """Expect *record* already passed through enrich_run_record."""
    safe_append_log(LOG_FILE, format_run_log_block(record))


def fetch_klines(params) -> pd.DataFrame:
    step = interval_to_ms(INTERVAL)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    bars = indicator_history_bars(params.length)
    start_ms = now_ms - step * bars
    klines = fetch_binance_futures_klines(SYMBOL, INTERVAL, start_ms, now_ms)
    return klines


def wallet_equity_usdt(trader) -> Optional[float]:
    try:
        bal = trader.get_balance_summary("USDT")
        if bal.get("found"):
            return float(bal["wallet_balance"])
    except Exception:
        pass
    return None


def run_once() -> Dict[str, Any]:
    ensure_runtime_dir(RUNTIME_DIR)
    params = configure_strategy_from_env(reload_env=True)
    state = load_state()
    record: Dict[str, Any] = {
        "run_at": utc_now_iso(),
        "config": build_config_snapshot(params),
        "state_before": state.to_dict(),
    }

    trader = None if DRY_RUN else try_create_trader()
    if not DRY_RUN and trader is None:
        record["action"] = "ERROR"
        record["error"] = "Missing bn_api_key / bn_api_secret"
        record = enrich_run_record(record)
        append_log(record)
        return record

    if trader is not None:
        record["hedge_mode"] = ensure_dual_side_position_mode(trader)
        try:
            record["account_before"] = trader.get_account_snapshot(SYMBOL)
        except Exception as exc:
            record["account_before_error"] = str(exc)

    klines = fetch_klines(params)
    df = compute_fibb_channels(klines, params)

    if FORCE_BAR_TIME:
        matches = df.index[df["open_time"] == pd.Timestamp(FORCE_BAR_TIME, tz="UTC")].tolist()
        if not matches:
            raise ValueError(f"FIBB_FORCE_BAR_TIME not in klines: {FORCE_BAR_TIME}")
        bar_index = matches[0]
        if IGNORE_STATE:
            state.last_bar_time = None
    else:
        bar_index = latest_closed_bar_index(df)

    equity = wallet_equity_usdt(trader) if trader else None
    bar_log = process_bar(
        df,
        bar_index,
        state,
        params,
        trader=trader,
        symbol=SYMBOL,
        dry_run=DRY_RUN,
        wallet_equity=equity,
        persist_state=lambda: save_state(state),
    )
    record["bar"] = bar_log
    record["bar_index"] = bar_index
    record["klines_count"] = len(df)
    record["channels_at_bar"] = {
        k: float(df.iloc[bar_index][k])
        for k in ("basis", "top1", "top2", "top3", "bott1", "bott2", "bott3")
        if k in df.columns and pd.notna(df.iloc[bar_index][k])
    }

    save_state(state)
    record["state_after"] = state.to_dict()
    record["action"] = "SKIPPED" if bar_log.get("skipped") else "PROCESSED"

    if trader is not None:
        try:
            record["account_after"] = trader.get_account_snapshot(SYMBOL)
        except Exception as exc:
            record["account_after_error"] = str(exc)

    record = enrich_run_record(record)
    append_log(record)
    return record


def main() -> int:
    try:
        ensure_runtime_dir(RUNTIME_DIR)
        with single_instance_lock(LOCK_FILE, blocking=False) as acquired:
            if not acquired:
                record = enrich_run_record(
                    {
                        "run_at": utc_now_iso(),
                        "action": "SKIPPED_ALREADY_RUNNING",
                        "config": build_config_snapshot(),
                        "error_hint": (
                            "另一個 FiBB trader 正在執行（flock）；"
                            "略過本次以避免重複下單"
                        ),
                    }
                )
                append_log(record)
                print_run_output(record)
                return 0
            record = run_once()
        print_run_output(record)
        if record.get("action") == "ERROR":
            return 1
        return 0
    except Exception as exc:
        err = enrich_run_record(
            {
                "run_at": utc_now_iso(),
                "action": "ERROR",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "config": build_config_snapshot(),
                "error_hint": (
                    "若為 minQty：多為 tp_mode=2 重掛 TP 或平倉時 leg.qty 過小；"
                    "請確認伺服器已部署 prepare_order_qty 修復"
                ),
            }
        )
        append_log(err)
        print_run_output(err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
