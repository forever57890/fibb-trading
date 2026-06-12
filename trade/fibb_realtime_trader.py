#!/usr/bin/env python3
"""
FiBB realtime trader — intrabar entry on channel touch via Binance kline WebSocket.

Runs as a long-lived process (not cron). On each 15m kline tick, checks exits and
opens immediately when price touches a leg band (same rules as 15m bar close logic).

  python3 -m fibb_trading.trade.fibb_realtime_trader

Env:
  FIBB_RT_DRY_RUN          default follows FIBB_DRY_RUN
  FIBB_RT_SYMBOL           default FIBB_SYMBOL
  FIBB_RT_INTERVAL         default FIBB_INTERVAL (15m)
  FIBB_RT_WS_URL           optional override WebSocket URL
  FIBB_RT_POLL_SEC         REST poll fallback interval (default 2)
  FIBB_RT_KLINE_REFRESH_SEC refresh history (default 120)

Do NOT run fibb_15m_trader cron on the same account simultaneously.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
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
from fibb_trading.core.data_fetch import interval_to_minutes
from fibb_trading.env_loader import load_fibb_env
from fibb_trading.trade.exchange import ensure_dual_side_position_mode, try_create_trader
from fibb_trading.trade.fibb_intrabar import (
    FormingBar,
    build_curr_prev,
    current_bar_open_time,
    history_ready,
)
from fibb_trading.trade.fibb_live_engine import (
    finalize_closed_bar,
    latest_closed_bar_index,
    process_intrabar_tick,
)
from fibb_trading.trade.live_state import LiveState
from fibb_trading.trade.run_logging import enrich_run_record, format_run_log_block, print_run_output
from fibb_trading.trade.runtime_io import (
    ensure_runtime_dir,
    safe_append_log,
    safe_read_json,
    safe_write_json,
    single_instance_lock,
    state_file_lock,
)

load_fibb_env()

_TRADE_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("FIBB_RT_RUNTIME_DIR", str(_TRADE_ROOT / "runtime_realtime")))
STATE_FILE = RUNTIME_DIR / "fibb_realtime_state.json"
LOCK_FILE = Path(os.getenv("FIBB_TRADE_LOCK", str(_TRADE_ROOT / "runtime" / "fibb_trade.lock")))
LEG_LOCKS_DIR = Path(
    os.getenv("FIBB_LEG_LOCKS_DIR", str(LOCK_FILE.parent / "leg_locks"))
)
LOG_FILE = RUNTIME_DIR / "fibb_realtime_runs.log"

SYMBOL = os.getenv("FIBB_RT_SYMBOL", os.getenv("FIBB_SYMBOL", "BTCUSDT"))
INTERVAL = os.getenv("FIBB_RT_INTERVAL", os.getenv("FIBB_INTERVAL", "15m"))
DRY_RUN = os.getenv("FIBB_RT_DRY_RUN", os.getenv("FIBB_DRY_RUN", "1")) == "1"
POLL_SEC = float(os.getenv("FIBB_RT_POLL_SEC", "2"))
KLINE_REFRESH_SEC = float(os.getenv("FIBB_RT_KLINE_REFRESH_SEC", "120"))
WS_URL = os.getenv(
    "FIBB_RT_WS_URL",
    f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_{INTERVAL}",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> LiveState:
    return LiveState.from_dict(safe_read_json(STATE_FILE))


def save_state(state: LiveState) -> None:
    safe_write_json(STATE_FILE, state.to_dict())


def append_log(record: dict) -> None:
    safe_append_log(LOG_FILE, format_run_log_block(enrich_run_record(record)))


def build_config_snapshot(params=None) -> Dict[str, Any]:
    p = params or configure_strategy_from_env(reload_env=True)
    return {
        "mode": "realtime",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "dry_run": DRY_RUN,
        "ws_url": WS_URL,
        "poll_sec": POLL_SEC,
        "tp_mode": p.tp_mode,
        "params": params_snapshot_dict(p),
        "has_api_keys": bool(os.getenv("bn_api_key") and os.getenv("bn_api_secret")),
    }


def fetch_klines_df(params) -> pd.DataFrame:
    step = interval_to_ms(INTERVAL)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    bars = indicator_history_bars(params.length)
    start_ms = now_ms - step * bars
    klines = fetch_binance_futures_klines(SYMBOL, INTERVAL, start_ms, now_ms)
    return compute_fibb_channels(klines, params)


def wallet_equity_usdt(trader) -> Optional[float]:
    try:
        bal = trader.get_balance_summary("USDT")
        if bal.get("found"):
            return float(bal["wallet_balance"])
    except Exception:
        pass
    return None


class RealtimeEngine:
    def __init__(self) -> None:
        ensure_runtime_dir(RUNTIME_DIR)
        self.params = configure_strategy_from_env(reload_env=True)
        self.state = load_state()
        self.trader = None if DRY_RUN else try_create_trader()
        self.df: Optional[pd.DataFrame] = None
        self.forming: Optional[FormingBar] = None
        self.lock = threading.RLock()
        self.last_kline_refresh = 0.0
        self.interval_minutes = interval_to_minutes(INTERVAL)

        if not DRY_RUN and self.trader is None:
            raise RuntimeError("Missing bn_api_key / bn_api_secret")
        if self.trader is not None:
            ensure_dual_side_position_mode(self.trader)

    def refresh_klines(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_kline_refresh < KLINE_REFRESH_SEC:
            return
        self.df = fetch_klines_df(self.params)
        self.last_kline_refresh = now

    def _log_tick(self, action: str, bar_log: dict, *, extra: Optional[dict] = None) -> None:
        record: Dict[str, Any] = {
            "run_at": utc_now_iso(),
            "action": action,
            "config": build_config_snapshot(self.params),
            "bar": bar_log,
            "state_after": self.state.to_dict(),
        }
        if extra:
            record.update(extra)
        if self.trader is not None:
            try:
                record["account_after"] = self.trader.get_account_snapshot(SYMBOL)
            except Exception as exc:
                record["account_after_error"] = str(exc)
        append_log(record)
        print_run_output(record, include_json=False)

    def handle_kline(self, k: dict) -> None:
        with self.lock:
            with state_file_lock(STATE_FILE, blocking=True):
                self.state = load_state()
                self.refresh_klines()
                if self.df is None or not history_ready(self.df, self.params):
                    return

                if self.forming is None:
                    self.forming = FormingBar.from_kline(k)
                elif self.forming.open_time == pd.Timestamp(int(k["t"]), unit="ms", tz="UTC"):
                    self.forming.merge_kline(k)
                else:
                    self.forming = FormingBar.from_kline(k)

                equity = wallet_equity_usdt(self.trader) if self.trader else None
                persist = lambda: save_state(self.state)

                if k.get("x"):
                    self.refresh_klines(force=True)
                    bar_index = latest_closed_bar_index(self.df)
                    fin_log = finalize_closed_bar(
                        self.df,
                        bar_index,
                        self.state,
                        self.params,
                        trader=self.trader,
                        symbol=SYMBOL,
                        dry_run=DRY_RUN,
                        wallet_equity=equity,
                        persist_state=persist,
                        leg_locks_dir=LEG_LOCKS_DIR,
                    )
                    if not fin_log.get("skipped"):
                        self._log_tick("FINALIZED", fin_log)
                    self.forming = None
                    save_state(self.state)
                    return

                curr, prev = build_curr_prev(self.df, self.forming, self.params)
                tick_log = process_intrabar_tick(
                    self.df,
                    curr,
                    prev,
                    self.state,
                    self.params,
                    trader=self.trader,
                    symbol=SYMBOL,
                    dry_run=DRY_RUN,
                    wallet_equity=equity,
                    persist_state=persist,
                    leg_locks_dir=LEG_LOCKS_DIR,
                )
                save_state(self.state)
                if tick_log.get("had_action"):
                    self._log_tick("INTRABAR", tick_log)

    def poll_mark_fallback(self) -> None:
        """REST fallback when WebSocket is unavailable."""
        if self.trader is None and not DRY_RUN:
            return
        with self.lock:
            self.refresh_klines(force=True)
            if self.df is None or not history_ready(self.df, self.params):
                return
            if self.trader is not None:
                mark = self.trader.get_mark_price(SYMBOL)
            else:
                mark = float(self.df.iloc[-1]["close"])
            bar_open = current_bar_open_time(pd.Timestamp.now(tz="UTC"), self.interval_minutes)
            if self.forming is None or self.forming.open_time != bar_open:
                self.forming = FormingBar.from_price(bar_open, mark)
            else:
                self.forming.update_price(mark)
            k = {
                "t": int(bar_open.timestamp() * 1000),
                "o": self.forming.open,
                "h": self.forming.high,
                "l": self.forming.low,
                "c": self.forming.close,
                "x": False,
            }
        self.handle_kline(k)


def run_websocket(engine: RealtimeEngine) -> None:
    try:
        from websocket import WebSocketApp
    except ImportError as exc:
        raise RuntimeError(
            "websocket-client is required for realtime mode. "
            "pip install websocket-client"
        ) from exc

    def on_message(_ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
            k = payload.get("k")
            if k:
                engine.handle_kline(k)
        except Exception as exc:
            append_log(
                enrich_run_record(
                    {
                        "run_at": utc_now_iso(),
                        "action": "WS_ERROR",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            )

    def on_error(_ws: Any, error: Any) -> None:
        print(f"[realtime] websocket error: {error}", file=sys.stderr)

    def on_close(_ws: Any, *_args: Any) -> None:
        print("[realtime] websocket closed, reconnecting in 5s...", file=sys.stderr)
        time.sleep(5)
        run_websocket(engine)

    ws = WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    print(f"[realtime] connecting {WS_URL} symbol={SYMBOL} dry_run={DRY_RUN}")
    ws.run_forever(ping_interval=20, ping_timeout=10)


def run_poll_loop(engine: RealtimeEngine) -> None:
    print(f"[realtime] poll mode every {POLL_SEC}s symbol={SYMBOL} dry_run={DRY_RUN}")
    while True:
        try:
            engine.poll_mark_fallback()
        except Exception as exc:
            append_log(
                enrich_run_record(
                    {
                        "run_at": utc_now_iso(),
                        "action": "POLL_ERROR",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            )
        time.sleep(POLL_SEC)


def main() -> int:
    use_ws = os.getenv("FIBB_RT_USE_WS", "1") == "1"
    try:
        ensure_runtime_dir(RUNTIME_DIR)
        ensure_runtime_dir(LEG_LOCKS_DIR)
        with single_instance_lock(LOCK_FILE, blocking=False) as acquired:
            if not acquired:
                print("[realtime] another instance is running; exit", file=sys.stderr)
                return 0
            engine = RealtimeEngine()
            append_log(
                enrich_run_record(
                    {
                        "run_at": utc_now_iso(),
                        "action": "STARTED",
                        "config": build_config_snapshot(engine.params),
                    }
                )
            )
            if use_ws:
                try:
                    run_websocket(engine)
                except RuntimeError:
                    print("[realtime] falling back to REST poll", file=sys.stderr)
                    run_poll_loop(engine)
            else:
                run_poll_loop(engine)
        return 0
    except KeyboardInterrupt:
        print("[realtime] stopped")
        return 0
    except Exception as exc:
        append_log(
            enrich_run_record(
                {
                    "run_at": utc_now_iso(),
                    "action": "ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "config": build_config_snapshot(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
