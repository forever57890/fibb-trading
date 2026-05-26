"""
One 15m bar step for live trading — same order as run_fibb_backtest:
  exits -> entries -> arm deferred channel stops.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from fibb_trading.core.fibb_config import FibbParams
from fibb_trading.core.fibb_logic import (
    OpenLeg,
    analyze_entry_legs,
    analyze_open_leg_exits,
    arm_deferred_channel_stops,
    bracket_prices,
    detect_entry_signals,
    refresh_channel_take_profits,
    resolve_entry_qty,
    resolve_take_profit,
    try_exit_leg,
    uses_deferred_channel_sl,
)
from fibb_trading.trade.exchange import (
    BinanceFuturesTrader,
    cancel_leg_tp,
    market_close_leg_qty,
    open_leg_with_tp,
    replace_leg_tp,
)
from fibb_trading.trade.live_state import (
    LiveState,
    append_closed_trade,
    get_tp_algo_id,
    leg_from_dict,
    open_legs_objects,
    save_open_legs,
)


def _closed_stub(leg: OpenLeg, ts: pd.Timestamp, exit_price: float, reason: str, fee_rate: float) -> dict:
    from fibb_trading.core.fibb_logic import leg_pnl

    gross, fee, net = leg_pnl(leg, exit_price, fee_rate)
    notional = leg.qty * leg.entry_price
    return {
        "entry_id": leg.entry_id,
        "side": leg.side,
        "qty": leg.qty,
        "entry_time": pd.Timestamp(leg.entry_time).isoformat(),
        "exit_time": pd.Timestamp(ts).isoformat(),
        "entry_price": leg.entry_price,
        "exit_price": exit_price,
        "exit_reason": reason,
        "gross_pnl": gross,
        "fee": fee,
        "net_pnl": net,
        "win": net > 0,
    }


def process_bar(
    df: pd.DataFrame,
    bar_index: int,
    state: LiveState,
    params: FibbParams,
    *,
    trader: Optional[BinanceFuturesTrader] = None,
    symbol: str = "BTCUSDT",
    dry_run: bool = False,
    wallet_equity: Optional[float] = None,
    reprice_tp_to_basis: bool = True,
) -> Dict[str, Any]:
    """
    Process a single closed 15m bar. Mutates *state* and returns action log.
    """
    if bar_index < 1:
        return {"skipped": True, "reason": "warmup"}

    curr = df.iloc[bar_index]
    prev = df.iloc[bar_index - 1]
    ts = curr["open_time"]
    ts_iso = pd.Timestamp(ts).isoformat()

    if state.last_bar_time == ts_iso:
        return {"skipped": True, "reason": "already_processed", "bar_time": ts_iso}

    open_legs = open_legs_objects(state)

    bar_high = float(curr["high"])
    bar_low = float(curr["low"])
    close_price = float(curr["close"])

    log: Dict[str, Any] = {
        "bar_time": ts_iso,
        "close": close_price,
        "exits": [],
        "entries": [],
        "armed_stops": [],
        "tp_reprices": [],
        "hold_diagnostics": [],
    }

    refresh_channel_take_profits(open_legs, curr, params)

    # --- exits ---
    for entry_id in list(open_legs.keys()):
        leg = open_legs[entry_id]
        exit_price, reason = try_exit_leg(leg, bar_high, bar_low)
        if exit_price is None:
            continue

        exec_result: Dict[str, Any] = {"entry_id": entry_id, "reason": reason, "exit_price": exit_price}

        # Cancel the TP algo order before market-closing the leg
        tp_algo_id = get_tp_algo_id(state, entry_id)
        if trader is not None and tp_algo_id is not None:
            exec_result["cancel_tp"] = cancel_leg_tp(trader, symbol, tp_algo_id)

        if trader is not None:
            exec_result["exchange"] = market_close_leg_qty(
                trader, symbol, leg.side, leg.qty, dry_run=dry_run
            )
        elif dry_run:
            exec_result["exchange"] = {"status": "DRY_RUN_CLOSE"}

        closed = _closed_stub(leg, ts, exit_price, reason, params.fee_rate)
        append_closed_trade(state, closed)
        log["exits"].append({**exec_result, "trade": closed})
        del open_legs[entry_id]

    tp_algo_ids_updates: Dict[str, Any] = {}
    log["reprice_tp_to_basis"] = reprice_tp_to_basis
    basis_value = curr.get("basis")
    if reprice_tp_to_basis and not pd.isna(basis_value):
        basis_tp = float(basis_value)
        for entry_id, leg in open_legs.items():
            old_tp = leg.take_profit_price
            old_tp_algo_id = get_tp_algo_id(state, entry_id)
            leg.take_profit_price = basis_tp
            leg.take_profit_band = "basis"
            reprice_log: Dict[str, Any] = {
                "entry_id": entry_id,
                "old_tp": old_tp,
                "new_tp": basis_tp,
                "old_tp_algo_id": old_tp_algo_id,
            }
            if trader is not None:
                replace_result = replace_leg_tp(
                    trader,
                    symbol,
                    leg.side,
                    leg.qty,
                    basis_tp,
                    old_tp_algo_id=old_tp_algo_id,
                    dry_run=dry_run,
                )
                reprice_log["exchange"] = replace_result
                new_tp_algo_id = replace_result.get("tp_algo_id")
                if new_tp_algo_id is not None:
                    tp_algo_ids_updates[entry_id] = new_tp_algo_id
                elif old_tp_algo_id is not None:
                    tp_algo_ids_updates[entry_id] = old_tp_algo_id
            elif dry_run:
                reprice_log["exchange"] = {
                    "status": "DRY_RUN_REPRICE_TP",
                    "take_profit_price": basis_tp,
                }
            log["tp_reprices"].append(reprice_log)
    elif not reprice_tp_to_basis and open_legs:
        log["tp_reprice_note"] = "FIBB_REPRICE_TP_TO_BASIS=0，維持開倉時固定 % TP，不重掛中線"

    for entry_id, leg in open_legs.items():
        log["hold_diagnostics"].append(analyze_open_leg_exits(leg, bar_high, bar_low))

    # --- entries (live: no backtest end-date gate) ---
    entry_diagnostics = analyze_entry_legs(curr, prev, set(open_legs.keys()))
    diag_by_id = {d["entry_id"]: d for d in entry_diagnostics}
    if params.initial_capital <= 0 or params.leverage <= 0:
        equity = float(wallet_equity or 1e12)
    elif wallet_equity is not None:
        equity = float(wallet_equity)
    else:
        equity = params.initial_capital + state.realized_pnl

    new_tp_algo_ids: Dict[str, Any] = {}

    if len(open_legs) < params.max_open_legs:
        signals = detect_entry_signals(curr, prev, set(open_legs.keys()))
        for entry_id, side, qty, band in signals:
            if len(open_legs) >= params.max_open_legs:
                if entry_id in diag_by_id:
                    diag_by_id[entry_id]["blocked_reason"] = (
                        f"已達 max_open_legs={params.max_open_legs}"
                    )
                break
            qty = resolve_entry_qty(qty, close_price, equity, params)
            if qty <= 0:
                reason = "insufficient_equity"
                if entry_id in diag_by_id:
                    diag_by_id[entry_id]["blocked_reason"] = (
                        f"資金不足（equity={equity:.2f} leverage={params.leverage}）"
                    )
                log["entries"].append(
                    {"entry_id": entry_id, "skipped": True, "reason": reason}
                )
                continue

            tp, tp_band = resolve_take_profit(entry_id, side, close_price, curr, params)
            if uses_deferred_channel_sl(entry_id, params):
                sl = None
                sl_channel = False
            elif not params.use_deferred_channel_sl:
                _, sl = bracket_prices(side, close_price, params)
                sl_channel = False
            else:
                sl = None
                sl_channel = False

            exec_result: Dict[str, Any] = {"entry_id": entry_id, "qty": qty, "side": side}
            if trader is not None:
                open_result = open_leg_with_tp(
                    trader, symbol, side, qty, tp, dry_run=dry_run
                )
                exec_result["exchange"] = open_result
                exec_result["tp_algo_id"] = open_result.get("tp_algo_id")
                new_tp_algo_ids[entry_id] = open_result.get("tp_algo_id")
            elif dry_run:
                exec_result["exchange"] = {"status": "DRY_RUN_OPEN"}
                exec_result["tp_algo_id"] = None

            open_legs[entry_id] = OpenLeg(
                entry_id=entry_id,
                side=side,
                qty=qty,
                entry_time=ts,
                entry_price=close_price,
                take_profit_price=tp,
                stop_loss_price=sl,
                band=band,
                take_profit_band=tp_band,
                sl_use_channel=sl_channel,
            )
            log["entries"].append(exec_result)
            if entry_id in diag_by_id:
                diag_by_id[entry_id]["status"] = "opened"
                diag_by_id[entry_id]["reason"] = "已開倉"
    elif entry_diagnostics:
        for d in entry_diagnostics:
            if d.get("status") == "touch_signal":
                d["blocked_reason"] = (
                    f"已達 max_open_legs={params.max_open_legs}，無法再開新 leg"
                )

    log["entry_diagnostics"] = entry_diagnostics

    # --- arm deferred stops (after entries, same as backtest) ---
    before_arm = {k: v.stop_loss_price for k, v in open_legs.items()}
    arm_deferred_channel_stops(open_legs, curr, prev, params)
    for entry_id, leg in open_legs.items():
        if before_arm.get(entry_id) is None and leg.stop_loss_price is not None:
            log["armed_stops"].append(
                {
                    "entry_id": entry_id,
                    "stop_loss_price": leg.stop_loss_price,
                    "sl_use_channel": leg.sl_use_channel,
                }
            )

    combined_tp_algo_ids = {**tp_algo_ids_updates, **new_tp_algo_ids}
    save_open_legs(
        state,
        open_legs,
        tp_algo_ids=combined_tp_algo_ids if combined_tp_algo_ids else None,
    )
    state.last_bar_time = ts_iso
    log["open_legs_after"] = list(state.open_legs.keys())
    return log


def latest_closed_bar_index(klines: pd.DataFrame) -> int:
    """Index of the most recently *closed* 15m candle."""
    if len(klines) < 2:
        return len(klines) - 1
    now = pd.Timestamp.now(tz="UTC")
    last = klines.iloc[-1]
    if pd.Timestamp(last["close_time"]) > now:
        return len(klines) - 2
    return len(klines) - 1
