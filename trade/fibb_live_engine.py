"""
One 15m bar step for live trading — same order as run_fibb_backtest:
  exits -> entries -> arm deferred channel stops.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from fibb_trading.core.fibb_config import (
    FibbParams,
    tp_mode_channel_tracks_bar,
    tp_mode_uses_channel,
    tp_mode_label,
)
from fibb_trading.core.fibb_env import first_tradable_bar_index, indicator_history_bars
from fibb_trading.core.fibb_logic import (
    OpenLeg,
    analyze_entry_legs,
    analyze_open_leg_exits,
    arm_deferred_channel_stops,
    bracket_prices,
    detect_entry_signals,
    refresh_channel_take_profits,
    refresh_reprice_tp_to_basis,
    resolve_entry_qty,
    resolve_entry_take_profit,
    resolve_regime_controls,
    should_block_entry_by_regime,
    take_profit_price_pct,
    try_exit_leg,
    try_time_stop_exit,
    uses_deferred_channel_sl,
)
from fibb_trading.trade.exchange import (
    BinanceFuturesTrader,
    cancel_leg_tp,
    close_leg_with_ioc,
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


def _sync_exchange_take_profits(
    *,
    open_legs: Dict[str, OpenLeg],
    state: LiveState,
    trader: Optional[BinanceFuturesTrader],
    symbol: str,
    dry_run: bool,
    log: Dict[str, Any],
    tp_algo_ids_updates: Dict[str, Any],
    reprice_kind: str,
    old_tp_by_id: Optional[Dict[str, float]] = None,
) -> None:
    """Cancel/replace TP algo orders to match leg.take_profit_price (basis or channel)."""
    for entry_id, leg in open_legs.items():
        new_tp = leg.take_profit_price
        old_tp_algo_id = get_tp_algo_id(state, entry_id)
        reprice_log: Dict[str, Any] = {
            "entry_id": entry_id,
            "kind": reprice_kind,
            "old_tp": (old_tp_by_id or {}).get(entry_id, new_tp),
            "new_tp": new_tp,
            "old_tp_algo_id": old_tp_algo_id,
            "take_profit_band": leg.take_profit_band,
        }
        if trader is not None:
            replace_result = replace_leg_tp(
                trader,
                symbol,
                leg.side,
                leg.qty,
                new_tp,
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
                "take_profit_price": new_tp,
            }
        log["tp_reprices"].append(reprice_log)


def _execute_leg_exit(
    *,
    entry_id: str,
    leg: OpenLeg,
    ts: pd.Timestamp,
    exit_price: float,
    reason: str,
    state: LiveState,
    trader: Optional[BinanceFuturesTrader],
    symbol: str,
    dry_run: bool,
    fee_rate: float,
    log: Dict[str, Any],
    open_legs: Dict[str, OpenLeg],
) -> None:
    exec_result: Dict[str, Any] = {
        "entry_id": entry_id,
        "reason": reason,
        "exit_price": exit_price,
    }

    tp_algo_id = get_tp_algo_id(state, entry_id)
    if trader is not None and tp_algo_id is not None:
        exec_result["cancel_tp"] = cancel_leg_tp(trader, symbol, tp_algo_id)

    if trader is not None:
        exec_result["exchange"] = close_leg_with_ioc(
            trader, symbol, leg.side, leg.qty, dry_run=dry_run
        )
    elif dry_run:
        exec_result["exchange"] = {"status": "DRY_RUN_CLOSE"}

    closed = _closed_stub(leg, ts, exit_price, reason, fee_rate)
    append_closed_trade(state, closed)
    log["exits"].append({**exec_result, "trade": closed})
    del open_legs[entry_id]


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
) -> Dict[str, Any]:
    """
    Process a single closed 15m bar. Mutates *state* and returns action log.
    """
    if bar_index < 1:
        return {
            "skipped": True,
            "reason": "insufficient_history",
            "bar_index": bar_index,
            "klines_count": len(df),
            "need_at_least": 2,
        }

    curr = df.iloc[bar_index]
    prev = df.iloc[bar_index - 1]

    channel_cols = ("basis", "top1", "bott1")
    missing = [
        c
        for c in channel_cols
        if c not in df.columns or pd.isna(curr.get(c)) or pd.isna(prev.get(c))
    ]
    if missing or bar_index < first_tradable_bar_index(params.length):
        return {
            "skipped": True,
            "reason": "insufficient_history",
            "bar_index": bar_index,
            "klines_count": len(df),
            "history_bars_requested": indicator_history_bars(params.length),
            "need_bar_index_at_least": first_tradable_bar_index(params.length),
            "missing_channels": missing,
        }
    ts = curr["open_time"]
    ts_iso = pd.Timestamp(ts).isoformat()

    if state.last_bar_time == ts_iso:
        return {"skipped": True, "reason": "already_processed", "bar_time": ts_iso}

    open_legs = open_legs_objects(state)

    bar_high = float(curr["high"])
    bar_low = float(curr["low"])
    close_price = float(curr["close"])
    controls = resolve_regime_controls(curr, params)

    log: Dict[str, Any] = {
        "bar_time": ts_iso,
        "close": close_price,
        "exits": [],
        "entries": [],
        "armed_stops": [],
        "tp_reprices": [],
        "hold_diagnostics": [],
        "regime": controls,
        "trade_sides": params.trade_sides,
    }

    if trader is not None and not dry_run:
        min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
        for side in ("LONG", "SHORT"):
            exchange_amt = abs(trader.get_position_amount(symbol, side))
            virtual_amt = sum(
                leg.qty for leg in open_legs.values() if leg.side == side
            )
            if exchange_amt > virtual_amt + min_qty * 2:
                log.setdefault("position_mismatch", []).append(
                    {
                        "side": side,
                        "exchange_amt": exchange_amt,
                        "virtual_amt": virtual_amt,
                        "excess": round(exchange_amt - virtual_amt, 8),
                        "open_leg_ids": [
                            leg.entry_id
                            for leg in open_legs.values()
                            if leg.side == side
                        ],
                    }
                )

    old_channel_tps: Dict[str, float] = {}
    if tp_mode_channel_tracks_bar(params):
        old_channel_tps = {eid: leg.take_profit_price for eid, leg in open_legs.items()}
        refresh_channel_take_profits(open_legs, curr, params)

    # --- exits (intrabar TP/SL, then bar-close time stop) ---
    for entry_id in list(open_legs.keys()):
        leg = open_legs[entry_id]
        exit_price, reason = try_exit_leg(leg, bar_high, bar_low)
        if exit_price is None:
            exit_price, reason = try_time_stop_exit(
                leg, ts, close_price, controls["max_holding_hours"]
            )
        if exit_price is None:
            continue
        _execute_leg_exit(
            entry_id=entry_id,
            leg=leg,
            ts=ts,
            exit_price=exit_price,
            reason=reason,
            state=state,
            trader=trader,
            symbol=symbol,
            dry_run=dry_run,
            fee_rate=params.fee_rate,
            log=log,
            open_legs=open_legs,
        )

    tp_algo_ids_updates: Dict[str, Any] = {}
    log["tp_mode"] = params.tp_mode
    log["tp_mode_label"] = tp_mode_label(params.tp_mode)
    if params.tp_mode == 1:
        old_tps = {eid: leg.take_profit_price for eid, leg in open_legs.items()}
        refresh_reprice_tp_to_basis(open_legs, curr, params)
        _sync_exchange_take_profits(
            open_legs=open_legs,
            state=state,
            trader=trader,
            symbol=symbol,
            dry_run=dry_run,
            log=log,
            tp_algo_ids_updates=tp_algo_ids_updates,
            reprice_kind="basis",
            old_tp_by_id=old_tps,
        )
    elif tp_mode_channel_tracks_bar(params) and open_legs:
        _sync_exchange_take_profits(
            open_legs=open_legs,
            state=state,
            trader=trader,
            symbol=symbol,
            dry_run=dry_run,
            log=log,
            tp_algo_ids_updates=tp_algo_ids_updates,
            reprice_kind="channel",
            old_tp_by_id=old_channel_tps,
        )
    elif params.tp_mode == 3 and open_legs:
        log["tp_reprice_note"] = "FIBB_TP_MODE=3，通道止盈開倉時鎖定，不重掛"
    elif params.tp_mode == 0 and open_legs:
        log["tp_reprice_note"] = "FIBB_TP_MODE=0，維持開倉時固定 % TP，不重掛"

    for entry_id, leg in open_legs.items():
        log["hold_diagnostics"].append(analyze_open_leg_exits(leg, bar_high, bar_low))

    # --- entries (live: no backtest end-date gate) ---
    entry_diagnostics = analyze_entry_legs(
        curr, prev, set(open_legs.keys()), params
    )
    diag_by_id = {d["entry_id"]: d for d in entry_diagnostics}
    if params.initial_capital <= 0 or params.leverage <= 0:
        equity = float(wallet_equity or 1e12)
    elif wallet_equity is not None:
        equity = float(wallet_equity)
    else:
        equity = params.initial_capital + state.realized_pnl

    new_tp_algo_ids: Dict[str, Any] = {}

    if len(open_legs) < params.max_open_legs:
        signals = detect_entry_signals(curr, prev, set(open_legs.keys()), params)
        for entry_id, side, qty, band in signals:
            if len(open_legs) >= params.max_open_legs:
                if entry_id in diag_by_id:
                    diag_by_id[entry_id]["blocked_reason"] = (
                        f"已達 max_open_legs={params.max_open_legs}"
                    )
                break
            requested_qty = qty
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
            if trader is not None:
                min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
                stepped_qty = trader.quantize_qty_down(symbol, qty)
                order_qty = trader.prepare_order_qty(symbol, qty)
                if order_qty <= 0:
                    reason = "below_min_qty"
                    max_notional = equity * params.leverage
                    detail = (
                        f"下單量 {qty}（策略 {requested_qty}）經 step 取整為 {stepped_qty}，"
                        f"低於 minQty {min_qty}；"
                        f"equity={equity:.2f} leverage={params.leverage} "
                        f"close={close_price:.2f} max_notional≈{max_notional:.2f}"
                    )
                    if entry_id in diag_by_id:
                        diag_by_id[entry_id]["blocked_reason"] = detail
                    log["entries"].append(
                        {
                            "entry_id": entry_id,
                            "skipped": True,
                            "reason": reason,
                            "requested_qty": requested_qty,
                            "resolved_qty": qty,
                            "stepped_qty": stepped_qty,
                            "min_qty": min_qty,
                            "equity": equity,
                            "close": close_price,
                        }
                    )
                    continue
                qty = order_qty
            blocked, block_reason = should_block_entry_by_regime(
                entry_id, side, curr, params
            )
            if blocked:
                if entry_id in diag_by_id:
                    diag_by_id[entry_id]["blocked_reason"] = block_reason
                log["entries"].append(
                    {
                        "entry_id": entry_id,
                        "skipped": True,
                        "reason": block_reason,
                    }
                )
                continue

            entry_tp, tp_band = resolve_entry_take_profit(
                entry_id,
                side,
                close_price,
                curr,
                params,
                tp_pct_override=controls["tp_pct"],
            )
            channel_tp_at_open = entry_tp if tp_mode_uses_channel(params) else None
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
            fill_entry = close_price
            tp = None
            if trader is not None:
                open_result = open_leg_with_tp(
                    trader,
                    symbol,
                    side,
                    qty,
                    controls["tp_pct"],
                    signal_close=close_price,
                    take_profit_price=channel_tp_at_open,
                    dry_run=dry_run,
                )
                if open_result.get("status") == "SKIPPED_BELOW_MIN_QTY":
                    detail = (
                        f"交易所 minQty {open_result.get('min_qty')}；"
                        f"requested={open_result.get('requested_qty')} "
                        f"stepped={open_result.get('stepped_qty')}"
                    )
                    if entry_id in diag_by_id:
                        diag_by_id[entry_id]["blocked_reason"] = detail
                    log["entries"].append(
                        {
                            "entry_id": entry_id,
                            "skipped": True,
                            "reason": "below_min_qty",
                            "exchange": open_result,
                        }
                    )
                    continue
                exec_result["exchange"] = open_result
                exec_result["tp_algo_id"] = open_result.get("tp_algo_id")
                fill_entry = float(open_result.get("fill_entry_price") or close_price)
                tp = float(open_result.get("take_profit_price") or 0)
                new_tp_algo_ids[entry_id] = open_result.get("tp_algo_id")
                leg_qty = float(open_result.get("final_leg_filled_qty") or qty)
                if leg_qty <= 0:
                    leg_qty = qty
                exec_result["leg_qty"] = leg_qty
                if open_result.get("overfill_warning"):
                    exec_result["overfill_warning"] = True
                    exec_result["overfill_qty"] = open_result.get("overfill_qty")
                qty = leg_qty
            elif dry_run:
                fill_entry = close_price
                tp = take_profit_price_pct(side, fill_entry, params)
                exec_result["exchange"] = {"status": "DRY_RUN_OPEN"}
                exec_result["tp_algo_id"] = None

            if tp is None:
                tp = (
                    entry_tp
                    if tp_mode_uses_channel(params)
                    else take_profit_price_pct(
                        side,
                        fill_entry,
                        params,
                        tp_pct_override=controls["tp_pct"],
                    )
                )

            open_legs[entry_id] = OpenLeg(
                entry_id=entry_id,
                side=side,
                qty=qty,
                entry_time=ts,
                entry_price=fill_entry,
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
