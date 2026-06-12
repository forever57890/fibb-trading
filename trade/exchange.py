"""
Binance USD-M futures execution for FiBB live trading.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fibb_trading.env_loader import load_fibb_env, require_binance_keys
from fibb_trading.trade.binance_futures_trader import (
    BinanceFuturesAPIError,
    BinanceFuturesTrader,
)

load_fibb_env()


def try_create_trader() -> Optional[BinanceFuturesTrader]:
    if not (os.getenv("bn_api_key") and os.getenv("bn_api_secret")):
        return None
    return BinanceFuturesTrader()


def position_side_for_leg(side: str) -> str:
    """Strategy side SHORT/LONG -> Binance hedge positionSide."""
    if side not in {"SHORT", "LONG"}:
        raise ValueError(f"Unsupported side: {side}")
    return side


def market_open_leg(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Increase LONG or SHORT leg by qty (hedge mode)."""
    position_side = position_side_for_leg(side)
    qty = trader.prepare_order_qty(symbol, qty)
    if qty <= 0:
        return {
            "status": "SKIPPED_BELOW_MIN_QTY",
            "symbol": symbol,
            "position_side": position_side,
        }
    if dry_run:
        return {
            "status": "DRY_RUN_OPEN",
            "symbol": symbol,
            "position_side": position_side,
            "qty": qty,
        }
    current = abs(trader.get_position_amount(symbol, position_side))
    target = trader.prepare_order_qty(symbol, current + qty)
    if target <= 0:
        return {
            "status": "SKIPPED_BELOW_MIN_QTY",
            "symbol": symbol,
            "position_side": position_side,
        }
    return trader.adjust_position_with_ioc_then_market(
        symbol,
        position_side,
        target_qty=target,
        mode="open",
        dry_run=False,
    )


def _position_entry_price(
    trader: BinanceFuturesTrader, symbol: str, position_side: str
) -> float:
    for pos in trader.get_positions_summary(symbol=symbol, only_open=True):
        if pos.get("position_side") == position_side:
            return float(pos.get("entry_price") or 0)
    return 0.0


def _marginal_fill_entry_price(
    qty_before: float,
    entry_before: float,
    qty_after: float,
    entry_after: float,
) -> float:
    """Average entry price for the qty added this open (hedge positionSide)."""
    delta = qty_after - qty_before
    if delta <= 0:
        return entry_after or entry_before
    if qty_before <= 0:
        return entry_after
    if entry_after <= 0:
        return entry_before
    return (entry_after * qty_after - entry_before * qty_before) / delta


def _tp_price_from_pct(side: str, entry_price: float, tp_pct: float) -> float:
    if side == "LONG":
        return entry_price * (1 + tp_pct)
    return entry_price * (1 - tp_pct)


def _order_executed_qty(
    trader: BinanceFuturesTrader, symbol: str, order_id: Any
) -> float:
    if order_id is None:
        return 0.0
    try:
        order = trader.get_order(symbol, int(order_id))
        return float(order.get("executedQty") or 0)
    except Exception:
        return 0.0


def _leg_filled_total(open_leg: Dict[str, Any]) -> float:
    return (
        float(open_leg.get("pre_limit_filled_qty") or 0)
        + float(open_leg.get("ioc_filled_qty") or 0)
        + float(open_leg.get("market_remainder_qty") or 0)
    )


def _position_amount(trader: BinanceFuturesTrader, symbol: str, position_side: str) -> float:
    return abs(trader.get_position_amount(symbol, position_side))


def _open_leg_filled_qty(
    trader: BinanceFuturesTrader,
    symbol: str,
    position_side: str,
    qty_before: float,
    leg_qty: float,
) -> float:
    """How much of this leg open actually reached the exchange (source of truth)."""
    delta = _position_amount(trader, symbol, position_side) - qty_before
    return min(leg_qty, max(0.0, delta))


def _open_leg_remaining_qty(
    trader: BinanceFuturesTrader,
    symbol: str,
    position_side: str,
    qty_before: float,
    leg_qty: float,
) -> float:
    return max(0.0, leg_qty - _open_leg_filled_qty(trader, symbol, position_side, qty_before, leg_qty))


def _close_leg_closed_qty(qty_before: float, qty_after: float, leg_qty: float) -> float:
    return min(leg_qty, max(0.0, qty_before - qty_after))


def _close_leg_remaining_qty(
    trader: BinanceFuturesTrader,
    symbol: str,
    position_side: str,
    qty_before: float,
    leg_qty: float,
) -> float:
    qty_after = _position_amount(trader, symbol, position_side)
    return max(0.0, leg_qty - _close_leg_closed_qty(qty_before, qty_after, leg_qty))


def _order_fill_qty(
    trader: BinanceFuturesTrader, symbol: str, order: Dict[str, Any]
) -> float:
    executed = float(order.get("executedQty") or 0)
    if executed > 0:
        return executed
    order_id = order.get("orderId")
    if order_id is None:
        return 0.0
    return _order_executed_qty(trader, symbol, order_id)


def _tp_trigger_valid(side: str, mark_price: float, trigger_price: float) -> bool:
    """Binance TP must be on the profitable side of mark or it can fill immediately."""
    if mark_price <= 0 or trigger_price <= 0:
        return False
    if side == "LONG":
        return trigger_price > mark_price
    return trigger_price < mark_price


def open_leg_with_tp(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    tp_pct: float,
    *,
    signal_close: float,
    take_profit_price: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Open a leg, then place TAKE_PROFIT_MARKET.

    Default TP: fill_entry ± tp_pct. If take_profit_price is set (tp_mode 2/3 通道價)，
    使用該絕對價格掛 TP。
    """
    position_side = position_side_for_leg(side)
    requested_qty = qty
    qty = trader.prepare_order_qty(symbol, qty)
    if qty <= 0:
        min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
        stepped = trader.quantize_qty_down(symbol, requested_qty)
        return {
            "status": "SKIPPED_BELOW_MIN_QTY",
            "symbol": symbol,
            "position_side": position_side,
            "requested_qty": requested_qty,
            "stepped_qty": stepped,
            "min_qty": min_qty,
        }
    close_side = "SELL" if position_side == "LONG" else "BUY"

    if dry_run:
        fill_entry = signal_close
        if take_profit_price is not None:
            tp_price = trader.round_price(symbol, take_profit_price)
        else:
            tp_price = trader.round_price(symbol, _tp_price_from_pct(side, fill_entry, tp_pct))
        return {
            "status": "DRY_RUN_OPEN",
            "symbol": symbol,
            "position_side": position_side,
            "qty": qty,
            "signal_close": signal_close,
            "fill_entry_price": fill_entry,
            "take_profit_price": tp_price,
            "requested_qty": qty,
            "pre_limit_filled_qty": 0.0,
            "ioc_filled_qty": 0.0,
            "market_remainder_qty": 0.0,
            "final_leg_filled_qty": qty,
            "tp_algo_id": None,
        }

    qty_before = abs(trader.get_position_amount(symbol, position_side))
    entry_before = _position_entry_price(trader, symbol, position_side)
    min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
    interval_ms, max_attempts = trader._ioc_settings()  # noqa: SLF001
    wait_ms = trader._pre_ioc_limit_wait_ms()  # noqa: SLF001

    open_leg: Dict[str, Any] = {
        "mode": "open_exact_qty",
        "position_side": position_side,
        "requested_qty": qty,
        "execution_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
        "pre_ioc_limit_wait_ms": wait_ms,
        "ioc_interval_ms": interval_ms,
        "ioc_max_attempts": max_attempts,
        "pre_limit": {},
        "pre_limit_filled_qty": 0.0,
        "ioc_attempts": [],
        "ioc_filled_qty": 0.0,
        "market_remainder_qty": 0.0,
        "status": "NOOP",
    }

    # Drop stale entry limits from failed parallel runs before placing a new one.
    open_leg["cancel_stale_limits"] = trader.cancel_regular_open_orders(symbol)

    # 1) Pre-limit for this leg qty only (not target position).
    pre_side, pre_price = trader._pre_limit_order_params(symbol, position_side, "open")  # noqa: SLF001
    pre_order = trader.create_order(
        {
            "symbol": symbol,
            "side": pre_side,
            "positionSide": position_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": qty,
            "price": pre_price,
        }
    )
    open_leg["pre_limit"] = {
        "side": pre_side,
        "price": pre_price,
        "order_qty": qty,
        "order_id": pre_order.get("orderId"),
        "executed_qty_immediate": float(pre_order.get("executedQty") or 0),
    }

    time.sleep(wait_ms / 1000.0)
    pre_order_id = pre_order.get("orderId")
    pos_mid = _position_amount(trader, symbol, position_side)
    pre_filled = min(qty, max(0.0, pos_mid - qty_before))
    pre_filled = min(
        qty,
        max(pre_filled, min(qty, _order_executed_qty(trader, symbol, pre_order_id))),
    )
    if pre_order_id is not None:
        try:
            open_leg["pre_limit"]["cancel"] = trader.cancel_order(symbol, int(pre_order_id))
            pos_after_cancel = _position_amount(trader, symbol, position_side)
            pre_filled = min(
                qty,
                max(
                    pre_filled,
                    max(0.0, pos_after_cancel - qty_before),
                    min(qty, _order_executed_qty(trader, symbol, pre_order_id)),
                ),
            )
        except BinanceFuturesAPIError as exc:
            open_leg["pre_limit"]["cancel_error"] = str(exc.payload)

    # Drop unfilled passive limits so IOC does not stack on a live GTC.
    try:
        open_leg["pre_limit"]["cancel_open_orders"] = trader.cancel_regular_open_orders(symbol)
        pos_after_sweep = _position_amount(trader, symbol, position_side)
        pre_filled = min(qty, max(pre_filled, max(0.0, pos_after_sweep - qty_before)))
    except BinanceFuturesAPIError as exc:
        open_leg["pre_limit"]["cancel_open_orders_error"] = str(exc.payload)

    open_leg["pre_limit_filled_qty"] = pre_filled

    # 2) IOC loop for the remainder of this leg (position delta is source of truth).
    for attempt in range(1, max_attempts + 1):
        remaining = _open_leg_remaining_qty(
            trader, symbol, position_side, qty_before, qty
        )
        if remaining < min_qty:
            open_leg["status"] = "OPENED_LEG_QTY_FULL"
            break
        side_ioc, price_ioc, level_qty = trader._ioc_order_params(symbol, position_side, "open")  # noqa: SLF001
        raw_qty = min(remaining, level_qty)
        if raw_qty < min_qty:
            time.sleep(interval_ms / 1000.0)
            continue
        order_qty = trader.prepare_order_qty(symbol, raw_qty)
        if order_qty <= 0:
            time.sleep(interval_ms / 1000.0)
            continue
        ioc_order = trader.create_order(
            {
                "symbol": symbol,
                "side": side_ioc,
                "positionSide": position_side,
                "type": "LIMIT",
                "timeInForce": "IOC",
                "quantity": order_qty,
                "price": price_ioc,
            }
        )
        executed = _order_fill_qty(trader, symbol, ioc_order)
        position_filled = _open_leg_filled_qty(
            trader, symbol, position_side, qty_before, qty
        )
        open_leg["ioc_filled_qty"] = max(
            float(open_leg.get("ioc_filled_qty") or 0),
            max(0.0, position_filled - pre_filled),
        )
        remaining = _open_leg_remaining_qty(
            trader, symbol, position_side, qty_before, qty
        )
        open_leg["ioc_attempts"].append(
            {
                "attempt": attempt,
                "side": side_ioc,
                "price": price_ioc,
                "order_qty": order_qty,
                "executed_qty": executed,
                "position_filled_qty": position_filled,
                "book_level_qty": level_qty,
                "remaining_after": remaining,
                "order_id": ioc_order.get("orderId"),
                "order_status": ioc_order.get("status"),
            }
        )
        if remaining < min_qty:
            open_leg["status"] = "OPENED_IOC"
            break
        time.sleep(interval_ms / 1000.0)
    else:
        open_leg["status"] = "IOC_MAX_ATTEMPTS_REACHED"

    # 3) MARKET fill the final remainder for this leg only.
    remaining = _open_leg_remaining_qty(trader, symbol, position_side, qty_before, qty)
    if remaining >= min_qty:
        order_side = "BUY" if position_side == "LONG" else "SELL"
        market_qty = trader.prepare_order_qty(symbol, remaining)
        if market_qty > 0:
            market_order = trader.create_order(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "positionSide": position_side,
                    "type": "MARKET",
                    "quantity": market_qty,
                }
            )
            open_leg["market_remainder_qty"] = market_qty
            open_leg["market_order"] = market_order
            open_leg["status"] = "OPENED_IOC_THEN_MARKET"
    elif open_leg["pre_limit_filled_qty"] >= min_qty and open_leg["status"] in {
        "OPENED_IOC",
        "IOC_MAX_ATTEMPTS_REACHED",
    }:
        open_leg["status"] = "OPENED_PRE_LIMIT"

    qty_after = abs(trader.get_position_amount(symbol, position_side))
    entry_after = _position_entry_price(trader, symbol, position_side)
    fill_entry = _marginal_fill_entry_price(qty_before, entry_before, qty_after, entry_after)
    if fill_entry <= 0:
        fill_entry = signal_close
    position_delta = max(0.0, qty_after - qty_before)
    tracked_fill = _leg_filled_total(open_leg)
    final_leg_filled_qty = (
        min(qty, position_delta) if position_delta > 0 else min(qty, tracked_fill)
    )
    open_leg["position_before"] = qty_before
    open_leg["position_after"] = qty_after
    open_leg["position_delta"] = position_delta
    if position_delta > qty + min_qty * 0.5:
        open_leg["overfill_qty"] = round(position_delta - qty, 8)
        open_leg["overfill_warning"] = True
    if take_profit_price is not None:
        tp_price = trader.round_price(symbol, take_profit_price)
    else:
        tp_price = trader.round_price(symbol, _tp_price_from_pct(side, fill_entry, tp_pct))

    tp_algo_id = None
    tp_order = None
    tp_qty = trader.prepare_order_qty(symbol, final_leg_filled_qty)
    if tp_qty > 0:
        mark_price = trader.get_mark_price(symbol)
        if not _tp_trigger_valid(side, mark_price, tp_price):
            tp_price = trader.round_price(symbol, _tp_price_from_pct(side, fill_entry, tp_pct))
        try:
            tp_order = trader.create_algo_conditional_order(
                symbol=symbol,
                side=close_side,
                position_side=position_side,
                order_type="TAKE_PROFIT_MARKET",
                quantity=tp_qty,
                trigger_price=tp_price,
            )
            tp_algo_id = tp_order.get("algoId") or tp_order.get("clientAlgoId")
        except Exception as exc:
            tp_order = {"error": str(exc)}

    return {
        **open_leg,
        "signal_close": signal_close,
        "fill_entry_price": fill_entry,
        "final_leg_filled_qty": final_leg_filled_qty,
        "tracked_fill_qty": tracked_fill,
        "take_profit_price": tp_price,
        "tp_order": tp_order,
        "tp_algo_id": tp_algo_id,
    }


def cancel_leg_tp(
    trader: BinanceFuturesTrader,
    symbol: str,
    tp_algo_id: Any,
) -> Dict[str, Any]:
    """Cancel a single TP algo order by algoId before closing the leg."""
    if tp_algo_id is None:
        return {"status": "NO_TP_ALGO_ID"}
    try:
        result = trader._request(  # noqa: SLF001
            "DELETE",
            "/fapi/v1/algoOrder",
            params={"symbol": symbol, "algoId": tp_algo_id},
            signed=True,
        )
        return {"status": "CANCELLED", "result": result}
    except BinanceFuturesAPIError as exc:
        return {"status": "CANCEL_ERROR", "error": str(exc.payload)}


def replace_leg_tp(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    take_profit_price: float,
    old_tp_algo_id: Any = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Replace TP algo order for one leg: cancel old TP (if any), then create new TP.
    """
    position_side = position_side_for_leg(side)
    close_side = "SELL" if position_side == "LONG" else "BUY"
    requested_qty = qty
    rounded_qty = trader.prepare_order_qty(symbol, qty)
    tp_price = trader.round_price(symbol, take_profit_price)

    if rounded_qty <= 0:
        stepped = trader.quantize_qty_down(symbol, requested_qty)
        min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
        return {
            "status": "SKIPPED_BELOW_MIN_QTY",
            "position_side": position_side,
            "requested_qty": requested_qty,
            "stepped_qty": stepped,
            "min_qty": min_qty,
            "take_profit_price": tp_price,
            "old_tp_algo_id": old_tp_algo_id,
            "tp_algo_id": old_tp_algo_id,
            "note": "保留既有 TP，不取消",
        }

    if dry_run:
        return {
            "status": "DRY_RUN_REPRICE_TP",
            "position_side": position_side,
            "qty": rounded_qty,
            "take_profit_price": tp_price,
            "old_tp_algo_id": old_tp_algo_id,
            "tp_algo_id": old_tp_algo_id,
        }

    mark_price = trader.get_mark_price(symbol)
    if not _tp_trigger_valid(side, mark_price, tp_price):
        return {
            "status": "SKIPPED_INVALID_TP",
            "position_side": position_side,
            "take_profit_price": tp_price,
            "mark_price": mark_price,
            "old_tp_algo_id": old_tp_algo_id,
            "tp_algo_id": old_tp_algo_id,
            "note": "TP trigger on wrong side of mark; kept existing TP",
        }

    cancel_result = cancel_leg_tp(trader, symbol, old_tp_algo_id)

    try:
        tp_order = trader.create_algo_conditional_order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=rounded_qty,
            trigger_price=tp_price,
        )
        tp_algo_id = tp_order.get("algoId") or tp_order.get("clientAlgoId")
        return {
            "status": "REPLACED",
            "cancel": cancel_result,
            "tp_order": tp_order,
            "take_profit_price": tp_price,
            "tp_algo_id": tp_algo_id,
        }
    except Exception as exc:
        return {
            "status": "REPLACE_ERROR",
            "cancel": cancel_result,
            "error": str(exc),
            "take_profit_price": tp_price,
            "tp_algo_id": old_tp_algo_id,
        }


def close_leg_with_ioc(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Close one virtual leg by qty: passive GTC pre-limit → IOC → MARKET remainder.

    Mirrors open_leg_with_tp execution (exact leg qty, not full positionSide).
    """
    position_side = position_side_for_leg(side)
    requested_qty = qty
    qty = trader.prepare_order_qty(symbol, qty)
    if qty <= 0:
        stepped = trader.quantize_qty_down(symbol, requested_qty)
        min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
        return {
            "status": "SKIPPED_BELOW_MIN_QTY",
            "symbol": symbol,
            "position_side": position_side,
            "requested_qty": requested_qty,
            "stepped_qty": stepped,
            "min_qty": min_qty,
        }
    qty_before = abs(trader.get_position_amount(symbol, position_side))
    current = trader.prepare_order_qty(symbol, min(qty, qty_before))
    if current <= 0:
        return {
            "status": "SKIPPED_NO_POSITION",
            "symbol": symbol,
            "position_side": position_side,
            "requested_qty": requested_qty,
            "qty_before": qty_before,
        }
    min_qty = trader._min_trade_qty(symbol)  # noqa: SLF001
    interval_ms, max_attempts = trader._ioc_settings()  # noqa: SLF001
    wait_ms = trader._pre_ioc_limit_wait_ms()  # noqa: SLF001

    if dry_run:
        return {
            "status": "DRY_RUN_CLOSE",
            "symbol": symbol,
            "position_side": position_side,
            "requested_qty": qty,
            "execution_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
            "pre_ioc_limit_wait_ms": wait_ms,
            "ioc_interval_ms": interval_ms,
            "ioc_max_attempts": max_attempts,
        }

    if current <= 0:
        return {"status": "NO_QTY", "qty": 0.0, "position_side": position_side}

    close_leg: Dict[str, Any] = {
        "mode": "close_exact_qty",
        "position_side": position_side,
        "requested_qty": current,
        "execution_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
        "pre_ioc_limit_wait_ms": wait_ms,
        "ioc_interval_ms": interval_ms,
        "ioc_max_attempts": max_attempts,
        "pre_limit": {},
        "pre_limit_filled_qty": 0.0,
        "ioc_attempts": [],
        "ioc_filled_qty": 0.0,
        "market_remainder_qty": 0.0,
        "status": "NOOP",
    }

    def _closed_total() -> float:
        return _leg_filled_total(close_leg)

    # 1) Pre-limit for this leg qty only.
    pre_side, pre_price = trader._pre_limit_order_params(symbol, position_side, "close")  # noqa: SLF001
    pre_order = trader.create_order(
        {
            "symbol": symbol,
            "side": pre_side,
            "positionSide": position_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": current,
            "price": pre_price,
        }
    )
    close_leg["pre_limit"] = {
        "side": pre_side,
        "price": pre_price,
        "order_qty": current,
        "order_id": pre_order.get("orderId"),
        "executed_qty_immediate": float(pre_order.get("executedQty") or 0),
    }

    time.sleep(wait_ms / 1000.0)
    pre_order_id = pre_order.get("orderId")
    pos_mid = abs(trader.get_position_amount(symbol, position_side))
    pre_filled = min(current, max(0.0, qty_before - pos_mid))
    pre_filled = min(
        current,
        max(pre_filled, min(current, _order_executed_qty(trader, symbol, pre_order_id))),
    )
    if pre_order_id is not None:
        try:
            close_leg["pre_limit"]["cancel"] = trader.cancel_order(symbol, int(pre_order_id))
            pos_after_cancel = abs(trader.get_position_amount(symbol, position_side))
            pre_filled = min(
                current,
                max(
                    pre_filled,
                    max(0.0, qty_before - pos_after_cancel),
                    min(current, _order_executed_qty(trader, symbol, pre_order_id)),
                ),
            )
        except BinanceFuturesAPIError as exc:
            close_leg["pre_limit"]["cancel_error"] = str(exc.payload)

    close_leg["pre_limit_filled_qty"] = pre_filled

    try:
        close_leg["pre_limit"]["cancel_open_orders"] = trader.cancel_regular_open_orders(symbol)
        pos_after_sweep = _position_amount(trader, symbol, position_side)
        pre_filled = min(
            current,
            max(pre_filled, max(0.0, qty_before - pos_after_sweep)),
        )
        close_leg["pre_limit_filled_qty"] = pre_filled
    except BinanceFuturesAPIError as exc:
        close_leg["pre_limit"]["cancel_open_orders_error"] = str(exc.payload)

    # 2) IOC loop for the remainder of this leg (position delta is source of truth).
    for attempt in range(1, max_attempts + 1):
        remaining = _close_leg_remaining_qty(
            trader, symbol, position_side, qty_before, current
        )
        if remaining < min_qty:
            close_leg["status"] = "CLOSED_LEG_QTY_FULL"
            break
        side_ioc, price_ioc, level_qty = trader._ioc_order_params(  # noqa: SLF001
            symbol, position_side, "close"
        )
        raw_qty = min(remaining, level_qty)
        if raw_qty < min_qty:
            time.sleep(interval_ms / 1000.0)
            continue
        order_qty = trader.prepare_order_qty(symbol, raw_qty)
        if order_qty <= 0:
            time.sleep(interval_ms / 1000.0)
            continue
        ioc_order = trader.create_order(
            {
                "symbol": symbol,
                "side": side_ioc,
                "positionSide": position_side,
                "type": "LIMIT",
                "timeInForce": "IOC",
                "quantity": order_qty,
                "price": price_ioc,
            }
        )
        executed = _order_fill_qty(trader, symbol, ioc_order)
        closed_qty = _close_leg_closed_qty(
            qty_before,
            _position_amount(trader, symbol, position_side),
            current,
        )
        close_leg["ioc_filled_qty"] = max(
            float(close_leg.get("ioc_filled_qty") or 0),
            max(0.0, closed_qty - pre_filled),
        )
        remaining = _close_leg_remaining_qty(
            trader, symbol, position_side, qty_before, current
        )
        close_leg["ioc_attempts"].append(
            {
                "attempt": attempt,
                "side": side_ioc,
                "price": price_ioc,
                "order_qty": order_qty,
                "executed_qty": executed,
                "position_closed_qty": closed_qty,
                "book_level_qty": level_qty,
                "remaining_after": remaining,
                "order_id": ioc_order.get("orderId"),
                "order_status": ioc_order.get("status"),
            }
        )
        if remaining < min_qty:
            close_leg["status"] = "CLOSED_IOC"
            break
        time.sleep(interval_ms / 1000.0)
    else:
        close_leg["status"] = "IOC_MAX_ATTEMPTS_REACHED"

    # 3) MARKET fill the final remainder for this leg only.
    remaining = _close_leg_remaining_qty(
        trader, symbol, position_side, qty_before, current
    )
    if remaining >= min_qty:
        order_side = "SELL" if position_side == "LONG" else "BUY"
        market_qty = trader.prepare_order_qty(symbol, remaining)
        if market_qty > 0:
            market_order = trader.create_order(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "positionSide": position_side,
                    "type": "MARKET",
                    "quantity": market_qty,
                }
            )
            close_leg["market_remainder_qty"] = market_qty
            close_leg["market_order"] = market_order
            close_leg["status"] = "CLOSED_IOC_THEN_MARKET"
    elif close_leg["pre_limit_filled_qty"] >= min_qty and close_leg["status"] in {
        "CLOSED_IOC",
        "IOC_MAX_ATTEMPTS_REACHED",
    }:
        close_leg["status"] = "CLOSED_PRE_LIMIT"

    qty_after = abs(trader.get_position_amount(symbol, position_side))
    position_delta = max(0.0, qty_before - qty_after)
    tracked_close = _closed_total()
    final_leg_closed_qty = (
        min(current, position_delta) if position_delta > 0 else min(current, tracked_close)
    )
    close_leg["position_before"] = qty_before
    close_leg["position_after"] = qty_after
    close_leg["position_delta"] = position_delta
    close_leg["final_leg_closed_qty"] = final_leg_closed_qty
    close_leg["tracked_close_qty"] = tracked_close
    if position_delta > current + min_qty * 0.5:
        close_leg["overclose_qty"] = round(position_delta - current, 8)
        close_leg["overclose_warning"] = True
    if close_leg["status"] == "NOOP" and final_leg_closed_qty >= min_qty:
        close_leg["status"] = "CLOSED"
    return close_leg


def market_close_leg_qty(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Reduce one virtual leg by qty (partial close)."""
    position_side = position_side_for_leg(side)
    qty = trader.prepare_order_qty(symbol, qty)
    current = abs(trader.get_position_amount(symbol, position_side))
    if qty > current:
        qty = trader.prepare_order_qty(symbol, current)

    if dry_run:
        return {
            "status": "DRY_RUN_CLOSE",
            "symbol": symbol,
            "position_side": position_side,
            "qty": qty,
        }

    if qty <= 0:
        return {"status": "NO_QTY", "qty": 0.0}

    order_side = "SELL" if position_side == "LONG" else "BUY"
    order = trader.create_order(
        {
            "symbol": symbol,
            "side": order_side,
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": qty,
        }
    )
    return {
        "status": "CLOSED_MARKET",
        "qty": qty,
        "order": order,
        "position_side": position_side,
    }


def ensure_dual_side_position_mode(trader: BinanceFuturesTrader) -> Dict[str, Any]:
    """Hedge mode (dual side) required for simultaneous LONG/SHORT legs."""
    try:
        data = trader._signed_get("/fapi/v1/positionSide/dual")  # noqa: SLF001
        if data.get("dualSidePosition"):
            return {"dual_side": True, "changed": False}
        if os.getenv("FIBB_ENABLE_HEDGE_MODE", "0") != "1":
            return {
                "dual_side": False,
                "changed": False,
                "warning": "Set FIBB_ENABLE_HEDGE_MODE=1 to auto-enable hedge mode",
            }
        result = trader._signed_post(  # noqa: SLF001
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true"},
        )
        return {"dual_side": True, "changed": True, "result": result}
    except BinanceFuturesAPIError as exc:
        return {"error": str(exc.payload), "dual_side": None}


__all__ = [
    "BinanceFuturesAPIError",
    "BinanceFuturesTrader",
    "cancel_leg_tp",
    "close_leg_with_ioc",
    "ensure_dual_side_position_mode",
    "market_close_leg_qty",
    "market_open_leg",
    "open_leg_with_tp",
    "replace_leg_tp",
    "require_binance_keys",
    "try_create_trader",
]
