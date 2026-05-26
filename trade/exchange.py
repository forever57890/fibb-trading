"""
Binance USD-M futures execution for FiBB live trading.
"""

from __future__ import annotations

import os
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
    qty = trader.round_qty(symbol, qty)
    if dry_run:
        return {
            "status": "DRY_RUN_OPEN",
            "symbol": symbol,
            "position_side": position_side,
            "qty": qty,
        }
    current = abs(trader.get_position_amount(symbol, position_side))
    target = trader.round_qty(symbol, current + qty)
    return trader.adjust_position_with_ioc_then_market(
        symbol,
        position_side,
        target_qty=target,
        mode="open",
        dry_run=False,
    )


def open_leg_with_tp(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    take_profit_price: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Open a leg with qty, then place a TAKE_PROFIT_MARKET algo order for the filled qty.
    Returns the open result plus tp_algo_id (None on dry run or if TP placement fails).
    """
    position_side = position_side_for_leg(side)
    qty = trader.round_qty(symbol, qty)
    tp_price = trader.round_price(symbol, take_profit_price)
    close_side = "SELL" if position_side == "LONG" else "BUY"

    if dry_run:
        return {
            "status": "DRY_RUN_OPEN",
            "symbol": symbol,
            "position_side": position_side,
            "qty": qty,
            "take_profit_price": tp_price,
            "tp_algo_id": None,
        }

    current = abs(trader.get_position_amount(symbol, position_side))
    target = trader.round_qty(symbol, current + qty)
    open_result = trader.adjust_position_with_ioc_then_market(
        symbol,
        position_side,
        target_qty=target,
        mode="open",
        dry_run=False,
    )

    filled_qty = abs(trader.get_position_amount(symbol, position_side))
    filled_qty = trader.round_qty(symbol, filled_qty)

    tp_algo_id = None
    tp_order = None
    if filled_qty >= (trader._min_trade_qty(symbol) or 0):
        try:
            tp_order = trader.create_algo_conditional_order(
                symbol=symbol,
                side=close_side,
                position_side=position_side,
                order_type="TAKE_PROFIT_MARKET",
                quantity=filled_qty,
                trigger_price=tp_price,
            )
            tp_algo_id = tp_order.get("algoId") or tp_order.get("clientAlgoId")
        except Exception as exc:
            tp_order = {"error": str(exc)}

    return {
        **open_result,
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
    rounded_qty = trader.round_qty(symbol, qty)
    tp_price = trader.round_price(symbol, take_profit_price)

    if dry_run:
        return {
            "status": "DRY_RUN_REPRICE_TP",
            "position_side": position_side,
            "qty": rounded_qty,
            "take_profit_price": tp_price,
            "old_tp_algo_id": old_tp_algo_id,
            "tp_algo_id": old_tp_algo_id,
        }

    cancel_result = cancel_leg_tp(trader, symbol, old_tp_algo_id)
    if rounded_qty <= 0:
        return {
            "status": "NO_QTY",
            "cancel": cancel_result,
            "take_profit_price": tp_price,
            "tp_algo_id": None,
        }

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


def market_close_leg_qty(
    trader: BinanceFuturesTrader,
    symbol: str,
    side: str,
    qty: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Reduce one virtual leg by qty (partial close)."""
    position_side = position_side_for_leg(side)
    qty = trader.round_qty(symbol, qty)
    current = abs(trader.get_position_amount(symbol, position_side))
    if qty > current:
        qty = trader.round_qty(symbol, current)

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
    "ensure_dual_side_position_mode",
    "market_close_leg_qty",
    "market_open_leg",
    "open_leg_with_tp",
    "replace_leg_tp",
    "require_binance_keys",
    "try_create_trader",
]
