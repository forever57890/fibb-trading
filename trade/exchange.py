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
    "ensure_dual_side_position_mode",
    "market_close_leg_qty",
    "market_open_leg",
    "require_binance_keys",
    "try_create_trader",
]
