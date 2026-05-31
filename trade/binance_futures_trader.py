"""
Binance USD-M futures helpers (requests only, no python-binance).

API reference:
https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from fibb_trading.env_loader import load_fibb_env, require_binance_keys

load_fibb_env()


DEFAULT_BASE_URL = "https://fapi.binance.com"
RECV_WINDOW = 5000
DEFAULT_IOC_INTERVAL_MS = 100
DEFAULT_IOC_MAX_ATTEMPTS = 10
DEFAULT_PRE_IOC_LIMIT_WAIT_MS = 2000


class BinanceFuturesAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Binance API error HTTP {status_code}: {payload}")


class BinanceFuturesTrader:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        recv_window: int = RECV_WINDOW,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("bn_api_key")
        self.api_secret = api_secret or os.getenv("bn_api_secret")
        self.base_url = (base_url or os.getenv("FIBB_BINANCE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self._exchange_info_cache: Optional[Dict[str, Any]] = None

        if not self.api_key or not self.api_secret:
            require_binance_keys()
            self.api_key = os.getenv("bn_api_key")
            self.api_secret = os.getenv("bn_api_secret")

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        signed = {k: v for k, v in params.items() if v is not None}
        signed["timestamp"] = self._server_timestamp()
        signed["recvWindow"] = self.recv_window
        query = urlencode(signed, doseq=True)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed["signature"] = signature
        return signed

    def _server_timestamp(self) -> int:
        data = self._public_get("/fapi/v1/time")
        return int(data["serverTime"])

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params, signed=False)

    def _signed_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params, signed=True)

    def _signed_post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("POST", path, params=params, signed=True)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        params = params or {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        method = method.upper()

        if method == "GET":
            response = requests.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        elif method == "POST":
            response = requests.post(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        elif method == "DELETE":
            response = requests.delete(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise BinanceFuturesAPIError(response.status_code, payload)

        if not response.text:
            return {}
        return response.json()

    def get_exchange_info(self) -> Dict[str, Any]:
        if self._exchange_info_cache is None:
            self._exchange_info_cache = self._public_get("/fapi/v1/exchangeInfo")
        return self._exchange_info_cache

    def get_symbol_filters(self, symbol: str) -> Tuple[float, float, float]:
        info = self.get_exchange_info()
        symbol_info = next(s for s in info["symbols"] if s["symbol"] == symbol)
        step_size = tick_size = min_qty = 0.0
        for f in symbol_info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
        return step_size, tick_size, min_qty

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        rounded = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return float(rounded)

    def quantize_qty_down(self, symbol: str, qty: float) -> float:
        """Round down to LOT_SIZE step; result may be below minQty."""
        step_size, _, _ = self.get_symbol_filters(symbol)
        return self._round_step(qty, step_size)

    def prepare_order_qty(self, symbol: str, qty: float) -> float:
        """Exchange-valid quantity after step rounding, or 0.0 if below minQty."""
        stepped = self.quantize_qty_down(symbol, qty)
        min_qty = self._min_trade_qty(symbol)
        return stepped if stepped >= min_qty else 0.0

    def round_qty(self, symbol: str, qty: float) -> float:
        prepared = self.prepare_order_qty(symbol, qty)
        if prepared <= 0:
            stepped = self.quantize_qty_down(symbol, qty)
            min_qty = self._min_trade_qty(symbol)
            raise ValueError(
                f"Quantity {stepped} is below minQty {min_qty} for {symbol}"
            )
        return prepared

    def round_price(self, symbol: str, price: float) -> float:
        _, tick_size, _ = self.get_symbol_filters(symbol)
        return self._round_step(price, tick_size)

    def get_mark_price(self, symbol: str) -> float:
        data = self._public_get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def get_book_ticker(self, symbol: str) -> Dict[str, float]:
        """GET /fapi/v1/ticker/bookTicker — best bid/ask (first level)."""
        data = self._public_get("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return {
            "best_bid": float(data["bidPrice"]),
            "best_bid_qty": float(data["bidQty"]),
            "best_ask": float(data["askPrice"]),
            "best_ask_qty": float(data["askQty"]),
        }

    def get_balances(self) -> List[Dict[str, Any]]:
        """GET /fapi/v2/balance — futures wallet balance per asset (USER_DATA)."""
        return self._signed_get("/fapi/v2/balance")

    def get_balance_summary(self, asset: str = "USDT") -> Dict[str, Any]:
        """Summarize one asset from /fapi/v2/balance (default USDT margin wallet)."""
        rows = self.get_balances()
        row = next((r for r in rows if r.get("asset") == asset), None)
        if not row:
            return {"asset": asset, "found": False}

        return {
            "asset": asset,
            "found": True,
            "wallet_balance": float(row.get("balance") or 0),
            "cross_wallet_balance": float(row.get("crossWalletBalance") or 0),
            "available_balance": float(row.get("availableBalance") or 0),
            "cross_un_pnl": float(row.get("crossUnPnl") or 0),
            "max_withdraw_amount": float(row.get("maxWithdrawAmount") or 0),
            "update_time": row.get("updateTime"),
        }

    def get_positions(
        self,
        symbol: Optional[str] = None,
        only_open: bool = True,
    ) -> List[Dict[str, Any]]:
        """GET /fapi/v2/positionRisk — position info (USER_DATA)."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        rows = self._signed_get("/fapi/v2/positionRisk", params or None)
        if only_open:
            rows = [r for r in rows if abs(float(r.get("positionAmt") or 0)) > 0]
        return rows

    def get_positions_summary(
        self,
        symbol: Optional[str] = None,
        only_open: bool = True,
    ) -> List[Dict[str, Any]]:
        """Normalized open positions for logging / pre-trade checks."""
        out = []
        for pos in self.get_positions(symbol=symbol, only_open=only_open):
            out.append(
                {
                    "symbol": pos.get("symbol"),
                    "position_side": pos.get("positionSide"),
                    "position_amt": float(pos.get("positionAmt") or 0),
                    "entry_price": float(pos.get("entryPrice") or 0),
                    "mark_price": float(pos.get("markPrice") or 0),
                    "unrealized_pnl": float(pos.get("unRealizedProfit") or 0),
                    "leverage": pos.get("leverage"),
                    "liquidation_price": float(pos.get("liquidationPrice") or 0),
                    "margin_type": pos.get("marginType"),
                }
            )
        return out

    def check_account(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Pre-trade account snapshot: USDT balance + open positions.
        symbol: if set, position list is filtered to that symbol only.
        """
        return {
            "balance_usdt": self.get_balance_summary("USDT"),
            "positions": self.get_positions_summary(symbol=symbol, only_open=True),
        }

    def get_open_orders_summary(self, symbol: str) -> List[Dict[str, Any]]:
        """GET /fapi/v1/openOrders — normalized for run logs."""
        rows = self._signed_get("/fapi/v1/openOrders", {"symbol": symbol})
        out: List[Dict[str, Any]] = []
        for order in rows:
            out.append(
                {
                    "order_id": order.get("orderId"),
                    "client_order_id": order.get("clientOrderId"),
                    "type": order.get("type"),
                    "side": order.get("side"),
                    "position_side": order.get("positionSide"),
                    "price": float(order.get("price") or 0),
                    "stop_price": float(order.get("stopPrice") or 0),
                    "orig_qty": float(order.get("origQty") or 0),
                    "executed_qty": float(order.get("executedQty") or 0),
                    "status": order.get("status"),
                    "time_in_force": order.get("timeInForce"),
                    "reduce_only": order.get("reduceOnly"),
                }
            )
        return out

    def get_algo_open_orders_summary(self, symbol: str) -> List[Dict[str, Any]]:
        """GET /fapi/v1/openAlgoOrders — TP/SL and other conditional algo orders."""
        rows = self._signed_get("/fapi/v1/openAlgoOrders", {"symbol": symbol})
        if isinstance(rows, dict):
            rows = rows.get("orders") or rows.get("data") or []
        out: List[Dict[str, Any]] = []
        for order in rows:
            out.append(
                {
                    "algo_id": order.get("algoId"),
                    "client_algo_id": order.get("clientAlgoId"),
                    "algo_type": order.get("algoType"),
                    "order_type": order.get("orderType"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "position_side": order.get("positionSide"),
                    "quantity": float(order.get("quantity") or 0),
                    "trigger_price": float(order.get("triggerPrice") or 0),
                    "price": float(order.get("price") or 0),
                    "algo_status": order.get("algoStatus"),
                    "working_type": order.get("workingType"),
                    "close_position": order.get("closePosition"),
                }
            )
        return out

    def get_account_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Balance, positions, open orders, and mark price for logging."""
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "api",
            "symbol": symbol,
            "mark_price": self.get_mark_price(symbol),
            "balance_usdt": self.get_balance_summary("USDT"),
            "positions": self.get_positions_summary(symbol=symbol, only_open=True),
            "open_orders": self.get_open_orders_summary(symbol),
            "algo_open_orders": self.get_algo_open_orders_summary(symbol),
        }

    def get_position_amount(self, symbol: str, position_side: str) -> float:
        for pos in self.get_positions(symbol=symbol, only_open=True):
            if pos.get("positionSide") == position_side:
                return float(pos.get("positionAmt") or 0)
        return 0.0

    def create_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """POST /fapi/v1/order (signed TRADE endpoint)."""
        return self._signed_post("/fapi/v1/order", params)

    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """GET /fapi/v1/order — query a single order by orderId."""
        return self._signed_get(
            "/fapi/v1/order",
            {"symbol": symbol, "orderId": int(order_id)},
        )

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """DELETE /fapi/v1/order — cancel one regular order (not algo TP/SL)."""
        return self._request(
            "DELETE",
            "/fapi/v1/order",
            params={"symbol": symbol, "orderId": int(order_id)},
            signed=True,
        )

    def create_algo_conditional_order(
        self,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        quantity: float,
        trigger_price: float,
        working_type: str = "CONTRACT_PRICE",
    ) -> Dict[str, Any]:
        """
        POST /fapi/v1/algoOrder — TP/SL (CONDITIONAL).
        order_type: TAKE_PROFIT_MARKET | STOP_MARKET | ...
        """
        if order_type not in {
            "STOP_MARKET",
            "TAKE_PROFIT_MARKET",
            "STOP",
            "TAKE_PROFIT",
            "TRAILING_STOP_MARKET",
        }:
            raise ValueError(f"Unsupported algo order type: {order_type}")

        return self._signed_post(
            "/fapi/v1/algoOrder",
            {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": side,
                "positionSide": position_side,
                "type": order_type,
                "quantity": quantity,
                "triggerPrice": trigger_price,
                "workingType": working_type,
            },
        )

    def cancel_algo_open_orders(self, symbol: str) -> Dict[str, Any]:
        """DELETE /fapi/v1/algoOpenOrders — cancel all open algo (TP/SL) orders."""
        return self._request(
            "DELETE",
            "/fapi/v1/algoOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )

    def cancel_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel regular limit/IOC orders and algo TP/SL before close."""
        regular: Dict[str, Any] = {}
        algo: Dict[str, Any] = {}
        try:
            regular = self._request(
                "DELETE",
                "/fapi/v1/allOpenOrders",
                params={"symbol": symbol},
                signed=True,
            )
        except BinanceFuturesAPIError:
            pass
        try:
            algo = self.cancel_algo_open_orders(symbol)
        except BinanceFuturesAPIError:
            pass
        return {"regular": regular, "algo": algo}

    def _min_trade_qty(self, symbol: str) -> float:
        _, _, min_qty = self.get_symbol_filters(symbol)
        return min_qty

    def _ioc_settings(self) -> Tuple[int, int]:
        interval_ms = int(
            os.getenv(
                "FIBB_IOC_INTERVAL_MS",
                os.getenv("FIBB_CLOSE_IOC_INTERVAL_MS", str(DEFAULT_IOC_INTERVAL_MS)),
            )
        )
        max_attempts = int(
            os.getenv(
                "FIBB_IOC_MAX_ATTEMPTS",
                os.getenv("FIBB_CLOSE_IOC_MAX_ATTEMPTS", str(DEFAULT_IOC_MAX_ATTEMPTS)),
            )
        )
        return interval_ms, max_attempts

    def _pre_ioc_limit_wait_ms(self) -> int:
        return int(os.getenv("FIBB_PRE_IOC_LIMIT_WAIT_MS", str(DEFAULT_PRE_IOC_LIMIT_WAIT_MS)))

    def _remaining_qty(
        self,
        symbol: str,
        position_side: str,
        mode: str,
        target_qty: float,
    ) -> Tuple[float, float]:
        """Return (remaining_to_execute, current_position_qty)."""
        current = abs(self.get_position_amount(symbol, position_side))
        if mode == "open":
            remaining = target_qty - current
        else:
            remaining = current
        min_qty = self._min_trade_qty(symbol)
        if remaining < min_qty:
            return 0.0, current
        return self.round_qty(symbol, remaining), current

    def _pre_limit_order_params(
        self, symbol: str, position_side: str, mode: str
    ) -> Tuple[str, float]:
        """
        Passive queue limit (GTC): post at same-side best price and wait for fill.
        Not opponent/taker pricing — that is reserved for the IOC phase.

        open LONG  -> BUY  @ best bid (join bid queue)
        open SHORT -> SELL @ best ask (join ask queue)
        close LONG -> SELL @ best ask
        close SHORT-> BUY  @ best bid
        """
        book = self.get_book_ticker(symbol)
        if mode == "open":
            if position_side == "LONG":
                return "BUY", self.round_price(symbol, book["best_bid"])
            return "SELL", self.round_price(symbol, book["best_ask"])
        if position_side == "LONG":
            return "SELL", self.round_price(symbol, book["best_ask"])
        return "BUY", self.round_price(symbol, book["best_bid"])

    def _execute_pre_ioc_limit(
        self,
        symbol: str,
        position_side: str,
        mode: str,
        target_qty: float,
        leg: Dict[str, Any],
        dry_run: bool,
    ) -> None:
        """
        Before IOC loop: one GTC LIMIT for full remaining qty at passive queue price,
        wait (default 2000ms), cancel open orders, then IOC @ opponent handles the rest.
        """
        wait_ms = self._pre_ioc_limit_wait_ms()
        remaining, pos_before = self._remaining_qty(symbol, position_side, mode, target_qty)
        order_side, price = self._pre_limit_order_params(symbol, position_side, mode)

        pre: Dict[str, Any] = {
            "wait_ms": wait_ms,
            "pricing": "passive_queue",
            "side": order_side,
            "price": price,
            "order_qty": remaining,
            "position_before": pos_before,
            "skipped": True,
        }

        if remaining <= 0:
            pre["reason"] = "no_remaining"
            leg["pre_limit"] = pre
            return

        pre["skipped"] = False

        if dry_run:
            pre["status"] = "DRY_RUN_PRE_LIMIT"
            pre["note"] = (
                f"Would place passive GTC LIMIT qty={remaining} @ {price} (queue), "
                f"wait {wait_ms}ms, cancel, then IOC @ opponent"
            )
            leg["pre_limit"] = pre
            return

        order = self.create_order(
            {
                "symbol": symbol,
                "side": order_side,
                "positionSide": position_side,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": remaining,
                "price": price,
            }
        )
        pre["order"] = order
        pre["order_id"] = order.get("orderId")
        pre["executed_qty_immediate"] = float(order.get("executedQty") or 0)

        time.sleep(wait_ms / 1000.0)

        try:
            self.cancel_open_orders(symbol)
            pre["cancelled_open_orders"] = True
        except BinanceFuturesAPIError as exc:
            pre["cancel_error"] = str(exc.payload)

        pos_after = abs(self.get_position_amount(symbol, position_side))
        if mode == "open":
            filled = max(0.0, pos_after - pos_before)
        else:
            filled = max(0.0, pos_before - pos_after)

        pre["position_after"] = pos_after
        pre["filled_qty"] = filled
        pre["status"] = "PLACED_WAITED_CANCELLED"
        leg["pre_limit"] = pre
        leg["pre_limit_filled_qty"] = filled

    def _ioc_order_params(
        self, symbol: str, position_side: str, mode: str
    ) -> Tuple[str, float, float]:
        """
        Opponent first level for IOC.
        open LONG  -> BUY  @ best ask
        open SHORT -> SELL @ best bid
        close LONG -> SELL @ best bid
        close SHORT-> BUY  @ best ask
        """
        book = self.get_book_ticker(symbol)
        if mode == "open":
            if position_side == "LONG":
                return (
                    "BUY",
                    self.round_price(symbol, book["best_ask"]),
                    book["best_ask_qty"],
                )
            return (
                "SELL",
                self.round_price(symbol, book["best_bid"]),
                book["best_bid_qty"],
            )
        if position_side == "LONG":
            return (
                "SELL",
                self.round_price(symbol, book["best_bid"]),
                book["best_bid_qty"],
            )
        return (
            "BUY",
            self.round_price(symbol, book["best_ask"]),
            book["best_ask_qty"],
        )

    def adjust_position_with_ioc_then_market(
        self,
        symbol: str,
        position_side: str,
        target_qty: float,
        mode: str,
        dry_run: bool = False,
        interval_ms: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        mode ``open``: build position up to ``target_qty``.
        mode ``close``: reduce position to zero.

        1) GTC LIMIT for full remaining qty @ passive queue price, wait (default 2000ms), cancel
        2) IOC @ opponent 1st level every interval_ms (max attempts)
        3) MARKET remainder
        """
        if mode not in {"open", "close"}:
            raise ValueError(f"mode must be 'open' or 'close', got {mode}")

        default_interval, default_max = self._ioc_settings()
        interval_ms = interval_ms if interval_ms is not None else default_interval
        max_attempts = max_attempts if max_attempts is not None else default_max

        min_qty = self._min_trade_qty(symbol)
        target_qty = self.round_qty(symbol, target_qty)
        initial_qty = abs(self.get_position_amount(symbol, position_side))

        leg: Dict[str, Any] = {
            "mode": mode,
            "position_side": position_side,
            "target_qty": target_qty,
            "initial_qty": initial_qty,
            "pre_limit": {},
            "pre_limit_filled_qty": 0.0,
            "ioc_attempts": [],
            "ioc_filled_qty": 0.0,
            "market_remainder_qty": 0.0,
            "pre_ioc_limit_wait_ms": self._pre_ioc_limit_wait_ms(),
            "ioc_interval_ms": interval_ms,
            "ioc_max_attempts": max_attempts,
            "execution_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
            "status": "NOOP",
        }

        if mode == "close" and initial_qty < min_qty:
            leg["status"] = "NO_POSITION_TO_CLOSE"
            return leg
        if mode == "open" and target_qty < min_qty:
            leg["status"] = "TARGET_TOO_SMALL"
            return leg

        if dry_run:
            order_side, price, _ = self._ioc_order_params(symbol, position_side, mode)
            self._execute_pre_ioc_limit(
                symbol, position_side, mode, target_qty, leg, dry_run=True
            )
            leg["status"] = f"DRY_RUN_{mode.upper()}"
            leg["simulated_ioc_side"] = order_side
            leg["simulated_ioc_price"] = price
            leg["note"] = (
                f"Pre LIMIT full qty @ {price}, wait {leg['pre_ioc_limit_wait_ms']}ms, "
                f"then IOC {max_attempts}x every {interval_ms}ms, then MARKET"
            )
            return leg

        self._execute_pre_ioc_limit(
            symbol, position_side, mode, target_qty, leg, dry_run=False
        )

        interval_sec = interval_ms / 1000.0
        done_status = "OPENED_IOC" if mode == "open" else "CLOSED_IOC"

        for attempt in range(1, max_attempts + 1):
            current = abs(self.get_position_amount(symbol, position_side))
            remaining = (
                target_qty - current if mode == "open" else current
            )

            if remaining < min_qty:
                leg["status"] = done_status
                break

            order_side, price, level_qty = self._ioc_order_params(
                symbol, position_side, mode
            )
            order_qty = self.round_qty(symbol, min(remaining, level_qty))
            if order_qty < min_qty:
                time.sleep(interval_sec)
                continue

            order = self.create_order(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "positionSide": position_side,
                    "type": "LIMIT",
                    "timeInForce": "IOC",
                    "quantity": order_qty,
                    "price": price,
                }
            )
            executed = float(order.get("executedQty") or 0)
            leg["ioc_filled_qty"] += executed
            leg["ioc_attempts"].append(
                {
                    "attempt": attempt,
                    "side": order_side,
                    "price": price,
                    "order_qty": order_qty,
                    "executed_qty": executed,
                    "book_level_qty": level_qty,
                    "remaining_before": remaining,
                    "position_before": current,
                    "order_id": order.get("orderId"),
                    "order_status": order.get("status"),
                }
            )

            current_after = abs(self.get_position_amount(symbol, position_side))
            remaining_after = (
                target_qty - current_after if mode == "open" else current_after
            )
            if remaining_after < min_qty:
                leg["status"] = done_status
                break

            time.sleep(interval_sec)
        else:
            leg["status"] = "IOC_MAX_ATTEMPTS_REACHED"

        current = abs(self.get_position_amount(symbol, position_side))
        remaining = target_qty - current if mode == "open" else current

        if remaining >= min_qty:
            remainder = self.round_qty(symbol, remaining)
            order_side, _, _ = self._ioc_order_params(symbol, position_side, mode)
            market_order = self.create_order(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "positionSide": position_side,
                    "type": "MARKET",
                    "quantity": remainder,
                }
            )
            leg["market_remainder_qty"] = remainder
            leg["market_order"] = market_order
            leg["status"] = (
                "OPENED_IOC_THEN_MARKET" if mode == "open" else "CLOSED_IOC_THEN_MARKET"
            )
        elif leg.get("pre_limit_filled_qty", 0) >= min_qty and leg["status"] in {
            done_status,
            "IOC_MAX_ATTEMPTS_REACHED",
        }:
            leg["status"] = (
                "OPENED_PRE_LIMIT" if mode == "open" else "CLOSED_PRE_LIMIT"
            )

        leg["final_qty"] = abs(self.get_position_amount(symbol, position_side))
        return leg

    def close_position_with_ioc_then_market(
        self,
        symbol: str,
        position_side: str,
        dry_run: bool = False,
        interval_ms: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        current = abs(self.get_position_amount(symbol, position_side))
        return self.adjust_position_with_ioc_then_market(
            symbol,
            position_side,
            target_qty=current,
            mode="close",
            dry_run=dry_run,
            interval_ms=interval_ms,
            max_attempts=max_attempts,
        )

    def close_open_positions_at_market(
        self,
        symbol: str,
        dry_run: bool = False,
        interval_ms: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Daily normal exit for all open legs on symbol.
        Each leg: IOC @ opponent 1st level (100ms, up to 100 tries) then MARKET remainder.
        """
        default_interval, default_max = self._ioc_settings()
        interval_ms = interval_ms if interval_ms is not None else default_interval
        max_attempts = max_attempts if max_attempts is not None else default_max

        positions = self.get_positions_summary(symbol=symbol, only_open=True)
        mark_price = self.get_mark_price(symbol)

        result: Dict[str, Any] = {
            "symbol": symbol,
            "mark_price": mark_price,
            "dry_run": dry_run,
            "close_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
            "pre_ioc_limit_wait_ms": self._pre_ioc_limit_wait_ms(),
            "ioc_interval_ms": interval_ms,
            "ioc_max_attempts": max_attempts,
            "positions_before": positions,
            "positions_after": [],
            "closed": [],
            "status": "NO_POSITION_TO_CLOSE",
        }

        if not positions:
            return result

        if dry_run:
            result["status"] = "DRY_RUN_CLOSE"
            for pos in positions:
                leg = self.close_position_with_ioc_then_market(
                    symbol,
                    pos["position_side"],
                    dry_run=True,
                    interval_ms=interval_ms,
                    max_attempts=max_attempts,
                )
                leg["quantity"] = abs(pos["position_amt"])
                result["closed"].append(leg)
            result["positions_after"] = self.get_positions_summary(
                symbol=symbol, only_open=True
            )
            return result

        if positions:
            self.cancel_open_orders(symbol)

        for pos in positions:
            leg = self.close_position_with_ioc_then_market(
                symbol,
                pos["position_side"],
                dry_run=False,
                interval_ms=interval_ms,
                max_attempts=max_attempts,
            )
            result["closed"].append(leg)

        result["status"] = "CLOSED" if result["closed"] else "NO_POSITION_TO_CLOSE"
        result["positions_after"] = self.get_positions_summary(
            symbol=symbol, only_open=True
        )
        try:
            result["account_after"] = self.check_account(symbol=symbol)
            result["open_orders_after"] = self.get_open_orders_summary(symbol)
            result["algo_open_orders_after"] = self.get_algo_open_orders_summary(symbol)
        except BinanceFuturesAPIError:
            pass
        return result

    def open_position_with_brackets(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        dry_run: bool = False,
        interval_ms: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        side: LONG or SHORT (strategy side)
        Entry: IOC @ opponent 1st level (100ms, up to 100), then MARKET remainder.
        Then TAKE_PROFIT_MARKET + STOP_MARKET via algo API on filled qty (hedge mode).
        """
        if side not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported side: {side}")

        target_qty = self.round_qty(symbol, quantity)
        tp_price = self.round_price(symbol, take_profit_price)
        sl_price = self.round_price(symbol, stop_loss_price)
        close_side = "SELL" if side == "LONG" else "BUY"
        position_side = side

        default_interval, default_max = self._ioc_settings()
        interval_ms = interval_ms if interval_ms is not None else default_interval
        max_attempts = max_attempts if max_attempts is not None else default_max

        mark_price = self.get_mark_price(symbol)
        result: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "target_quantity": target_qty,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "mark_price_at_entry": mark_price,
            "dry_run": dry_run,
            "open_mode": "PRE_LIMIT_THEN_IOC_THEN_MARKET",
            "pre_ioc_limit_wait_ms": self._pre_ioc_limit_wait_ms(),
            "ioc_interval_ms": interval_ms,
            "ioc_max_attempts": max_attempts,
            "planned_brackets": {
                "take_profit_price": tp_price,
                "stop_loss_price": sl_price,
                "close_side": close_side,
                "api": "algo",
            },
            "bracket_api": "algo",
        }

        try:
            result["account_before"] = self.check_account(symbol=symbol)
            result["positions_before"] = self.get_positions_summary(
                symbol=symbol, only_open=True
            )
            result["open_orders_before"] = self.get_open_orders_summary(symbol)
            result["algo_open_orders_before"] = self.get_algo_open_orders_summary(symbol)
        except BinanceFuturesAPIError:
            pass

        entry_leg = self.adjust_position_with_ioc_then_market(
            symbol,
            position_side,
            target_qty=target_qty,
            mode="open",
            dry_run=dry_run,
            interval_ms=interval_ms,
            max_attempts=max_attempts,
        )
        result["entry"] = entry_leg
        result["status"] = entry_leg.get("status", "UNKNOWN")

        if dry_run:
            result["simulated_filled_quantity"] = target_qty
            return result

        filled_qty = abs(self.get_position_amount(symbol, position_side))
        result["filled_quantity"] = filled_qty
        result["positions_after_entry"] = self.get_positions_summary(
            symbol=symbol, only_open=True
        )
        if filled_qty < self._min_trade_qty(symbol):
            result["status"] = "OPEN_INCOMPLETE"
            result["account_after_entry"] = self.check_account(symbol=symbol)
            return result

        filled_qty = self.round_qty(symbol, filled_qty)
        result["account_after_entry"] = self.check_account(symbol=symbol)

        tp_order = self.create_algo_conditional_order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=filled_qty,
            trigger_price=tp_price,
        )
        sl_order = self.create_algo_conditional_order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="STOP_MARKET",
            quantity=filled_qty,
            trigger_price=sl_price,
        )
        result["take_profit_order"] = tp_order
        result["stop_loss_order"] = sl_order
        result["open_orders_after"] = self.get_open_orders_summary(symbol)
        result["algo_open_orders_after"] = self.get_algo_open_orders_summary(symbol)
        result["account_after"] = self.check_account(symbol=symbol)
        if entry_leg.get("status", "").startswith("OPENED"):
            result["status"] = "OPENED"
        return result


if __name__ == "__main__":
    import json
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    trader = BinanceFuturesTrader()
    snapshot = trader.check_account(symbol=symbol)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
