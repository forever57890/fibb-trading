"""
FiBB 通道策略邏輯（對應 Pine Script「FiBB 15m BTC Layer Strategy」）。

通道：basis = SMA(close, len)，偏移 = ta.atr(len) × Fibonacci 倍率（Wilder RMA）。
進場：價格由內向外穿越 T1/T2/T3 做空、B1/B2/B3 做多（每 leg 獨立持倉）。
出場：預設全 leg 固定 % 止盈；T1/T3/B1/B3 無止損；T2/B2 觸 T3/B3 後以 T2/B2 通道價止損。
止盈模式由 FibbParams.tp_mode 控制（0=固定 %，1=basis，2=通道隨 K，3=通道鎖定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from fibb_trading.core import fibb_config
from fibb_trading.core.fibb_config import (
    CHANNEL_TP_ENTRY_INDEX,
    channel_tp_target_band,
    tp_mode_channel_tracks_bar,
    tp_mode_uses_channel,
    DEFERRED_CHANNEL_SL,
    DEFAULT_PARAMS,
    FibbParams,
)


@dataclass
class OpenLeg:
    entry_id: str
    side: str
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    take_profit_price: float
    stop_loss_price: Optional[float]  # None = 尚未啟用止損
    band: str
    take_profit_band: str = ""  # 通道止盈所跟隨的欄位（basis / top1 / …）
    sl_use_channel: bool = False  # True：止損價為 T2/B2 通道（回撤觸軌平倉）


def compute_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA，對應 Pine `ta.rma`（首值為前 length 根 SMA）。"""
    alpha = 1.0 / length
    values = series.to_numpy(dtype=float)
    out = pd.Series(float("nan"), index=series.index, dtype=float)
    if len(values) < length:
        return out
    out.iloc[length - 1] = values[:length].mean()
    for i in range(length, len(values)):
        prev = out.iloc[i - 1]
        v = values[i]
        out.iloc[i] = prev if pd.isna(v) else alpha * v + (1.0 - alpha) * prev
    return out


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """對應 Pine `ta.atr(length)` = `ta.rma(ta.tr(true), length)`。"""
    return compute_rma(compute_true_range(df), length)


def compute_fibb_channels(df: pd.DataFrame, params: FibbParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """在 K 線 DataFrame 上附加 basis / top1-3 / bott1-3。"""
    out = df.copy()
    length = params.length
    basis = out["close"].rolling(length, min_periods=length).mean()
    avg = compute_atr(out, length)

    r1 = avg * params.fib_ratio_1
    r2 = avg * params.fib_ratio_2
    r3 = avg * params.fib_ratio_3

    out["basis"] = basis
    out["top1"] = basis + r1
    out["top2"] = basis + r2
    out["top3"] = basis + r3
    out["bott1"] = basis - r1
    out["bott2"] = basis - r2
    out["bott3"] = basis - r3
    return out


def _touch_short(curr: pd.Series, prev: pd.Series, band_col: str) -> bool:
    level = curr[band_col]
    prev_level = prev[band_col]
    if pd.isna(level) or pd.isna(prev_level):
        return False
    return float(curr["high"]) >= float(level) and float(prev["high"]) < float(prev_level)


def _touch_long(curr: pd.Series, prev: pd.Series, band_col: str) -> bool:
    level = curr[band_col]
    prev_level = prev[band_col]
    if pd.isna(level) or pd.isna(prev_level):
        return False
    return float(curr["low"]) <= float(level) and float(prev["low"]) > float(prev_level)


def analyze_entry_legs(
    curr: pd.Series,
    prev: pd.Series,
    open_entry_ids: set,
) -> List[dict]:
    """
    每個 leg 在本根 K 的進場狀態（供實盤 log 解釋為何未開單）。
    status: channels_not_ready | already_open | touch_signal | no_touch
    """
    rows: List[dict] = []
    for entry_id, side, qty, band in fibb_config.ALL_LEGS:
        level = curr.get(band)
        prev_level = prev.get(band)
        row: dict = {
            "entry_id": entry_id,
            "side": side,
            "band": band,
            "qty_btc": qty,
        }
        if pd.isna(level) or pd.isna(prev_level):
            row["status"] = "channels_not_ready"
            row["reason"] = f"通道 {band} 尚未就緒（歷史 K 不足，無法計算 SMA/ATR）"
            if side == "SHORT":
                row["touch"] = {
                    "high": float(curr["high"]) if pd.notna(curr.get("high")) else None,
                    "prev_high": float(prev["high"]) if pd.notna(prev.get("high")) else None,
                    "band": None,
                    "prev_band": None,
                }
            else:
                row["touch"] = {
                    "low": float(curr["low"]) if pd.notna(curr.get("low")) else None,
                    "prev_low": float(prev["low"]) if pd.notna(prev.get("low")) else None,
                    "band": None,
                    "prev_band": None,
                }
            rows.append(row)
            continue

        level_f = float(level)
        prev_level_f = float(prev_level)
        if side == "SHORT":
            high = float(curr["high"])
            prev_high = float(prev["high"])
            crossed = high >= level_f and prev_high < prev_level_f
            row["touch"] = {
                "rule": "high >= band 且 high[1] < band[1]",
                "high": high,
                "prev_high": prev_high,
                "band": level_f,
                "prev_band": prev_level_f,
                "crossed": crossed,
            }
        else:
            low = float(curr["low"])
            prev_low = float(prev["low"])
            crossed = low <= level_f and prev_low > prev_level_f
            row["touch"] = {
                "rule": "low <= band 且 low[1] > band[1]",
                "low": low,
                "prev_low": prev_low,
                "band": level_f,
                "prev_band": prev_level_f,
                "crossed": crossed,
            }

        if entry_id in open_entry_ids:
            row["status"] = "already_open"
            row["reason"] = "該 leg 已有持倉，不重複進場"
        elif crossed:
            row["status"] = "touch_signal"
            row["reason"] = "觸軌進場訊號"
        else:
            row["status"] = "no_touch"
            if side == "SHORT":
                if high < level_f:
                    row["reason"] = "未觸軌：最高價低於上軌"
                elif prev_high >= prev_level_f:
                    row["reason"] = "未觸軌：前一根已在上軌之上（非首次穿越）"
                else:
                    row["reason"] = "未觸軌"
            else:
                if low > level_f:
                    row["reason"] = "未觸軌：最低價高於下軌"
                elif prev_low <= prev_level_f:
                    row["reason"] = "未觸軌：前一根已在下軌之下（非首次穿越）"
                else:
                    row["reason"] = "未觸軌"
        rows.append(row)
    return rows


def analyze_open_leg_exits(
    leg: OpenLeg, bar_high: float, bar_low: float
) -> dict:
    """持倉 leg 在本根 K 是否應平倉（供 log）。"""
    exit_price, reason = try_exit_leg(leg, bar_high, bar_low)
    out: dict = {
        "entry_id": leg.entry_id,
        "side": leg.side,
        "qty": leg.qty,
        "entry_price": leg.entry_price,
        "take_profit": leg.take_profit_price,
        "stop_loss": leg.stop_loss_price,
        "sl_use_channel": leg.sl_use_channel,
        "bar_high": bar_high,
        "bar_low": bar_low,
        "would_exit": exit_price is not None,
        "exit_reason": reason,
        "exit_price": exit_price,
    }
    if leg.side == "LONG":
        out["tp_hit"] = bar_high >= leg.take_profit_price
        if leg.stop_loss_price is None:
            out["sl_hit"] = False
        elif leg.sl_use_channel:
            out["sl_hit"] = bar_high >= leg.stop_loss_price
        else:
            out["sl_hit"] = bar_low <= leg.stop_loss_price
    else:
        out["tp_hit"] = bar_low <= leg.take_profit_price
        if leg.stop_loss_price is None:
            out["sl_hit"] = False
        elif leg.sl_use_channel:
            out["sl_hit"] = bar_low <= leg.stop_loss_price
        else:
            out["sl_hit"] = bar_high >= leg.stop_loss_price
    if not out["would_exit"]:
        if leg.stop_loss_price is None:
            out["hold_reason"] = "未觸及止盈；此 leg 尚無止損"
        else:
            out["hold_reason"] = "未觸及止盈或止損"
    return out


def detect_entry_signals(
    curr: pd.Series,
    prev: pd.Series,
    open_entry_ids: set,
) -> List[Tuple[str, str, float, str]]:
    """
    回傳本根 K 線收盤可開倉的 leg 列表：(entry_id, side, qty, band_col)。
    """
    signals: List[Tuple[str, str, float, str]] = []
    for entry_id, side, qty, band in fibb_config.SHORT_LEGS:
        if entry_id in open_entry_ids:
            continue
        if _touch_short(curr, prev, band):
            signals.append((entry_id, side, qty, band))
    for entry_id, side, qty, band in fibb_config.LONG_LEGS:
        if entry_id in open_entry_ids:
            continue
        if _touch_long(curr, prev, band):
            signals.append((entry_id, side, qty, band))
    return signals


def take_profit_price_pct(
    side: str, entry_price: float, params: FibbParams = DEFAULT_PARAMS
) -> float:
    """固定百分比止盈（tp_mode=0 或進場暫用 % 時）。"""
    if side == "LONG":
        return entry_price * (1 + params.tp_pct)
    return entry_price * (1 - params.tp_pct)


def uses_channel_tp(entry_id: str, params: FibbParams) -> bool:
    return tp_mode_uses_channel(params) and entry_id in CHANNEL_TP_ENTRY_INDEX


def channel_tp_level(
    entry_id: str, side: str, bar: pd.Series, channel_tp_offset: int
) -> float:
    col = channel_tp_target_band(entry_id, side, channel_tp_offset)
    level = bar[col]
    if pd.isna(level):
        raise ValueError(f"Channel TP level {col} is NaN for {entry_id}")
    return float(level)


def refresh_channel_take_profits(
    open_legs: Dict[str, OpenLeg], curr: pd.Series, params: FibbParams
) -> None:
    """持倉期間每根 K 更新通道止盈價（軌道隨 SMA/ATR 移動）。"""
    if not tp_mode_channel_tracks_bar(params):
        return
    for leg in open_legs.values():
        if not leg.take_profit_band:
            continue
        level = curr[leg.take_profit_band]
        if not pd.isna(level):
            leg.take_profit_price = float(level)


def refresh_reprice_tp_to_basis(
    open_legs: Dict[str, OpenLeg], curr: pd.Series, params: FibbParams
) -> None:
    """
    持倉期間每根 K 將所有未平 leg 的止盈價改為當根 basis（與實盤 FIBB_REPRICE_TP_TO_BASIS 一致）。
    應在當根出場檢查之後、新開倉之前呼叫（新倉本根仍用進場 % TP，下一根才跟 basis）。
    """
    if params.tp_mode != 1:
        return
    basis = curr.get("basis")
    if pd.isna(basis):
        return
    basis_tp = float(basis)
    for leg in open_legs.values():
        leg.take_profit_price = basis_tp
        leg.take_profit_band = "basis"


def resolve_take_profit(
    entry_id: str, side: str, entry_price: float, bar: pd.Series, params: FibbParams
) -> Tuple[float, str]:
    """回傳 (止盈價, 止盈通道欄位名；% 模式時欄位名為空字串)。"""
    if uses_channel_tp(entry_id, params):
        col = channel_tp_target_band(entry_id, side, params.channel_tp_offset)
        return channel_tp_level(entry_id, side, bar, params.channel_tp_offset), col
    return take_profit_price_pct(side, entry_price, params), ""


def resolve_entry_take_profit(
    entry_id: str, side: str, entry_price: float, bar: pd.Series, params: FibbParams
) -> Tuple[float, str]:
    """
    開倉當下止盈價。
    tp_mode=1：先固定 %（下一根 K 才跟 basis）；0/2 見 resolve_take_profit。
    """
    if params.tp_mode == 1:
        return take_profit_price_pct(side, entry_price, params), ""
    return resolve_take_profit(entry_id, side, entry_price, bar, params)


def bracket_prices(
    side: str, entry_price: float, params: FibbParams = DEFAULT_PARAMS
) -> Tuple[float, float]:
    """% TP/SL（--pct-tp 或測試用）。"""
    tp = take_profit_price_pct(side, entry_price, params)
    if side == "LONG":
        sl = entry_price * (1 - params.sl_pct)
    else:
        sl = entry_price * (1 + params.sl_pct)
    return tp, sl


def uses_deferred_channel_sl(entry_id: str, params: FibbParams) -> bool:
    return params.use_deferred_channel_sl and entry_id in DEFERRED_CHANNEL_SL


def try_exit_leg(
    leg: OpenLeg, high: float, low: float
) -> Tuple[Optional[float], Optional[str]]:
    """
    檢查平倉：
    - 一般 leg：通道或 % 止盈；無止損（T1/T3/B1/B3）
    - T2/B2：觸及外層後，內層通道價止損（空單價格回落觸 T2、多單回升觸 B2）
    同根 TP 與止損同時觸及時，止損優先。
    """
    sl_price = leg.stop_loss_price
    if leg.side == "LONG":
        tp_hit = high >= leg.take_profit_price
        if sl_price is None:
            sl_hit = False
        elif leg.sl_use_channel:
            sl_hit = high >= sl_price
        else:
            sl_hit = low <= sl_price
        if sl_hit and tp_hit:
            reason = "CHANNEL_STOP" if leg.sl_use_channel else "STOP_LOSS"
            return sl_price, reason
        if tp_hit:
            return leg.take_profit_price, "TAKE_PROFIT"
        if sl_hit:
            return sl_price, "CHANNEL_STOP" if leg.sl_use_channel else "STOP_LOSS"
    else:
        tp_hit = low <= leg.take_profit_price
        if sl_price is None:
            sl_hit = False
        elif leg.sl_use_channel:
            sl_hit = low <= sl_price
        else:
            sl_hit = high >= sl_price
        if sl_hit and tp_hit:
            reason = "CHANNEL_STOP" if leg.sl_use_channel else "STOP_LOSS"
            return sl_price, reason
        if tp_hit:
            return leg.take_profit_price, "TAKE_PROFIT"
        if sl_hit:
            return sl_price, "CHANNEL_STOP" if leg.sl_use_channel else "STOP_LOSS"
    return None, None


def arm_deferred_channel_stops(
    open_legs: Dict[str, OpenLeg],
    curr: pd.Series,
    prev: pd.Series,
    params: FibbParams,
) -> None:
    """
    價格觸及外層 T3/B3 時，為仍持有的 T2/B2 設定內層通道止損價（當根 top2/bott2）。
    """
    if not params.use_deferred_channel_sl:
        return
    for inner_id, (outer_band, inner_band) in DEFERRED_CHANNEL_SL.items():
        leg = open_legs.get(inner_id)
        if leg is None or leg.stop_loss_price is not None:
            continue
        if leg.side == "SHORT":
            outer_touch = _touch_short(curr, prev, outer_band)
        else:
            outer_touch = _touch_long(curr, prev, outer_band)
        if not outer_touch:
            continue
        level = curr[inner_band]
        if pd.isna(level):
            continue
        leg.stop_loss_price = float(level)
        leg.sl_use_channel = True


def backtest_equity(closed: List[dict], params: FibbParams) -> float:
    """已實現權益（不含未平倉浮動）。"""
    realized = sum(t["net_pnl"] for t in closed)
    return params.initial_capital + realized


def resolve_entry_qty(
    requested_qty: float, entry_price: float, equity: float, params: FibbParams
) -> float:
    """
    依 equity × leverage 上限縮單，對應 TradingView 資金不足時自動減少 fixed qty。
    回傳 0 表示資金不足以開最小單位。
    """
    if params.initial_capital <= 0 or params.leverage <= 0:
        return requested_qty
    max_qty = (equity * params.leverage) / entry_price
    if max_qty >= requested_qty:
        return requested_qty
    if max_qty < 1.0:
        return 0.0
    return float(int(max_qty))


def leg_pnl(
    leg: OpenLeg, exit_price: float, fee_rate: float
) -> Tuple[float, float, float]:
    if leg.side == "LONG":
        gross = leg.qty * (exit_price - leg.entry_price)
    else:
        gross = leg.qty * (leg.entry_price - exit_price)
    fee = (leg.qty * leg.entry_price + leg.qty * exit_price) * fee_rate
    net = gross - fee
    return gross, fee, net


def run_fibb_backtest(
    klines: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    params: FibbParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    逐根 K 線模擬（process_orders_on_close）：
    1) 先用當根 high/low 檢查既有 leg 的 TP/SL（區間外仍平倉，對應 Pine strategy.exit）
    2) 僅在 [start_ts, end_ts] 內以收盤價開新倉（對應 inDateRange）
    """
    df = compute_fibb_channels(klines, params)
    warmup = params.length + 1
    if len(df) <= warmup:
        return pd.DataFrame()

    open_legs: Dict[str, OpenLeg] = {}
    closed: List[dict] = []

    for i in range(warmup, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = curr["open_time"]
        if ts < start_ts:
            continue

        bar_high = float(curr["high"])
        bar_low = float(curr["low"])
        close_price = float(curr["close"])
        in_entry_range = ts <= end_ts

        refresh_channel_take_profits(open_legs, curr, params)

        # --- exits (intrabar) ---
        for entry_id in list(open_legs.keys()):
            leg = open_legs[entry_id]
            exit_price, reason = try_exit_leg(leg, bar_high, bar_low)
            if exit_price is None:
                continue
            gross, fee, net = leg_pnl(leg, exit_price, params.fee_rate)
            notional = leg.qty * leg.entry_price
            closed.append(
                {
                    "entry_id": leg.entry_id,
                    "side": leg.side,
                    "qty": leg.qty,
                    "band": leg.band,
                    "take_profit_band": leg.take_profit_band,
                    "entry_time": leg.entry_time,
                    "exit_time": ts,
                    "entry_price": leg.entry_price,
                    "exit_price": exit_price,
                    "take_profit_price": leg.take_profit_price,
                    "stop_loss_price": leg.stop_loss_price,
                    "exit_reason": reason,
                    "take_profit_hit": reason == "TAKE_PROFIT",
                    "stop_loss_hit": reason in ("STOP_LOSS", "CHANNEL_STOP"),
                    "channel_stop_hit": reason == "CHANNEL_STOP",
                    "gross_pnl": gross,
                    "fee": fee,
                    "net_pnl": net,
                    "net_return": net / notional if notional else 0.0,
                    "win": net > 0,
                }
            )
            del open_legs[entry_id]

        refresh_reprice_tp_to_basis(open_legs, curr, params)

        # --- entries at close (Pine: inDateRange only) ---
        if not in_entry_range:
            continue
        if len(open_legs) >= params.max_open_legs:
            continue

        equity = backtest_equity(closed, params)
        signals = detect_entry_signals(curr, prev, set(open_legs.keys()))
        for entry_id, side, qty, band in signals:
            if len(open_legs) >= params.max_open_legs:
                break
            qty = resolve_entry_qty(qty, close_price, equity, params)
            if qty <= 0:
                continue
            tp, tp_band = resolve_entry_take_profit(
                entry_id, side, close_price, curr, params
            )
            if uses_deferred_channel_sl(entry_id, params):
                # T2/B2：進場無止損，外層觸軌後由 arm_deferred_channel_stops 啟用通道止損
                sl: Optional[float] = None
                sl_channel = False
            elif not params.use_deferred_channel_sl:
                # 經典 Pine：全 leg 固定 % 止損
                _, sl = bracket_prices(side, close_price, params)
                sl_channel = False
            else:
                # T1/T3/B1/B3：僅 % 止盈，不設止損
                sl = None
                sl_channel = False
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

        arm_deferred_channel_stops(open_legs, curr, prev, params)

    trades = pd.DataFrame(closed)
    if trades.empty:
        return trades
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    trades["cum_net_pnl"] = trades["net_pnl"].cumsum()
    trades["cum_peak"] = trades["cum_net_pnl"].cummax()
    trades["drawdown"] = trades["cum_net_pnl"] - trades["cum_peak"]
    return trades


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
        }
    wins = trades[trades["gross_pnl"] > 0]
    losses = trades[trades["gross_pnl"] <= 0]
    gross_profit = float(wins["gross_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["gross_pnl"].sum()) if not losses.empty else 0.0
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else None
    return {
        "total_trades": len(trades),
        "net_profit": float(trades["net_pnl"].sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "win_rate": float(trades["win"].mean()),
        "profit_factor": pf,
        "max_drawdown": float(trades["drawdown"].min()),
        "tp_hits": int(trades["take_profit_hit"].sum()),
        "sl_hits": int(trades["stop_loss_hit"].sum()),
    }


def leg_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("entry_id")
        .agg(
            trades=("net_pnl", "count"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
        )
        .reset_index()
    )
