"""
FiBB 通道策略邏輯（對應 Pine Script「FiBB 15m BTC Layer Strategy」）。

通道：basis = SMA(close, len)，偏移 = ta.atr(len) × Fibonacci 倍率（Wilder RMA）。
進場：價格由內側上一道通道抵達並穿越 T1/T2/T3 做空、B1/B2/B3 做多（每 leg 獨立持倉）。
出場：預設全 leg 固定 % 止盈；T1/T3/B1/B3 無止損；T2/B2 觸 T3/B3 後以 T2/B2 通道價止損。
止盈模式由 FibbParams.tp_mode 控制（0=固定 %，1=basis，2=通道隨 K，3=通道鎖定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fibb_trading.core import fibb_config
from fibb_trading.core.fibb_config import (
    CHANNEL_TP_ENTRY_INDEX,
    channel_tp_target_band,
    entry_legs_for_trade,
    side_entry_allowed,
    tp_mode_channel_tracks_bar,
    tp_mode_uses_channel,
    trade_sides_label,
    DEFERRED_CHANNEL_SL,
    DEFAULT_PARAMS,
    ENTRY_APPROACH_FROM,
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
    entry_bar_index: int = 0
    worst_unrealized_gross: float = 0.0  # 持倉期間最差毛浮盈虧（≤0 為浮虧）


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
    _attach_h4_regime_columns(out, params)
    return out


def _attach_h4_regime_columns(out: pd.DataFrame, params: FibbParams) -> None:
    """Attach 4H volatility/trend regime features to 15m bars."""
    if "open_time" not in out.columns or out.empty:
        return
    ohlc_cols = ("open", "high", "low", "close")
    if any(c not in out.columns for c in ohlc_cols):
        return

    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    base = out[["open_time", "open", "high", "low", "close"]].copy()
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    base = base.sort_values("open_time").set_index("open_time")
    h4 = (
        base.resample("4h", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    if h4.empty:
        return

    h4["h4_range_pct"] = (h4["high"] - h4["low"]) / h4["close"]
    h4["h4_range_pct"] = h4["h4_range_pct"].replace([float("inf"), float("-inf")], pd.NA)
    h4["h4_basis"] = h4["close"].rolling(params.length, min_periods=params.length).mean()
    h4["h4_basis_slope"] = h4["h4_basis"] - h4["h4_basis"].shift(1)
    q = min(max(float(params.regime_h4_high_vol_quantile), 0.01), 0.99)
    lookback = max(int(params.regime_h4_lookback), 5)
    h4["h4_range_pct_q_high"] = h4["h4_range_pct"].rolling(
        lookback, min_periods=min(lookback, 20)
    ).quantile(q)

    h4 = h4.reset_index()[["open_time", "h4_range_pct", "h4_basis_slope", "h4_range_pct_q_high"]]
    merged = pd.merge_asof(
        out.sort_values("open_time"),
        h4.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    out["h4_range_pct"] = merged["h4_range_pct"].to_numpy()
    out["h4_basis_slope"] = merged["h4_basis_slope"].to_numpy()
    out["h4_range_pct_q_high"] = merged["h4_range_pct_q_high"].to_numpy()


def _cross_short(curr: pd.Series, prev: pd.Series, band_col: str) -> bool:
    level = curr[band_col]
    prev_level = prev[band_col]
    if pd.isna(level) or pd.isna(prev_level):
        return False
    return float(curr["high"]) >= float(level) and float(prev["high"]) < float(prev_level)


def _cross_long(curr: pd.Series, prev: pd.Series, band_col: str) -> bool:
    level = curr[band_col]
    prev_level = prev[band_col]
    if pd.isna(level) or pd.isna(prev_level):
        return False
    return float(curr["low"]) <= float(level) and float(prev["low"]) > float(prev_level)


def _came_from_prev_channel_short(
    prev: pd.Series, entry_band: str, prev_band: str
) -> bool:
    """
    前一根 K 的價格須處於「內側通道」區間，才視為從該點位抵達外軌。
    T1：前一根 high 在 basis 以內（尚未進入 top1 區）。
    T2/T3：前一根 high 已觸及內軌，但尚未觸及本軌。
    """
    prev_inner = prev.get(prev_band)
    prev_entry = prev.get(entry_band)
    if pd.isna(prev_inner) or pd.isna(prev_entry):
        return False
    prev_high = float(prev["high"])
    prev_inner_f = float(prev_inner)
    prev_entry_f = float(prev_entry)
    if entry_band == "top1":
        return prev_high <= prev_inner_f
    return prev_high >= prev_inner_f and prev_high < prev_entry_f


def _came_from_prev_channel_long(
    prev: pd.Series, entry_band: str, prev_band: str
) -> bool:
    """
    B1：前一根仍在 basis 之上（尚未進入 bott1 區）。
    B2/B3：前一根 low 仍在內軌之上、且尚未觸及本軌（自內側往下穿越）。
    """
    prev_inner = prev.get(prev_band)
    prev_entry = prev.get(entry_band)
    if pd.isna(prev_inner) or pd.isna(prev_entry):
        return False
    prev_low = float(prev["low"])
    prev_inner_f = float(prev_inner)
    prev_entry_f = float(prev_entry)
    if entry_band == "bott1":
        return prev_low >= prev_inner_f
    return prev_low >= prev_inner_f and prev_low > prev_entry_f


def _touch_short(curr: pd.Series, prev: pd.Series, band_col: str, entry_id: str) -> bool:
    if not _cross_short(curr, prev, band_col):
        return False
    prev_band = ENTRY_APPROACH_FROM.get(entry_id)
    if prev_band is None:
        return True
    return _came_from_prev_channel_short(prev, band_col, prev_band)


def _touch_long(curr: pd.Series, prev: pd.Series, band_col: str, entry_id: str) -> bool:
    if not _cross_long(curr, prev, band_col):
        return False
    prev_band = ENTRY_APPROACH_FROM.get(entry_id)
    if prev_band is None:
        return True
    return _came_from_prev_channel_long(prev, band_col, prev_band)


def analyze_entry_legs(
    curr: pd.Series,
    prev: pd.Series,
    open_entry_ids: set,
    params: FibbParams = DEFAULT_PARAMS,
) -> List[dict]:
    """
    每個 leg 在本根 K 的進場狀態（供實盤 log 解釋為何未開單）。
    status: channels_not_ready | already_open | touch_signal | no_touch | side_disabled
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
        if not side_entry_allowed(params, side):
            row["status"] = "side_disabled"
            row["reason"] = (
                f"方向已關閉（trade_sides={trade_sides_label(params.trade_sides)}）"
            )
            rows.append(row)
            continue
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
        prev_band_col = ENTRY_APPROACH_FROM.get(entry_id, "")
        prev_band_level = (
            float(prev[prev_band_col])
            if prev_band_col and pd.notna(prev.get(prev_band_col))
            else None
        )
        if side == "SHORT":
            high = float(curr["high"])
            prev_high = float(prev["high"])
            crossed = _cross_short(curr, prev, band)
            from_prev = (
                _came_from_prev_channel_short(prev, band, prev_band_col)
                if prev_band_col
                else True
            )
            touched = crossed and from_prev
            row["touch"] = {
                "rule": "穿越上軌且前一根自內側通道抵達",
                "high": high,
                "prev_high": prev_high,
                "band": level_f,
                "prev_band": prev_level_f,
                "approach_from_band": prev_band_col,
                "approach_from_level": prev_band_level,
                "crossed": crossed,
                "from_prev_channel": from_prev,
            }
        else:
            low = float(curr["low"])
            prev_low = float(prev["low"])
            crossed = _cross_long(curr, prev, band)
            from_prev = (
                _came_from_prev_channel_long(prev, band, prev_band_col)
                if prev_band_col
                else True
            )
            touched = crossed and from_prev
            row["touch"] = {
                "rule": "穿越下軌且前一根自內側通道抵達",
                "low": low,
                "prev_low": prev_low,
                "band": level_f,
                "prev_band": prev_level_f,
                "approach_from_band": prev_band_col,
                "approach_from_level": prev_band_level,
                "crossed": crossed,
                "from_prev_channel": from_prev,
            }

        if entry_id in open_entry_ids:
            row["status"] = "already_open"
            row["reason"] = "該 leg 已有持倉，不重複進場"
        elif touched:
            row["status"] = "touch_signal"
            row["reason"] = "自內側通道抵達並穿越進場"
        else:
            row["status"] = "no_touch"
            if side == "SHORT":
                if not crossed:
                    if high < level_f:
                        row["reason"] = "未觸軌：最高價低於上軌"
                    elif prev_high >= prev_level_f:
                        row["reason"] = "未觸軌：前一根已在上軌之上（非首次穿越）"
                    else:
                        row["reason"] = "未觸軌"
                elif not from_prev:
                    inner = prev_band_col or "?"
                    row["reason"] = (
                        f"未自內側通道抵達：前一根未從 {inner} 區間來（可能在通道間徘徊）"
                    )
                else:
                    row["reason"] = "未觸軌"
            else:
                if not crossed:
                    if low > level_f:
                        row["reason"] = "未觸軌：最低價高於下軌"
                    elif prev_low <= prev_level_f:
                        row["reason"] = "未觸軌：前一根已在下軌之下（非首次穿越）"
                    else:
                        row["reason"] = "未觸軌"
                elif not from_prev:
                    inner = prev_band_col or "?"
                    row["reason"] = (
                        f"未自內側通道抵達：前一根未從 {inner} 區間來（可能在通道間徘徊）"
                    )
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
    params: FibbParams = DEFAULT_PARAMS,
) -> List[Tuple[str, str, float, str]]:
    """
    回傳本根 K 線收盤可開倉的 leg 列表：(entry_id, side, qty, band_col)。
    僅包含 trade_sides 允許的方向（both / long / short）。
    """
    signals: List[Tuple[str, str, float, str]] = []
    for entry_id, side, qty, band in entry_legs_for_trade(params):
        if entry_id in open_entry_ids:
            continue
        if side == "SHORT":
            if _touch_short(curr, prev, band, entry_id):
                signals.append((entry_id, side, qty, band))
        elif _touch_long(curr, prev, band, entry_id):
            signals.append((entry_id, side, qty, band))
    return signals


def take_profit_price_pct(
    side: str,
    entry_price: float,
    params: FibbParams = DEFAULT_PARAMS,
    tp_pct_override: Optional[float] = None,
) -> float:
    """固定百分比止盈（tp_mode=0 或進場暫用 % 時）。"""
    tp_pct = float(tp_pct_override) if tp_pct_override is not None else params.tp_pct
    if side == "LONG":
        return entry_price * (1 + tp_pct)
    return entry_price * (1 - tp_pct)


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
    entry_id: str,
    side: str,
    entry_price: float,
    bar: pd.Series,
    params: FibbParams,
    tp_pct_override: Optional[float] = None,
) -> Tuple[float, str]:
    """回傳 (止盈價, 止盈通道欄位名；% 模式時欄位名為空字串)。"""
    if uses_channel_tp(entry_id, params):
        col = channel_tp_target_band(entry_id, side, params.channel_tp_offset)
        return channel_tp_level(entry_id, side, bar, params.channel_tp_offset), col
    return take_profit_price_pct(side, entry_price, params, tp_pct_override), ""


def resolve_entry_take_profit(
    entry_id: str,
    side: str,
    entry_price: float,
    bar: pd.Series,
    params: FibbParams,
    tp_pct_override: Optional[float] = None,
) -> Tuple[float, str]:
    """
    開倉當下止盈價。
    tp_mode=1：先固定 %（下一根 K 才跟 basis）；0/2 見 resolve_take_profit。
    """
    if params.tp_mode == 1:
        return take_profit_price_pct(side, entry_price, params, tp_pct_override), ""
    return resolve_take_profit(
        entry_id, side, entry_price, bar, params, tp_pct_override=tp_pct_override
    )


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


def resolve_regime_controls(curr: pd.Series, params: FibbParams) -> Dict[str, Any]:
    """Per-bar dynamic controls derived from 4H volatility/trend regime."""
    range_pct = curr.get("h4_range_pct")
    high_cut = curr.get("h4_range_pct_q_high")
    basis_slope = curr.get("h4_basis_slope")
    is_high_vol = False
    if (
        params.regime_enabled
        and pd.notna(range_pct)
        and pd.notna(high_cut)
        and float(range_pct) >= float(high_cut)
    ):
        is_high_vol = True
    trend = "flat"
    if pd.notna(basis_slope):
        if float(basis_slope) > 0:
            trend = "up"
        elif float(basis_slope) < 0:
            trend = "down"
    tp_pct = params.tp_pct
    max_hold = params.max_holding_hours
    if is_high_vol:
        tp_pct = params.tp_pct * max(float(params.regime_high_vol_tp_mult), 0.0)
        high_vol_hold = float(params.regime_high_vol_max_holding_hours)
        if high_vol_hold > 0:
            if max_hold > 0:
                max_hold = min(max_hold, high_vol_hold)
            else:
                max_hold = high_vol_hold
    return {
        "is_high_vol": is_high_vol,
        "trend": trend,
        "tp_pct": tp_pct,
        "max_holding_hours": max_hold,
        "h4_range_pct": float(range_pct) if pd.notna(range_pct) else None,
        "h4_range_pct_q_high": float(high_cut) if pd.notna(high_cut) else None,
    }


def should_block_entry_by_regime(
    entry_id: str, side: str, curr: pd.Series, params: FibbParams
) -> Tuple[bool, Optional[str]]:
    controls = resolve_regime_controls(curr, params)
    if not controls["is_high_vol"] or not params.regime_block_outer_countertrend:
        return False, None
    if entry_id not in {"T3 Short", "B3 Long"}:
        return False, None
    trend = controls["trend"]
    if trend == "up" and side == "SHORT":
        return True, "high_vol_countertrend_block"
    if trend == "down" and side == "LONG":
        return True, "high_vol_countertrend_block"
    return False, None


def _normalize_bar_ts(ts: pd.Timestamp) -> pd.Timestamp:
    bar_ts = pd.Timestamp(ts)
    if bar_ts.tzinfo is None:
        return bar_ts.tz_localize("UTC")
    return bar_ts.tz_convert("UTC")


def leg_hold_expired(
    leg: OpenLeg, bar_ts: pd.Timestamp, max_holding_hours: float
) -> bool:
    """True when leg has been open at least max_holding_hours (0 = disabled)."""
    if max_holding_hours <= 0:
        return False
    entry = _normalize_bar_ts(leg.entry_time)
    bar = _normalize_bar_ts(bar_ts)
    return (bar - entry).total_seconds() >= max_holding_hours * 3600.0


def try_time_stop_exit(
    leg: OpenLeg,
    bar_ts: pd.Timestamp,
    close_price: float,
    max_holding_hours: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Bar-close time stop when hold duration exceeds max_holding_hours."""
    if not leg_hold_expired(leg, bar_ts, max_holding_hours):
        return None, None
    return close_price, "TIME_STOP"


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
            outer_touch = _cross_short(curr, prev, outer_band)
        else:
            outer_touch = _cross_long(curr, prev, outer_band)
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
    if max_qty <= 0:
        return 0.0
    return max_qty


def leg_unrealized_gross_at_extreme(
    leg: OpenLeg, bar_low: float, bar_high: float
) -> float:
    """持倉中該根 K 最不利價格下的毛浮盈虧（不含手續費）。"""
    if leg.side == "LONG":
        return leg.qty * (bar_low - leg.entry_price)
    return leg.qty * (leg.entry_price - bar_high)


def mark_open_legs_unrealized(
    open_legs: Dict[str, OpenLeg], bar_low: float, bar_high: float
) -> float:
    """
    更新各 leg 持倉期間最差浮盈虧，並回傳當根所有未平倉 leg 合計最不利毛浮盈虧。
    """
    total = 0.0
    for leg in open_legs.values():
        u = leg_unrealized_gross_at_extreme(leg, bar_low, bar_high)
        total += u
        if u < leg.worst_unrealized_gross:
            leg.worst_unrealized_gross = u
    return total


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
    portfolio_worst_unrealized_gross = 0.0

    for i in range(warmup, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = curr["open_time"]
        if ts < start_ts:
            continue

        bar_high = float(curr["high"])
        bar_low = float(curr["low"])
        close_price = float(curr["close"])
        controls = resolve_regime_controls(curr, params)
        in_entry_range = ts <= end_ts

        refresh_channel_take_profits(open_legs, curr, params)

        if open_legs:
            bar_open_unrealized = mark_open_legs_unrealized(
                open_legs, bar_low, bar_high
            )
            if bar_open_unrealized < portfolio_worst_unrealized_gross:
                portfolio_worst_unrealized_gross = bar_open_unrealized

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
                    "time_stop_hit": reason == "TIME_STOP",
                    "channel_stop_hit": reason == "CHANNEL_STOP",
                    "gross_pnl": gross,
                    "fee": fee,
                    "net_pnl": net,
                    "net_return": net / notional if notional else 0.0,
                    "win": net > 0,
                    "holding_bars": max(1, i - leg.entry_bar_index),
                    "max_unrealized_gross": float(leg.worst_unrealized_gross),
                    "regime_is_high_vol": bool(controls["is_high_vol"]),
                    "regime_trend": controls["trend"],
                    "regime_h4_range_pct": controls["h4_range_pct"],
                    "regime_h4_range_pct_q_high": controls["h4_range_pct_q_high"],
                    "regime_tp_pct": float(controls["tp_pct"]),
                    "regime_max_holding_hours": float(controls["max_holding_hours"]),
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
        signals = detect_entry_signals(curr, prev, set(open_legs.keys()), params)
        for entry_id, side, qty, band in signals:
            if len(open_legs) >= params.max_open_legs:
                break
            blocked, _ = should_block_entry_by_regime(entry_id, side, curr, params)
            if blocked:
                continue
            qty = resolve_entry_qty(qty, close_price, equity, params)
            if qty <= 0:
                continue
            tp, tp_band = resolve_entry_take_profit(
                entry_id,
                side,
                close_price,
                curr,
                params,
                tp_pct_override=controls["tp_pct"],
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
                entry_bar_index=i,
                worst_unrealized_gross=0.0,
            )

        arm_deferred_channel_stops(open_legs, curr, prev, params)

    trades = pd.DataFrame(closed)
    if trades.empty:
        return trades
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    trades["cum_net_pnl"] = trades["net_pnl"].cumsum()
    trades["cum_peak"] = trades["cum_net_pnl"].cummax()
    trades["drawdown"] = trades["cum_net_pnl"] - trades["cum_peak"]
    trades.attrs["portfolio_max_unrealized_gross"] = float(
        portfolio_worst_unrealized_gross
    )
    return trades


def summarize_trades(
    trades: pd.DataFrame, *, bar_minutes: int = 15
) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "avg_holding_bars": None,
            "max_holding_bars": None,
            "avg_holding_minutes": None,
            "max_holding_minutes": None,
            "avg_max_unrealized_loss": None,
            "max_unrealized_loss": None,
            "portfolio_max_unrealized_loss": None,
        }
    wins = trades[trades["gross_pnl"] > 0]
    losses = trades[trades["gross_pnl"] <= 0]
    gross_profit = float(wins["gross_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["gross_pnl"].sum()) if not losses.empty else 0.0
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else None

    holding_bars = trades["holding_bars"] if "holding_bars" in trades.columns else None
    max_unrealized = (
        trades["max_unrealized_gross"] if "max_unrealized_gross" in trades.columns else None
    )
    avg_bars = float(holding_bars.mean()) if holding_bars is not None else None
    max_bars = int(holding_bars.max()) if holding_bars is not None else None
    avg_minutes = avg_bars * bar_minutes if avg_bars is not None else None
    max_minutes = max_bars * bar_minutes if max_bars is not None else None

    avg_mae = float(max_unrealized.mean()) if max_unrealized is not None else None
    worst_mae = float(max_unrealized.min()) if max_unrealized is not None else None
    portfolio_mae = trades.attrs.get("portfolio_max_unrealized_gross")

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
        "time_stop_hits": int(trades["time_stop_hit"].sum())
        if "time_stop_hit" in trades.columns
        else 0,
        "avg_holding_bars": avg_bars,
        "max_holding_bars": max_bars,
        "avg_holding_minutes": avg_minutes,
        "max_holding_minutes": max_minutes,
        "avg_max_unrealized_loss": avg_mae,
        "max_unrealized_loss": worst_mae,
        "portfolio_max_unrealized_loss": (
            float(portfolio_mae) if portfolio_mae is not None else None
        ),
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
