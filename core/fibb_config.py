"""FiBB 策略參數（對應 TradingView 腳本 inputs）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FibbParams:
    length: int = 20
    tp_pct: float = 0.005  # 0.5%
    sl_pct: float = 0.005
    fib_ratio_1: float = 1.618
    fib_ratio_2: float = 2.618
    fib_ratio_3: float = 4.236
    fee_rate: float = 0.0  # 與 TV 預設一致；可設 >0 模擬佣金
    max_open_legs: int = 6  # pyramiding
    initial_capital: float = 100_000.0  # 對應 strategy(..., initial_capital=100000)
    # 0 = 不檢查資金（舊行為）；>0 時依 equity×leverage 縮單（對應 TV 資金不足自動減量）
    leverage: float = 150.0
    # T2/B2：進場不設止損；觸及外層 T3/B3 後，才把止損設在當根 T2/B2 通道價
    # False = 恢復 Pine 全 leg 固定 % 止損
    use_deferred_channel_sl: bool = True
    # True = 止盈設在內層費波軌（T1→basis…）；False = 全 leg 用 tp_pct（延遲止損版預設）
    use_channel_tp: bool = False


# 止盈目標通道（空單往中軸/內軌，多單往中軸/內軌）
CHANNEL_TP_TARGET: dict[str, str] = {
    "T1 Short": "basis",
    "T2 Short": "top1",
    "T3 Short": "top2",
    "B1 Long": "basis",
    "B2 Long": "bott1",
    "B3 Long": "bott2",
}

# 內層 leg 觸及外層軌時，於內層通道價啟用止損（entry_id -> (outer_band, inner_band)）
DEFERRED_CHANNEL_SL: dict[str, tuple[str, str]] = {
    "T2 Short": ("top3", "top2"),
    "B2 Long": ("bott3", "bott2"),
}

# 進場 leg： (entry_id, side, qty_btc, band_key for short uses top*, long uses bott*)
SHORT_LEGS = (
    ("T1 Short", "SHORT", 0.01, "top1"),
    ("T2 Short", "SHORT", 0.02, "top2"),
    ("T3 Short", "SHORT", 0.03, "top3"),
)
LONG_LEGS = (
    ("B1 Long", "LONG", 0.01, "bott1"),
    ("B2 Long", "LONG", 0.02, "bott2"),
    ("B3 Long", "LONG", 0.03, "bott3"),
)
ALL_LEGS = SHORT_LEGS + LONG_LEGS

DEFAULT_PARAMS = FibbParams()
