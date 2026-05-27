"""
FiBB 策略參數（對應 TradingView 腳本 inputs）。

實盤與回測預設從專案根目錄 `.env` 載入（見 `core/fibb_env.py`、`.env.example`）。
此檔的 dataclass 預設值僅在環境變數未設定時使用。
"""

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
    # 止盈模式：0=固定 %；1=每根 K 跟 basis；2=通道止盈隨 K 漂移；3=通道止盈開倉鎖定
    tp_mode: int = 0
    # tp_mode 2/3：進場帶沿 CHANNEL_LADDER 向中線偏移格數（預設 2 → B3 止盈在 B1）
    channel_tp_offset: int = 2


TP_MODE_FIXED_PCT = 0
TP_MODE_BASIS = 1
TP_MODE_CHANNEL = 2
TP_MODE_CHANNEL_FIXED = 3


def normalize_tp_mode(mode: int) -> int:
    allowed = (
        TP_MODE_FIXED_PCT,
        TP_MODE_BASIS,
        TP_MODE_CHANNEL,
        TP_MODE_CHANNEL_FIXED,
    )
    if mode not in allowed:
        raise ValueError(f"tp_mode must be 0, 1, 2, or 3, got {mode}")
    return int(mode)


def tp_mode_label(mode: int) -> str:
    return {
        TP_MODE_FIXED_PCT: "fixed_pct",
        TP_MODE_BASIS: "basis",
        TP_MODE_CHANNEL: "channel",
        TP_MODE_CHANNEL_FIXED: "channel_fixed",
    }[normalize_tp_mode(mode)]


def tp_mode_uses_channel(params: FibbParams) -> bool:
    return params.tp_mode in (TP_MODE_CHANNEL, TP_MODE_CHANNEL_FIXED)


def tp_mode_channel_tracks_bar(params: FibbParams) -> bool:
    """True = 每根 K 依通道重算止盈價（並在實盤重掛 TP）。"""
    return params.tp_mode == TP_MODE_CHANNEL


# 通道止盈：沿價格階梯向中線移動 CHANNEL_TP_OFFSET 格（預設 2）
# 例：B3→B1、B1→T1、T3→T1、T1→B1
CHANNEL_LADDER: tuple[str, ...] = (
    "bott3",
    "bott2",
    "bott1",
    "basis",
    "top1",
    "top2",
    "top3",
)
CHANNEL_TP_ENTRY_INDEX: dict[str, int] = {
    "B3 Long": 0,
    "B2 Long": 1,
    "B1 Long": 2,
    "T1 Short": 4,
    "T2 Short": 5,
    "T3 Short": 6,
}
def normalize_channel_tp_offset(offset: int) -> int:
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"channel_tp_offset must be >= 0, got {offset}")
    return offset


def channel_tp_target_band(entry_id: str, side: str, channel_tp_offset: int) -> str:
    """
    依進場 leg 回傳止盈所跟隨的通道欄位名（tp_mode 2/3 共用）。
    多單沿階梯向上（往中線/對側）偏移 channel_tp_offset 格；空單向下偏移。
    """
    channel_tp_offset = normalize_channel_tp_offset(channel_tp_offset)
    if entry_id not in CHANNEL_TP_ENTRY_INDEX:
        raise KeyError(f"Unknown entry_id for channel TP: {entry_id}")
    idx = CHANNEL_TP_ENTRY_INDEX[entry_id]
    step = channel_tp_offset if side == "LONG" else -channel_tp_offset
    tp_idx = idx + step
    if tp_idx < 0 or tp_idx >= len(CHANNEL_LADDER):
        raise ValueError(
            f"Channel TP index out of range for {entry_id} ({side}): "
            f"entry_idx={idx} step={step} channel_tp_offset={channel_tp_offset}"
        )
    return CHANNEL_LADDER[tp_idx]

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
