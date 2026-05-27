"""
Load FiBB strategy parameters and leg qty tables from environment / .env.

All keys use the FIBB_ prefix. Percent inputs (FIBB_TP_PCT, FIBB_SL_PCT) are in
human units (0.5 = 0.5%%), converted to decimals for FibbParams.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from fibb_trading.core.fibb_config import (
    DEFAULT_PARAMS,
    FibbParams,
    LONG_LEGS,
    SHORT_LEGS,
    TP_MODE_BASIS,
    TP_MODE_CHANNEL,
    TP_MODE_FIXED_PCT,
    normalize_channel_tp_offset,
    normalize_tp_mode,
    tp_mode_label,
)
from fibb_trading.env_loader import load_fibb_env

LegTuple = Tuple[Tuple[str, str, float, str], ...]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_pct_decimal(key: str, default_pct: float) -> float:
    """Env percent (e.g. 0.5) -> decimal (0.005)."""
    return _env_float(key, default_pct) / 100.0


def _load_tp_mode_from_env(default: int) -> int:
    """
    FIBB_TP_MODE: 0=fixed %, 1=basis, 2=channel tracks bar, 3=channel locked at entry.
    Legacy: FIBB_CHANNEL_TP=1 -> 2; FIBB_REPRICE_TP_TO_BASIS=1 -> 1 (if TP_MODE unset).
    """
    raw = os.getenv("FIBB_TP_MODE")
    if raw is not None and raw.strip() != "":
        return normalize_tp_mode(int(raw))
    if os.getenv("FIBB_CHANNEL_TP") is not None and _env_bool("FIBB_CHANNEL_TP", False):
        return TP_MODE_CHANNEL
    if os.getenv("FIBB_REPRICE_TP_TO_BASIS") is not None:
        return TP_MODE_BASIS if _env_bool("FIBB_REPRICE_TP_TO_BASIS", False) else TP_MODE_FIXED_PCT
    return normalize_tp_mode(default)


def load_fibb_params_from_env(*, reload_env: bool = True) -> FibbParams:
    if reload_env:
        load_fibb_env()
    d = DEFAULT_PARAMS
    return FibbParams(
        length=_env_int("FIBB_LENGTH", d.length),
        tp_pct=_env_pct_decimal("FIBB_TP_PCT", d.tp_pct * 100.0),
        sl_pct=_env_pct_decimal("FIBB_SL_PCT", d.sl_pct * 100.0),
        fib_ratio_1=_env_float("FIBB_FIB_RATIO_1", d.fib_ratio_1),
        fib_ratio_2=_env_float("FIBB_FIB_RATIO_2", d.fib_ratio_2),
        fib_ratio_3=_env_float("FIBB_FIB_RATIO_3", d.fib_ratio_3),
        fee_rate=_env_float("FIBB_FEE_RATE", d.fee_rate),
        max_open_legs=_env_int("FIBB_MAX_OPEN_LEGS", d.max_open_legs),
        initial_capital=_env_float("FIBB_INITIAL_CAPITAL", d.initial_capital),
        leverage=_env_float("FIBB_LEVERAGE", d.leverage),
        use_deferred_channel_sl=_env_bool("FIBB_DEFERRED_SL", d.use_deferred_channel_sl),
        tp_mode=_load_tp_mode_from_env(d.tp_mode),
        channel_tp_offset=normalize_channel_tp_offset(
            _env_int("FIBB_CHANNEL_TP_OFFSET", d.channel_tp_offset)
        ),
    )


def load_leg_tables_from_env(*, reload_env: bool = False) -> Tuple[LegTuple, LegTuple]:
    if reload_env:
        load_fibb_env()
    qty_t1 = _env_float("FIBB_QTY_T1", SHORT_LEGS[0][2])
    qty_t2 = _env_float("FIBB_QTY_T2", SHORT_LEGS[1][2])
    qty_t3 = _env_float("FIBB_QTY_T3", SHORT_LEGS[2][2])
    short_legs: LegTuple = (
        ("T1 Short", "SHORT", qty_t1, "top1"),
        ("T2 Short", "SHORT", qty_t2, "top2"),
        ("T3 Short", "SHORT", qty_t3, "top3"),
    )
    long_legs: LegTuple = (
        ("B1 Long", "LONG", qty_t1, "bott1"),
        ("B2 Long", "LONG", qty_t2, "bott2"),
        ("B3 Long", "LONG", qty_t3, "bott3"),
    )
    return short_legs, long_legs


def apply_leg_tables_to_config(short_legs: LegTuple, long_legs: LegTuple) -> None:
    """Update fibb_config module leg tuples (read via fibb_config.SHORT_LEGS at runtime)."""
    import fibb_trading.core.fibb_config as cfg

    cfg.SHORT_LEGS = short_legs
    cfg.LONG_LEGS = long_legs
    cfg.ALL_LEGS = short_legs + long_legs


def configure_strategy_from_env(*, reload_env: bool = True) -> FibbParams:
    """Load .env, build FibbParams, and apply leg qty tables to fibb_config."""
    params = load_fibb_params_from_env(reload_env=reload_env)
    short_legs, long_legs = load_leg_tables_from_env(reload_env=False)
    apply_leg_tables_to_config(short_legs, long_legs)
    return params


def indicator_history_bars(length: int) -> int:
    """
    How many 15m bars to request from Binance for indicator calculation.

    Needs: `length` for SMA/ATR, previous bar for touch-cross, + margin because
    the latest candle may still be forming (excluded by latest_closed_bar_index).
    """
    return max(int(length) + 5, 10)


def first_tradable_bar_index(length: int) -> int:
    """First bar index where basis/top1/bott1 exist on both curr and prev."""
    return max(int(length), 1)


def params_snapshot_dict(params: FibbParams) -> Dict[str, Any]:
    import fibb_trading.core.fibb_config as cfg

    return {
        "length": params.length,
        "tp_pct": params.tp_pct,
        "sl_pct": params.sl_pct,
        "fib_ratio_1": params.fib_ratio_1,
        "fib_ratio_2": params.fib_ratio_2,
        "fib_ratio_3": params.fib_ratio_3,
        "fee_rate": params.fee_rate,
        "max_open_legs": params.max_open_legs,
        "initial_capital": params.initial_capital,
        "leverage": params.leverage,
        "use_deferred_channel_sl": params.use_deferred_channel_sl,
        "tp_mode": params.tp_mode,
        "tp_mode_label": tp_mode_label(params.tp_mode),
        "channel_tp_offset": params.channel_tp_offset,
        "qty_t1": cfg.SHORT_LEGS[0][2],
        "qty_t2": cfg.SHORT_LEGS[1][2],
        "qty_t3": cfg.SHORT_LEGS[2][2],
        
        "indicator_history_bars": indicator_history_bars(params.length),
    }
