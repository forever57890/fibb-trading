"""Helpers for realtime intrabar FiBB trading (forming 15m candle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from fibb_trading.core.fibb_config import FibbParams
from fibb_trading.core.fibb_env import first_tradable_bar_index
from fibb_trading.trade.fibb_live_engine import latest_closed_bar_index


@dataclass
class FormingBar:
    open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_kline(cls, k: dict) -> "FormingBar":
        return cls(
            open_time=pd.Timestamp(int(k["t"]), unit="ms", tz="UTC"),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
        )

    @classmethod
    def from_price(cls, open_time: pd.Timestamp, price: float) -> "FormingBar":
        return cls(
            open_time=open_time,
            open=price,
            high=price,
            low=price,
            close=price,
        )

    def update_price(self, price: float) -> None:
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)

    def merge_kline(self, k: dict) -> None:
        self.open = float(k["o"])
        self.high = max(self.high, float(k["h"]))
        self.low = min(self.low, float(k["l"]))
        self.close = float(k["c"])


def current_bar_open_time(now: pd.Timestamp, interval_minutes: int) -> pd.Timestamp:
    now = pd.Timestamp(now).tz_convert("UTC")
    minute = (now.minute // interval_minutes) * interval_minutes
    return now.replace(minute=minute, second=0, microsecond=0, nanosecond=0)


def build_curr_prev(
    df: pd.DataFrame,
    forming: FormingBar,
    params: FibbParams,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build (curr, prev) for intrabar signal checks.

    Channel columns come from the last fully computed row; OHLC from the forming bar.
    """
    if len(df) < 2:
        raise ValueError("need at least 2 klines")

    closed_idx = latest_closed_bar_index(df)
    last_row = df.iloc[-1]
    last_open = pd.Timestamp(last_row["open_time"])

    if forming.open_time > last_open:
        prev = df.iloc[closed_idx]
        curr = prev.copy()
    elif forming.open_time == last_open:
        prev = df.iloc[closed_idx - 1] if closed_idx >= 1 else df.iloc[0]
        curr = last_row.copy()
    else:
        prev = df.iloc[closed_idx - 1] if closed_idx >= 1 else df.iloc[0]
        curr = df.iloc[closed_idx].copy()

    curr["open_time"] = forming.open_time
    curr["open"] = forming.open
    curr["high"] = forming.high
    curr["low"] = forming.low
    curr["close"] = forming.close
    return curr, prev


def history_ready(df: pd.DataFrame, params: FibbParams) -> bool:
    if len(df) < 2:
        return False
    closed_idx = latest_closed_bar_index(df)
    if closed_idx < first_tradable_bar_index(params.length):
        return False
    row = df.iloc[closed_idx]
    for col in ("basis", "top1", "bott1"):
        if col not in df.columns or pd.isna(row.get(col)):
            return False
    return True
