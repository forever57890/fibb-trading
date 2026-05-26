"""Persist open legs between 15m bar runs (mirrors fibb_logic.OpenLeg)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fibb_trading.core.fibb_logic import OpenLeg


@dataclass
class LiveState:
    last_bar_time: Optional[str] = None
    open_legs: Dict[str, dict] = None  # type: ignore
    realized_pnl: float = 0.0
    trade_count: int = 0

    def __post_init__(self) -> None:
        if self.open_legs is None:
            self.open_legs = {}

    def to_dict(self) -> dict:
        return {
            "last_bar_time": self.last_bar_time,
            "open_legs": self.open_legs,
            "realized_pnl": self.realized_pnl,
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LiveState":
        return cls(
            last_bar_time=data.get("last_bar_time"),
            open_legs=dict(data.get("open_legs") or {}),
            realized_pnl=float(data.get("realized_pnl") or 0.0),
            trade_count=int(data.get("trade_count") or 0),
        )


def leg_to_dict(leg: OpenLeg) -> dict:
    d = asdict(leg)
    d["entry_time"] = pd.Timestamp(leg.entry_time).isoformat()
    return d


def leg_from_dict(d: dict) -> OpenLeg:
    return OpenLeg(
        entry_id=d["entry_id"],
        side=d["side"],
        qty=float(d["qty"]),
        entry_time=pd.Timestamp(d["entry_time"]),
        entry_price=float(d["entry_price"]),
        take_profit_price=float(d["take_profit_price"]),
        stop_loss_price=d.get("stop_loss_price"),
        band=d["band"],
        take_profit_band=d.get("take_profit_band") or "",
        sl_use_channel=bool(d.get("sl_use_channel")),
    )


def open_legs_objects(state: LiveState) -> Dict[str, OpenLeg]:
    return {k: leg_from_dict(v) for k, v in state.open_legs.items()}


def save_open_legs(state: LiveState, legs: Dict[str, OpenLeg]) -> None:
    state.open_legs = {k: leg_to_dict(v) for k, v in legs.items()}


def append_closed_trade(state: LiveState, record: dict) -> None:
    state.realized_pnl += float(record.get("net_pnl") or 0.0)
    state.trade_count += 1
