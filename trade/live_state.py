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
    # Realtime intrabar: entry_ids opened on the current forming 15m bar
    intrabar_bar_time: Optional[str] = None
    intrabar_opened: Optional[List[str]] = None
    last_finalize_bar_time: Optional[str] = None
    # bar_time_iso -> entry_ids already claimed/opened (prevents duplicate leg on same bar)
    bar_entry_claims: Optional[Dict[str, List[str]]] = None

    def __post_init__(self) -> None:
        if self.open_legs is None:
            self.open_legs = {}
        if self.intrabar_opened is None:
            self.intrabar_opened = []
        if self.bar_entry_claims is None:
            self.bar_entry_claims = {}

    def to_dict(self) -> dict:
        return {
            "last_bar_time": self.last_bar_time,
            "open_legs": self.open_legs,
            "realized_pnl": self.realized_pnl,
            "trade_count": self.trade_count,
            "intrabar_bar_time": self.intrabar_bar_time,
            "intrabar_opened": list(self.intrabar_opened or []),
            "last_finalize_bar_time": self.last_finalize_bar_time,
            "bar_entry_claims": dict(self.bar_entry_claims or {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LiveState":
        raw_claims = data.get("bar_entry_claims") or {}
        claims = {
            str(k): list(v) if isinstance(v, (list, tuple)) else []
            for k, v in raw_claims.items()
        }
        return cls(
            last_bar_time=data.get("last_bar_time"),
            open_legs=dict(data.get("open_legs") or {}),
            realized_pnl=float(data.get("realized_pnl") or 0.0),
            trade_count=int(data.get("trade_count") or 0),
            intrabar_bar_time=data.get("intrabar_bar_time"),
            intrabar_opened=list(data.get("intrabar_opened") or []),
            last_finalize_bar_time=data.get("last_finalize_bar_time"),
            bar_entry_claims=claims,
        )


def leg_to_dict(leg: OpenLeg, *, tp_algo_id: Any = None) -> dict:
    d = asdict(leg)
    d["entry_time"] = pd.Timestamp(leg.entry_time).isoformat()
    d["tp_algo_id"] = tp_algo_id
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


def get_tp_algo_id(state: "LiveState", entry_id: str) -> Any:
    """Return the stored TP algo order ID for a leg (None if not set)."""
    return (state.open_legs.get(entry_id) or {}).get("tp_algo_id")


def open_legs_objects(state: LiveState) -> Dict[str, OpenLeg]:
    return {k: leg_from_dict(v) for k, v in state.open_legs.items()}


def save_open_legs(
    state: LiveState,
    legs: Dict[str, OpenLeg],
    *,
    tp_algo_ids: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist legs; tp_algo_ids maps entry_id -> algoId (merged with existing if not provided)."""
    saved: Dict[str, dict] = {}
    for k, leg in legs.items():
        existing = state.open_legs.get(k) or {}
        if tp_algo_ids is not None:
            algo_id = tp_algo_ids.get(k)
        else:
            algo_id = existing.get("tp_algo_id")
        saved[k] = leg_to_dict(leg, tp_algo_id=algo_id)
    state.open_legs = saved


def append_closed_trade(state: LiveState, record: dict) -> None:
    state.realized_pnl += float(record.get("net_pnl") or 0.0)
    state.trade_count += 1


def bar_entry_claim_ids(state: LiveState, bar_time_iso: str) -> set:
    claims = state.bar_entry_claims or {}
    return set(claims.get(bar_time_iso) or [])


def claim_bar_entry(state: LiveState, bar_time_iso: str, entry_id: str) -> bool:
    """
    Reserve entry_id for this bar (persist before exchange call).

    Returns False if this leg was already claimed on this bar.
    """
    claims = dict(state.bar_entry_claims or {})
    ids = list(claims.get(bar_time_iso) or [])
    if entry_id in ids:
        return False
    ids.append(entry_id)
    claims[bar_time_iso] = ids
    state.bar_entry_claims = claims
    return True


def prune_bar_entry_claims(state: LiveState, *, keep_last: int = 32) -> None:
    claims = state.bar_entry_claims or {}
    if len(claims) <= keep_last:
        return
    keys = sorted(claims.keys())
    state.bar_entry_claims = {k: claims[k] for k in keys[-keep_last:]}
