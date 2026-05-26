"""Binance USD-M Futures K 線（支援 15m 等週期）。"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests

BN_BASE_URL = "https://fapi.binance.com"

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def parse_time(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp())


def interval_to_ms(interval: str) -> int:
    key = interval.strip().lower()
    if key not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_MS)}")
    return INTERVAL_MS[key]


def fetch_binance_futures_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> pd.DataFrame:
    """分頁拉取 K 線；open_time 為 UTC。"""
    step_ms = interval_to_ms(interval)
    url = f"{BN_BASE_URL}/fapi/v1/klines"
    all_rows: list = []
    cursor = start_ms

    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        last_err = None
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                rows = r.json()
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Binance API failed: {last_err}") from last_err

        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        next_cursor = last_open_time + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    k = pd.DataFrame(all_rows, columns=cols).drop_duplicates("open_time")
    k["open_time"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k["close_time"] = pd.to_datetime(k["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    return k.sort_values("open_time").reset_index(drop=True)
