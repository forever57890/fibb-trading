"""
比對 TradingView 策略清單匯出與 Python trade_details.json。

用法:
  PYTHONPATH=/path/to/parent python3 -m fibb_trading.backtest.compare_tv \\
    --tv-csv ~/Downloads/FiBB_Backtest_BINANCE_BTCUSDT_2026-05-26.csv \\
    --py-json backtest/test_data/trade_details.json \\
    --tz-offset 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_tv_trades(csv_path: Path, tz_offset_hours: int) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    entries = raw[raw["類型"].str.contains("進場", na=False)].copy()
    exits = raw[raw["類型"].str.contains("出場", na=False)].copy()
    exits = exits.rename(
        columns={
            "交易 #": "trade_id",
            "淨損益 USDT": "net_pnl",
            "日期和時間": "exit_time_local",
            "價格 USDT": "exit_price",
            "訊號": "exit_signal",
        }
    )
    entries = entries.rename(
        columns={
            "交易 #": "trade_id",
            "訊號": "entry_id",
            "價格 USDT": "entry_price",
            "日期和時間": "entry_time_local",
            "大小（數量）": "qty",
        }
    )
    df = exits.merge(
        entries[["trade_id", "entry_id", "entry_time_local", "entry_price", "qty"]],
        on="trade_id",
    )
    df["entry_time"] = pd.to_datetime(df["entry_time_local"]) - pd.Timedelta(
        hours=tz_offset_hours
    )
    df["exit_time"] = pd.to_datetime(df["exit_time_local"]) - pd.Timedelta(
        hours=tz_offset_hours
    )
    df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
    df["exit_time"] = df["exit_time"].dt.tz_localize("UTC")
    df["bucket"] = df["entry_time"].dt.floor("15min")
    df["win"] = df["net_pnl"] > 0
    return df


def load_py_trades(json_path: Path) -> pd.DataFrame:
    df = pd.read_json(json_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["bucket"] = df["entry_time"].dt.floor("15min")
    df["win"] = df["net_pnl"] > 0
    return df


def load_klines(klines_path: Path) -> pd.DataFrame:
    kl = pd.read_json(klines_path)
    kl["open_time"] = pd.to_datetime(kl["open_time"])
    return kl.set_index("open_time").sort_index()


def _bar_fields(kl: pd.DataFrame, ts: pd.Timestamp) -> dict:
    key = pd.Timestamp(ts).floor("15min")
    if key not in kl.index:
        return {"bar_open_time": None, "bar_high": None, "bar_low": None, "bar_close": None}
    row = kl.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "bar_open_time": key.isoformat(),
        "bar_high": float(row["high"]),
        "bar_low": float(row["low"]),
        "bar_close": float(row["close"]),
    }


def _intrabar_hits(side: str, high: float, low: float, tp: float, sl: float) -> dict:
    if side == "LONG":
        tp_hit = high >= tp
        sl_hit = low <= sl
    else:
        tp_hit = low <= tp
        sl_hit = high >= sl
    return {
        "tp_touched": tp_hit,
        "sl_touched": sl_hit,
        "both_touched": tp_hit and sl_hit,
    }


def match_trades(tv: pd.DataFrame, py: pd.DataFrame) -> pd.DataFrame:
    """依 entry_id + 15m bucket 配對；多對多時取進場價最接近的一筆。"""
    tv_cols = [
        "trade_id",
        "entry_id",
        "bucket",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "qty",
        "net_pnl",
        "win",
        "exit_signal",
        "entry_time_local",
        "exit_time_local",
    ]
    j = py.merge(tv[tv_cols], on=["entry_id", "bucket"], how="inner", suffixes=("_py", "_tv"))
    if j.empty:
        return j
    j["entry_price_diff"] = (j["entry_price_py"] - j["entry_price_tv"]).abs()
    j = j.sort_values(["entry_id", "bucket", "entry_price_diff"])
    j = j.drop_duplicates(subset=["entry_id", "bucket"], keep="first")
    return j


def export_disagreements(
    tv: pd.DataFrame,
    py: pd.DataFrame,
    *,
    klines: pd.DataFrame | None,
    tz_offset_hours: int,
    limit: int = 20,
    out_path: Path,
) -> pd.DataFrame:
    matched = match_trades(tv, py)
    dis = matched[matched["win_py"] != matched["win_tv"]].copy()
    dis["pnl_diff"] = dis["net_pnl_py"] - dis["net_pnl_tv"]
    dis = dis.reindex(dis["pnl_diff"].abs().sort_values(ascending=False).index)
    if limit > 0:
        dis = dis.head(limit)

    rows: list[dict] = []
    for i, r in dis.iterrows():
        row: dict = {
            "rank": len(rows) + 1,
            "entry_id": r["entry_id"],
            "side": r["side"],
            "entry_time_utc": r["entry_time_py"].isoformat(),
            "entry_time_tv_local": r["entry_time_local"],
            "exit_time_utc_py": r["exit_time_py"].isoformat(),
            "exit_time_tv_local": r["exit_time_local"],
            "entry_price_py": r["entry_price_py"],
            "entry_price_tv": r["entry_price_tv"],
            "exit_price_py": r["exit_price_py"],
            "exit_price_tv": r["exit_price_tv"],
            "qty_py": r["qty_py"],
            "qty_tv": r["qty_tv"],
            "net_pnl_py": r["net_pnl_py"],
            "net_pnl_tv": r["net_pnl_tv"],
            "win_py": bool(r["win_py"]),
            "win_tv": bool(r["win_tv"]),
            "exit_reason_py": r["exit_reason"],
            "exit_signal_tv": r["exit_signal"],
            "take_profit_py": r["take_profit_price"],
            "stop_loss_py": r["stop_loss_price"],
            "tv_trade_id": int(r["trade_id"]),
        }
        if klines is not None:
            entry_bar = _bar_fields(klines, r["entry_time_py"])
            exit_bar = _bar_fields(klines, r["exit_time_py"])
            row.update({f"entry_{k}": v for k, v in entry_bar.items()})
            row.update({f"exit_{k}": v for k, v in exit_bar.items()})
            if exit_bar["bar_high"] is not None:
                hits = _intrabar_hits(
                    r["side"],
                    exit_bar["bar_high"],
                    exit_bar["bar_low"],
                    r["take_profit_price"],
                    r["stop_loss_price"],
                )
                row.update(hits)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def compare(tv: pd.DataFrame, py: pd.DataFrame) -> dict:
    tv_wr = float(tv["win"].mean())
    py_wr = float(py["win"].mean())
    tv_be = int((tv["net_pnl"] == 0).sum())

    matched = match_trades(tv, py)
    j_left = py.merge(
        tv[["trade_id", "entry_id", "bucket", "net_pnl"]].drop_duplicates(
            subset=["entry_id", "bucket"]
        ),
        on=["entry_id", "bucket"],
        how="left",
        suffixes=("_py", "_tv"),
    )
    agree = float((matched["win_py"] == matched["win_tv"]).mean()) if len(matched) else 0.0

    pine_qty = {"T1 Short": 1, "T2 Short": 2, "T3 Short": 3, "B1 Long": 1, "B2 Long": 2, "B3 Long": 3}
    tv["pine_qty"] = tv["entry_id"].map(pine_qty)
    undersized = int((tv["qty"] < tv["pine_qty"]).sum())

    return {
        "tv_trades": len(tv),
        "py_trades": len(py),
        "tv_win_rate": tv_wr,
        "py_win_rate": py_wr,
        "tv_breakeven": tv_be,
        "matched_by_id_and_15m_bar": int(len(matched)),
        "py_unmatched": int(j_left["net_pnl_tv"].isna().sum()),
        "win_disagreements": int((matched["win_py"] != matched["win_tv"]).sum()) if len(matched) else 0,
        "win_agreement_on_matched": agree,
        "tv_qty_below_pine_script": undersized,
        "mean_entry_price_diff_on_matched": float(
            (matched["entry_price_py"] - matched["entry_price_tv"]).abs().mean()
        )
        if len(matched)
        else None,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare TradingView vs Python backtest trades")
    p.add_argument("--tv-csv", type=Path, required=True)
    p.add_argument("--py-json", type=Path, default=Path("backtest/test_data/trade_details.json"))
    p.add_argument(
        "--klines",
        type=Path,
        default=Path("backtest/test_data/binance_btcusdt_15m_klines.json"),
        help="K 線 JSON（供匯出平倉當根 high/low）",
    )
    p.add_argument(
        "--tz-offset",
        type=int,
        default=8,
        help="TradingView 匯出時間為圖表時區時，減去的小時數轉 UTC（台北 UTC+8 用 8）",
    )
    p.add_argument(
        "--export-disagreements",
        action="store_true",
        help="匯出輸贏不一致的交易明細 CSV",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("backtest/test_data/tv_py_disagreements.csv"),
    )
    p.add_argument("--limit", type=int, default=20, help="最多匯出幾筆（0=全部）")
    args = p.parse_args(argv)

    tv = load_tv_trades(args.tv_csv, args.tz_offset)
    py = load_py_trades(args.py_json)
    report = compare(tv, py)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.export_disagreements:
        klines = load_klines(args.klines) if args.klines.exists() else None
        out = export_disagreements(
            tv,
            py,
            klines=klines,
            tz_offset_hours=args.tz_offset,
            limit=args.limit,
            out_path=args.out,
        )
        print(f"\nExported {len(out)} disagreement rows -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
