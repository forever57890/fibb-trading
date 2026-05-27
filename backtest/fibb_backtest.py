import path_setup  # noqa: F401, E402

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from fibb_trading.backtest.backtest_io import ensure_test_data_dir
from fibb_trading.core.data_fetch import fetch_binance_futures_klines, interval_to_ms, parse_time
from fibb_trading.core.fibb_config import (
    FibbParams,
    normalize_channel_tp_offset,
    normalize_tp_mode,
)
from fibb_trading.core.fibb_env import configure_strategy_from_env, indicator_history_bars
from fibb_trading.env_loader import load_fibb_env
from fibb_trading.core.fibb_logic import (
    compute_fibb_channels,
    leg_summary,
    run_fibb_backtest,
    summarize_trades,
)

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
_BACKTEST_ROOT = Path(__file__).resolve().parent

# 與先前延遲止損回測一致（約 2026-02-24～2026-05-26，本機時區）
DEFAULT_START = "2025-01-01 00:00:00"
DEFAULT_END = "2026-05-26 23:59:59"


def resolve_period(
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> tuple[int, int]:
    if start is None:
        start = parse_time(DEFAULT_START)
    if end is None:
        end = parse_time(DEFAULT_END)
    return start, end


def plot_results(klines: pd.DataFrame, trades: pd.DataFrame, out_dir: Path) -> list:
    if trades.empty:
        return []
    test_data_dir = ensure_test_data_dir(out_dir / "test_data")
    outputs = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(trades["exit_time"], trades["cum_net_pnl"], label="Cumulative net PnL")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_title("Equity Curve")
    axes[0].legend()

    axes[1].plot(trades["exit_time"], trades["drawdown"], color="tab:red")
    axes[1].set_title("Drawdown")
    fig.tight_layout()
    p1 = test_data_dir / "equity_drawdown.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    outputs.append(p1)

    sample = klines.iloc[-500:].copy() if len(klines) > 500 else klines
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sample["open_time"], sample["close"], color="black", linewidth=0.8, label="Close")
    for col, alpha in [("top3", 0.3), ("top2", 0.5), ("top1", 0.7)]:
        if col in sample.columns:
            ax.plot(sample["open_time"], sample[col], linewidth=0.6, alpha=alpha)
    for col, alpha in [("bott1", 0.7), ("bott2", 0.5), ("bott3", 0.3)]:
        if col in sample.columns:
            ax.plot(sample["open_time"], sample[col], linewidth=0.6, alpha=alpha)
    long_t = trades[trades["side"] == "LONG"]
    short_t = trades[trades["side"] == "SHORT"]
    ax.scatter(long_t["entry_time"], long_t["entry_price"], marker="^", color="green", s=20)
    ax.scatter(short_t["entry_time"], short_t["entry_price"], marker="v", color="red", s=20)
    ax.set_title("FiBB channels (last bars) + entries")
    fig.tight_layout()
    p2 = test_data_dir / "fibb_channels_trades.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    outputs.append(p2)
    return outputs


def parse_args(argv=None):
    load_fibb_env()
    env_p = configure_strategy_from_env(reload_env=False)

    p = argparse.ArgumentParser(description="FiBB 15m BTC layer strategy backtest")
    p.add_argument("--start", type=parse_time, help=f"UTC start, e.g. {DEFAULT_START}")
    p.add_argument("--end", type=parse_time, help=f"UTC end, e.g. {DEFAULT_END}")
    p.add_argument("--symbol", default=os.getenv("FIBB_SYMBOL", SYMBOL))
    p.add_argument("--interval", default=os.getenv("FIBB_INTERVAL", INTERVAL))
    p.add_argument("--length", type=int, default=env_p.length)
    p.add_argument(
        "--tp-pct",
        type=float,
        default=env_p.tp_pct * 100.0,
        help="Take profit %% when --pct-tp (e.g. 0.5)",
    )
    p.add_argument(
        "--sl-pct",
        type=float,
        default=env_p.sl_pct * 100.0,
        help="Stop loss %% when --pct-tp",
    )
    p.add_argument(
        "--tp-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="TP: 0=fixed %%, 1=basis/bar, 2=channel/bar, 3=channel locked at entry",
    )
    p.add_argument(
        "--channel-tp-offset",
        type=int,
        default=None,
        help="Steps toward center for tp_mode 2/3 (default from FIBB_CHANNEL_TP_OFFSET)",
    )
    p.add_argument(
        "--channel-tp",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy -> --tp-mode 2
    )
    p.add_argument(
        "--no-reprice-tp-to-basis",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy -> --tp-mode 0
    )
    p.add_argument(
        "--pct-tp",
        action="store_true",
        help=argparse.SUPPRESS,  # 相容舊參數；等同未加 --channel-tp
    )
    p.add_argument(
        "--fee-rate",
        type=float,
        default=env_p.fee_rate,
        help="Commission rate per side (0 = match TradingView default)",
    )
    p.add_argument("--initial-capital", type=float, default=env_p.initial_capital)
    p.add_argument(
        "--leverage",
        type=float,
        default=env_p.leverage,
        help="Max notional = equity × leverage (match TV strategy properties)",
    )
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def main(
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    params: Optional[FibbParams] = None,
    output_dir: Optional[Path] = None,
):
    start, end = resolve_period(start, end)
    out_dir = Path(output_dir) if output_dir else _BACKTEST_ROOT
    test_data_dir = ensure_test_data_dir(out_dir / "test_data")

    if params is None:
        params = configure_strategy_from_env(reload_env=True)

    history_bars = indicator_history_bars(params.length)
    start_ts = pd.to_datetime(start, unit="s", utc=True)
    end_ts = pd.to_datetime(end, unit="s", utc=True)
    step = interval_to_ms(interval)
    fetch_start_ms = int(
        (start_ts - pd.Timedelta(milliseconds=step * history_bars)).timestamp() * 1000
    )
    end_ms = int(end_ts.timestamp() * 1000)

    print(f"Fetching {symbol} {interval} klines: {fetch_start_ms} -> {end_ms}")
    klines = fetch_binance_futures_klines(symbol, interval, fetch_start_ms, end_ms)
    klines.to_json(
        test_data_dir / f"binance_{symbol.lower()}_{interval}_klines.json",
        orient="records",
        date_format="iso",
        indent=2,
    )

    trades = run_fibb_backtest(klines, start_ts, end_ts, params)
    summary = summarize_trades(trades)

    trades.to_json(
        test_data_dir / "trade_details.json",
        orient="records",
        date_format="iso",
        indent=2,
    )
    (test_data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    leg_summary(trades).to_json(
        test_data_dir / "leg_summary.json",
        orient="records",
        date_format="iso",
        indent=2,
    )

    channels = compute_fibb_channels(klines, params)
    charts = plot_results(channels, trades, out_dir)

    print("Saved:")
    print(test_data_dir / "trade_details.json")
    print(test_data_dir / "summary.json")
    print(test_data_dir / "leg_summary.json")
    for c in charts:
        print(c)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def _resolve_cli_tp_mode(cli, env_p: FibbParams) -> int:
    if cli.tp_mode is not None:
        return normalize_tp_mode(cli.tp_mode)
    if cli.channel_tp and not cli.pct_tp:
        return 2
    if cli.no_reprice_tp_to_basis:
        return 0
    return env_p.tp_mode


if __name__ == "__main__":
    cli = parse_args()
    env_p = configure_strategy_from_env(reload_env=False)
    params = FibbParams(
        length=cli.length,
        tp_pct=cli.tp_pct / 100.0,
        sl_pct=cli.sl_pct / 100.0,
        fib_ratio_1=env_p.fib_ratio_1,
        fib_ratio_2=env_p.fib_ratio_2,
        fib_ratio_3=env_p.fib_ratio_3,
        fee_rate=cli.fee_rate,
        max_open_legs=env_p.max_open_legs,
        initial_capital=cli.initial_capital,
        leverage=cli.leverage,
        use_deferred_channel_sl=env_p.use_deferred_channel_sl,
        tp_mode=_resolve_cli_tp_mode(cli, env_p),
        channel_tp_offset=(
            normalize_channel_tp_offset(cli.channel_tp_offset)
            if cli.channel_tp_offset is not None
            else env_p.channel_tp_offset
        ),
    )
    main(
        start=cli.start,
        end=cli.end,
        symbol=cli.symbol,
        interval=cli.interval,
        params=params,
        output_dir=Path(cli.output_dir) if cli.output_dir else None,
    )
