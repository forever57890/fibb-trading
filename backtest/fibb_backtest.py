import path_setup  # noqa: F401, E402

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from fibb_trading.backtest.backtest_io import ensure_test_data_dir
from fibb_trading.core.data_fetch import (
    fetch_binance_futures_klines,
    interval_to_minutes,
    interval_to_ms,
    parse_time,
)
from fibb_trading.core.fibb_config import (
    FibbParams,
    normalize_channel_tp_offset,
    normalize_tp_mode,
    normalize_trade_sides,
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
DEFAULT_START = "2025-05-26 00:00:00"
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


def summarize_trades_by_regime(
    trades: pd.DataFrame, *, bar_minutes: int = 15
) -> dict:
    if trades.empty or "regime_is_high_vol" not in trades.columns:
        return {}
    out = {}
    groups = {
        "high_vol": trades[trades["regime_is_high_vol"] == True],  # noqa: E712
        "non_high_vol": trades[trades["regime_is_high_vol"] == False],  # noqa: E712
    }
    for key, df_g in groups.items():
        s = summarize_trades(df_g, bar_minutes=bar_minutes)
        out[key] = {
            "total_trades": s["total_trades"],
            "win_rate": s["win_rate"],
            "profit_factor": s["profit_factor"],
            "max_drawdown": s["max_drawdown"],
            "net_profit": s["net_profit"],
            "time_stop_hits": s.get("time_stop_hits", 0),
            "avg_holding_minutes": s["avg_holding_minutes"],
        }
    return out


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
    p.add_argument(
        "--max-holding-hours",
        type=float,
        default=env_p.max_holding_hours,
        help="Force close after N hours at bar close (0=off, env FIBB_MAX_HOLDING_HOURS)",
    )
    p.add_argument(
        "--regime-enabled",
        type=int,
        choices=[0, 1],
        default=1 if env_p.regime_enabled else 0,
        help="Enable 4H volatility regime controls (0/1)",
    )
    p.add_argument(
        "--regime-h4-lookback",
        type=int,
        default=env_p.regime_h4_lookback,
        help="4H range percentile lookback bars",
    )
    p.add_argument(
        "--regime-h4-high-vol-quantile",
        type=float,
        default=env_p.regime_h4_high_vol_quantile,
        help="High-vol threshold quantile (0~1)",
    )
    p.add_argument(
        "--regime-high-vol-tp-mult",
        type=float,
        default=env_p.regime_high_vol_tp_mult,
        help="TP pct multiplier under high volatility",
    )
    p.add_argument(
        "--regime-high-vol-max-holding-hours",
        type=float,
        default=env_p.regime_high_vol_max_holding_hours,
        help="Max hold hours under high volatility (0=keep base max hold)",
    )
    p.add_argument(
        "--regime-block-outer-countertrend",
        type=int,
        choices=[0, 1],
        default=1 if env_p.regime_block_outer_countertrend else 0,
        help="Under high volatility, block countertrend outer legs (T3/B3)",
    )
    p.add_argument(
        "--trade-sides",
        choices=["both", "long", "short"],
        default=env_p.trade_sides,
        help="Allowed entry directions: both, long only, or short only",
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

    print(
        f"Params: length={params.length} tp_mode={params.tp_mode} "
        f"deferred_sl={params.use_deferred_channel_sl} "
        f"max_holding_hours={params.max_holding_hours} "
        f"regime_enabled={params.regime_enabled} "
        f"trade_sides={params.trade_sides}"
    )
    print(f"Fetching {symbol} {interval} klines: {fetch_start_ms} -> {end_ms}")
    klines = fetch_binance_futures_klines(symbol, interval, fetch_start_ms, end_ms)
    klines.to_json(
        test_data_dir / f"binance_{symbol.lower()}_{interval}_klines.json",
        orient="records",
        date_format="iso",
        indent=2,
    )

    trades = run_fibb_backtest(klines, start_ts, end_ts, params)
    bar_minutes = interval_to_minutes(interval)
    summary = summarize_trades(trades, bar_minutes=bar_minutes)
    regime_summary = summarize_trades_by_regime(trades, bar_minutes=bar_minutes)

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
    (test_data_dir / "regime_summary.json").write_text(
        json.dumps(regime_summary, indent=2),
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
    print(test_data_dir / "regime_summary.json")
    print(test_data_dir / "leg_summary.json")
    for c in charts:
        print(c)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    if summary.get("total_trades", 0) > 0:
        print("\n持倉與浮虧:")
        print(
            f"  平均持倉: {summary['avg_holding_bars']:.2f} 根 K "
            f"({summary['avg_holding_minutes']:.0f} 分鐘)"
        )
        print(
            f"  最長持倉: {summary['max_holding_bars']} 根 K "
            f"({summary['max_holding_minutes']} 分鐘)"
        )
        print(
            f"  單筆最大未平倉毛虧損: {summary['max_unrealized_loss']:.4f} USDT "
            f"(各筆持倉期間最差，平均 {summary['avg_max_unrealized_loss']:.4f})"
        )
        print(
            f"  組合最大未平倉毛虧損: {summary['portfolio_max_unrealized_loss']:.4f} USDT "
            f"(同時持倉合計最不利)"
        )
        if summary.get("time_stop_hits") is not None:
            print(f"  持倉逾時平倉 (TIME_STOP): {summary['time_stop_hits']} 筆")
    if regime_summary:
        print("\nRegime 分組成效:")
        for key, s in regime_summary.items():
            print(
                f"  {key}: trades={s['total_trades']} win_rate={s['win_rate']} "
                f"pf={s['profit_factor']} mdd={s['max_drawdown']} "
                f"time_stop={s['time_stop_hits']}"
            )


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
        max_holding_hours=cli.max_holding_hours,
        regime_enabled=bool(cli.regime_enabled),
        regime_h4_lookback=cli.regime_h4_lookback,
        regime_h4_high_vol_quantile=cli.regime_h4_high_vol_quantile,
        regime_high_vol_tp_mult=cli.regime_high_vol_tp_mult,
        regime_high_vol_max_holding_hours=cli.regime_high_vol_max_holding_hours,
        regime_block_outer_countertrend=bool(cli.regime_block_outer_countertrend),
        trade_sides=normalize_trade_sides(cli.trade_sides),
    )
    main(
        start=cli.start,
        end=cli.end,
        symbol=cli.symbol,
        interval=cli.interval,
        params=params,
        output_dir=Path(cli.output_dir) if cli.output_dir else None,
    )
