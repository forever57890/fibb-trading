import unittest
from unittest.mock import patch

import pandas as pd

from fibb_trading.core.fibb_config import (
    FibbParams,
    entry_legs_for_trade,
    side_entry_allowed,
)
from fibb_trading.core.fibb_logic import (
    OpenLeg,
    arm_deferred_channel_stops,
    bracket_prices,
    compute_fibb_channels,
    compute_rma,
    compute_true_range,
    detect_entry_signals,
    leg_hold_expired,
    leg_unrealized_gross_at_extreme,
    mark_open_legs_unrealized,
    refresh_channel_take_profits,
    refresh_reprice_tp_to_basis,
    resolve_entry_qty,
    resolve_take_profit,
    run_fibb_backtest,
    summarize_trades,
    take_profit_price_pct,
    try_exit_leg,
    try_time_stop_exit,
    uses_deferred_channel_sl,
)


class TestFibbLogic(unittest.TestCase):
    DEFAULT_BACKTEST_START = "2024-06-01"
    DEFAULT_BACKTEST_END = "2024-06-02"

    def _ts(self, value: str | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        return ts if ts.tz is not None else ts.tz_localize("UTC")

    def _backtest_window(
        self,
        klines: pd.DataFrame,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if start is None:
            start_ts = klines["open_time"].iloc[0]
        else:
            start_ts = self._ts(start)
        if end is None:
            end_ts = klines["open_time"].iloc[-1]
        else:
            end_ts = self._ts(end)
        return start_ts, end_ts

    def _run_backtest(
        self,
        klines: pd.DataFrame | None = None,
        *,
        n_bars: int = 50,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        params: FibbParams | None = None,
    ) -> pd.DataFrame:
        bars = klines if klines is not None else self._sample_bars(n_bars)
        if start is None and end is None:
            start_ts = self._ts(self.DEFAULT_BACKTEST_START)
            end_ts = self._ts(self.DEFAULT_BACKTEST_END)
        else:
            start_ts, end_ts = self._backtest_window(bars, start=start, end=end)
        return run_fibb_backtest(
            bars,
            start_ts,
            end_ts,
            params or FibbParams(length=5, tp_pct=0.01, sl_pct=0.01),
        )

    def _sample_bars(self, n: int = 40) -> pd.DataFrame:
        ts = pd.date_range("2024-06-01", periods=n, freq="15min", tz="UTC")
        close = pd.Series(range(100, 100 + n), dtype=float)
        return pd.DataFrame(
            {
                "open_time": ts,
                "open": close,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1.0,
            }
        )

    def test_resolve_entry_qty_caps_by_equity(self):
        params = FibbParams(initial_capital=100_000.0, leverage=2.0)
        self.assertAlmostEqual(
            resolve_entry_qty(2.0, 150_000.0, 100_000.0, params),
            100_000.0 * 2.0 / 150_000.0,
        )
        self.assertAlmostEqual(
            resolve_entry_qty(2.0, 250_000.0, 100_000.0, params),
            100_000.0 * 2.0 / 250_000.0,
        )
        self.assertAlmostEqual(
            resolve_entry_qty(0.01, 200_000.0, 1.0, params), 1.0 * 2.0 / 200_000.0
        )

    def test_bracket_prices_long(self):
        params = FibbParams(tp_mode=0, tp_pct=0.01, sl_pct=0.01)
        tp, sl = bracket_prices("LONG", 100.0, params)
        self.assertAlmostEqual(tp, 101.0)
        self.assertAlmostEqual(sl, 99.0)

    def test_refresh_reprice_tp_to_basis(self):
        leg = OpenLeg(
            entry_id="B1 Long",
            side="LONG",
            qty=0.01,
            entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
            entry_price=100.0,
            take_profit_price=100.5,
            stop_loss_price=None,
            band="bott1",
            take_profit_band="",
        )
        open_legs = {"B1 Long": leg}
        bar = pd.Series({"basis": 99.0})
        refresh_reprice_tp_to_basis(
            open_legs, bar, FibbParams(tp_mode=1)
        )
        self.assertAlmostEqual(leg.take_profit_price, 99.0)
        self.assertEqual(leg.take_profit_band, "basis")
        refresh_reprice_tp_to_basis(
            open_legs, bar, FibbParams(tp_mode=0)
        )
        self.assertAlmostEqual(leg.take_profit_price, 99.0)

    def test_entry_short_requires_approach_from_prev_channel(self):
        from fibb_trading.core.fibb_logic import _touch_short

        curr = pd.Series({"high": 106.0, "top1": 101.0, "top2": 105.0, "basis": 100.0})
        # 僅穿越 top2，前一根已在 top2 區徘徊（未從 top1 來）
        prev_wander = pd.Series(
            {"high": 105.0, "top1": 100.0, "top2": 104.0, "basis": 99.0}
        )
        self.assertFalse(_touch_short(curr, prev_wander, "top2", "T2 Short"))
        # 自 top1 區抵達並穿越 top2
        prev_from_top1 = pd.Series(
            {"high": 101.0, "top1": 100.0, "top2": 104.0, "basis": 99.0}
        )
        self.assertTrue(_touch_short(curr, prev_from_top1, "top2", "T2 Short"))

    def test_trade_sides_filters_entry_signals(self):
        curr = pd.Series(
            {
                "high": 106.0,
                "low": 79.0,
                "top1": 101.0,
                "top2": 105.0,
                "top3": 110.0,
                "bott1": 90.0,
                "bott2": 85.0,
                "bott3": 80.0,
                "basis": 100.0,
            }
        )
        prev = pd.Series(
            {
                "high": 100.0,
                "low": 90.0,
                "top1": 100.0,
                "top2": 104.0,
                "top3": 109.0,
                "bott1": 90.0,
                "bott2": 85.0,
                "bott3": 79.5,
                "basis": 99.0,
            }
        )
        open_ids: set = set()

        short_only = FibbParams(trade_sides="short")
        long_only = FibbParams(trade_sides="long")
        self.assertTrue(side_entry_allowed(short_only, "SHORT"))
        self.assertFalse(side_entry_allowed(short_only, "LONG"))
        self.assertEqual(len(entry_legs_for_trade(short_only)), 3)
        self.assertEqual(len(entry_legs_for_trade(long_only)), 3)

        short_signals = detect_entry_signals(curr, prev, open_ids, short_only)
        long_signals = detect_entry_signals(curr, prev, open_ids, long_only)
        self.assertTrue(all(s[1] == "SHORT" for s in short_signals))
        self.assertTrue(all(s[1] == "LONG" for s in long_signals))

    def test_entry_long_requires_approach_from_prev_channel(self):
        from fibb_trading.core.fibb_logic import _touch_long

        curr = pd.Series({"low": 79.0, "bott2": 85.0, "bott3": 80.0, "bott1": 90.0})
        # 前一根已在 bott3 之下，僅算穿越但非自 bott2 抵達
        prev_already_below = pd.Series(
            {"low": 78.0, "bott1": 90.0, "bott2": 85.0, "bott3": 80.0}
        )
        self.assertFalse(_touch_long(curr, prev_already_below, "bott3", "B3 Long"))
        prev_from_bott2 = pd.Series(
            {"low": 86.0, "bott1": 90.0, "bott2": 85.0, "bott3": 79.5}
        )
        self.assertTrue(_touch_long(curr, prev_from_bott2, "bott3", "B3 Long"))

    def test_channel_tp_two_gap_offset(self):
        from fibb_trading.core.fibb_config import channel_tp_target_band

        off = 2
        self.assertEqual(channel_tp_target_band("B3 Long", "LONG", off), "bott1")
        self.assertEqual(channel_tp_target_band("B1 Long", "LONG", off), "top1")
        self.assertEqual(channel_tp_target_band("T3 Short", "SHORT", off), "top1")
        self.assertEqual(channel_tp_target_band("T1 Short", "SHORT", off), "bott1")
        self.assertEqual(channel_tp_target_band("B2 Long", "LONG", off), "basis")
        self.assertEqual(channel_tp_target_band("T2 Short", "SHORT", off), "basis")

    def test_mark_open_legs_unrealized_tracks_worst(self):
        leg = OpenLeg(
            entry_id="B1 Long",
            side="LONG",
            qty=1.0,
            entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
            entry_price=100.0,
            take_profit_price=110.0,
            stop_loss_price=None,
            band="bott1",
        )
        open_legs = {"B1 Long": leg}
        mark_open_legs_unrealized(open_legs, bar_low=95.0, bar_high=102.0)
        self.assertAlmostEqual(leg.worst_unrealized_gross, -5.0)
        total = mark_open_legs_unrealized(open_legs, bar_low=98.0, bar_high=101.0)
        self.assertAlmostEqual(leg.worst_unrealized_gross, -5.0)
        self.assertAlmostEqual(total, -2.0)

    def test_summarize_trades_holding_and_unrealized(self):
        trades = pd.DataFrame(
            {
                "gross_pnl": [10.0, -3.0],
                "net_pnl": [9.0, -4.0],
                "win": [True, False],
                "take_profit_hit": [True, False],
                "stop_loss_hit": [False, True],
                "holding_bars": [4, 20],
                "max_unrealized_gross": [-5.0, -12.0],
            }
        )
        trades["drawdown"] = [0.0, -4.0]
        trades.attrs["portfolio_max_unrealized_gross"] = -18.0
        summary = summarize_trades(trades, bar_minutes=15)
        self.assertEqual(summary["avg_holding_bars"], 12.0)
        self.assertEqual(summary["max_holding_bars"], 20)
        self.assertEqual(summary["avg_holding_minutes"], 180.0)
        self.assertEqual(summary["max_holding_minutes"], 300)
        self.assertAlmostEqual(summary["avg_max_unrealized_loss"], -8.5)
        self.assertAlmostEqual(summary["max_unrealized_loss"], -12.0)
        self.assertAlmostEqual(summary["portfolio_max_unrealized_loss"], -18.0)

    def test_channel_tp_offset_from_params(self):
        params = FibbParams(tp_mode=3, channel_tp_offset=1)
        tp, band = resolve_take_profit(
            "B3 Long", "LONG", 80.0, pd.Series({"bott2": 85.0, "bott3": 80.0}), params
        )
        self.assertEqual(band, "bott2")
        self.assertAlmostEqual(tp, 85.0)

    def test_channel_tp_t1_short_targets_bott1(self):
        bar = pd.Series(
            {
                "basis": 95.0,
                "top1": 100.0,
                "top2": 105.0,
                "top3": 110.0,
                "bott1": 90.0,
                "bott2": 85.0,
                "bott3": 80.0,
            }
        )
        params = FibbParams(tp_mode=2)
        tp, band = resolve_take_profit("T1 Short", "SHORT", 100.0, bar, params)
        self.assertEqual(band, "bott1")
        self.assertAlmostEqual(tp, 90.0)
        leg = OpenLeg(
            entry_id="T1 Short",
            side="SHORT",
            qty=1.0,
            entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
            entry_price=100.0,
            take_profit_price=tp,
            stop_loss_price=None,
            band="top1",
            take_profit_band="bott1",
        )
        price, reason = try_exit_leg(leg, high=99.0, low=89.0)
        self.assertEqual(reason, "TAKE_PROFIT")
        self.assertAlmostEqual(price, 90.0)

    def test_channel_tp_fixed_does_not_refresh_each_bar(self):
        leg = OpenLeg(
            entry_id="B1 Long",
            side="LONG",
            qty=0.01,
            entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
            entry_price=80.0,
            take_profit_price=100.0,
            stop_loss_price=None,
            band="bott1",
            take_profit_band="top1",
        )
        open_legs = {"B1 Long": leg}
        bar = pd.Series({"top1": 98.0, "bott1": 80.0})
        refresh_channel_take_profits(open_legs, bar, FibbParams(tp_mode=3))
        self.assertAlmostEqual(leg.take_profit_price, 100.0)
        refresh_channel_take_profits(open_legs, bar, FibbParams(tp_mode=2))
        self.assertAlmostEqual(leg.take_profit_price, 98.0)

    def test_channel_tp_b3_long_targets_bott1(self):
        bar = pd.Series(
            {
                "basis": 95.0,
                "top1": 100.0,
                "bott1": 90.0,
                "bott2": 85.0,
                "bott3": 80.0,
            }
        )
        params = FibbParams(tp_mode=2)
        tp, band = resolve_take_profit("B3 Long", "LONG", 80.0, bar, params)
        self.assertEqual(band, "bott1")
        self.assertAlmostEqual(tp, 90.0)

    def test_channels_have_columns(self):
        df = compute_fibb_channels(self._sample_bars(), FibbParams(length=5))
        self.assertIn("top1", df.columns)
        self.assertFalse(pd.isna(df["top1"].iloc[-1]))

    def test_atr_uses_wilder_rma_not_sma(self):
        tr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        length = 3
        wilder = compute_rma(tr, length)
        sma = tr.rolling(length, min_periods=length).mean()
        self.assertAlmostEqual(float(wilder.iloc[3]), 8.0 / 3.0)
        self.assertAlmostEqual(float(sma.iloc[3]), 3.0)

    def test_exit_sl_priority(self):
        leg = OpenLeg(
            entry_id="B1 Long",
            side="LONG",
            qty=1.0,
            entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
            entry_price=100.0,
            take_profit_price=100.5,
            stop_loss_price=99.5,
            band="bott1",
            sl_use_channel=False,
        )
        price, reason = try_exit_leg(leg, high=101.0, low=99.0)
        self.assertEqual(reason, "STOP_LOSS")

    def test_outer_legs_skip_pct_stop_in_deferred_sl_mode(self):
        params = FibbParams(use_deferred_channel_sl=True, tp_mode=0)
        for entry_id in ("T1 Short", "T3 Short", "B1 Long", "B3 Long"):
            self.assertFalse(uses_deferred_channel_sl(entry_id, params))
        self.assertTrue(uses_deferred_channel_sl("T2 Short", params))
        self.assertTrue(uses_deferred_channel_sl("B2 Long", params))

    def test_t2_short_channel_stop_after_t3_touch(self):
        ts = pd.date_range("2024-06-01", periods=6, freq="15min", tz="UTC")
        top2 = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        top3 = [105.0, 105.0, 105.0, 110.0, 105.0, 105.0]
        klines = pd.DataFrame(
            {
                "open_time": ts,
                "open": [102.0] * 6,
                "high": [102.0, 102.0, 102.0, 111.0, 99.0, 102.0],
                "low": [101.0] * 6,
                "close": [102.0] * 6,
                "volume": 1.0,
                "top2": top2,
                "top3": top3,
            }
        )
        params = FibbParams(
            length=2,
            tp_pct=0.01,
            sl_pct=0.01,
            use_deferred_channel_sl=True,
            tp_mode=0,
        )
        leg = OpenLeg(
            entry_id="T2 Short",
            side="SHORT",
            qty=2.0,
            entry_time=ts[1],
            entry_price=102.0,
            take_profit_price=take_profit_price_pct("SHORT", 102.0, params),
            stop_loss_price=None,
            band="top2",
            take_profit_band="",
        )
        open_legs = {"T2 Short": leg}
        arm_deferred_channel_stops(open_legs, klines.iloc[3], klines.iloc[2], params)
        self.assertEqual(leg.stop_loss_price, 100.0)
        self.assertTrue(leg.sl_use_channel)
        price, reason = try_exit_leg(leg, high=99.0, low=99.5)
        self.assertEqual(reason, "CHANNEL_STOP")
        self.assertEqual(price, 100.0)

    def test_backtest_returns_columns(self):
        trades = self._run_backtest(n_bars=50)
        if not trades.empty:
            self.assertIn("entry_id", trades.columns)
            self.assertIn("net_pnl", trades.columns)
            self.assertIn("take_profit_band", trades.columns)

    def test_exit_after_backtest_end(self):
        ts = pd.date_range("2024-06-01", periods=8, freq="15min", tz="UTC")
        klines = pd.DataFrame(
            {
                "open_time": ts,
                "open": [100.0] * 8,
                "high": [100.0, 100.0, 100.0, 100.0, 105.0, 100.0, 100.0, 100.0],
                "low": [100.0] * 8,
                "close": [100.0] * 8,
                "volume": 1.0,
            }
        )
        entry_bar = ts[3]
        end_bar = ts[3]

        def fake_signals(curr, _prev, open_ids, _params=None):
            if curr["open_time"] == entry_bar and "B1 Long" not in open_ids:
                return [("B1 Long", "LONG", 1.0, "bott1")]
            return []

        params = FibbParams(length=2, tp_pct=0.005, sl_pct=0.005)
        with patch(
            "fibb_trading.core.fibb_logic.detect_entry_signals", side_effect=fake_signals
        ):
            trades = run_fibb_backtest(klines, ts[0], end_bar, params)
        self.assertFalse(trades.empty)
        self.assertGreater(trades["exit_time"].iloc[0], end_bar)

    def test_backtest_custom_window(self):
        trades = self._run_backtest(
            n_bars=50,
            start="2024-06-01 06:00",
            end="2024-06-01 12:00",
        )
        self.assertIsInstance(trades, pd.DataFrame)

    def test_leg_hold_expired_and_time_stop_exit(self):
        entry_time = self._ts("2024-06-01 00:00")
        leg = OpenLeg(
            entry_id="B1 Long",
            side="LONG",
            qty=1.0,
            entry_time=entry_time,
            entry_price=100.0,
            take_profit_price=200.0,
            stop_loss_price=None,
            band="bott1",
        )
        self.assertFalse(
            leg_hold_expired(leg, self._ts("2024-06-01 23:00"), max_holding_hours=24.0)
        )
        self.assertTrue(
            leg_hold_expired(leg, self._ts("2024-06-02 00:00"), max_holding_hours=24.0)
        )
        self.assertFalse(leg_hold_expired(leg, self._ts("2024-06-02 00:00"), 0.0))
        price, reason = try_time_stop_exit(
            leg, self._ts("2024-06-02 00:00"), 101.5, max_holding_hours=24.0
        )
        self.assertEqual(reason, "TIME_STOP")
        self.assertEqual(price, 101.5)

    def test_backtest_time_stop_after_max_hold(self):
        ts = pd.date_range("2024-06-01", periods=10, freq="15min", tz="UTC")
        klines = pd.DataFrame(
            {
                "open_time": ts,
                "open": [100.0] * 10,
                "high": [100.0] * 10,
                "low": [100.0] * 10,
                "close": [100.0] * 10,
                "volume": 1.0,
            }
        )
        entry_bar = ts[3]

        def fake_signals(curr, _prev, open_ids, _params=None):
            if curr["open_time"] == entry_bar and "B1 Long" not in open_ids:
                return [("B1 Long", "LONG", 1.0, "bott1")]
            return []

        params = FibbParams(
            length=2,
            tp_pct=0.5,
            sl_pct=0.5,
            max_holding_hours=0.25,
        )
        with patch(
            "fibb_trading.core.fibb_logic.detect_entry_signals", side_effect=fake_signals
        ):
            trades = run_fibb_backtest(klines, ts[0], ts[-1], params)
        self.assertFalse(trades.empty)
        self.assertEqual(trades["exit_reason"].iloc[0], "TIME_STOP")
        self.assertTrue(trades["time_stop_hit"].iloc[0])


if __name__ == "__main__":
    unittest.main()
