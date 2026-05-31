import unittest
from unittest.mock import patch

from fibb_trading.trade.binance_futures_trader import BinanceFuturesTrader


class TestPrepareOrderQty(unittest.TestCase):
    def test_prepare_order_qty_accepts_at_min(self):
        trader = BinanceFuturesTrader.__new__(BinanceFuturesTrader)
        with patch.object(
            trader, "get_symbol_filters", return_value=(0.001, 0.1, 0.001)
        ):
            self.assertEqual(trader.prepare_order_qty("BTCUSDT", 0.001), 0.001)
            self.assertEqual(trader.prepare_order_qty("BTCUSDT", 0.0019), 0.001)

    def test_prepare_order_qty_rejects_below_min_after_step(self):
        trader = BinanceFuturesTrader.__new__(BinanceFuturesTrader)
        with patch.object(
            trader, "get_symbol_filters", return_value=(0.001, 0.1, 0.001)
        ):
            self.assertEqual(trader.prepare_order_qty("BTCUSDT", 0.0009), 0.0)
            self.assertEqual(trader.prepare_order_qty("BTCUSDT", 0.0), 0.0)

    def test_round_qty_raises_with_stepped_value(self):
        trader = BinanceFuturesTrader.__new__(BinanceFuturesTrader)
        with patch.object(
            trader, "get_symbol_filters", return_value=(0.001, 0.1, 0.001)
        ):
            with self.assertRaises(ValueError) as ctx:
                trader.round_qty("BTCUSDT", 0.0005)
            self.assertIn("0.0", str(ctx.exception))
            self.assertIn("minQty 0.001", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
