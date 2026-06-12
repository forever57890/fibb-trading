import unittest
from unittest.mock import MagicMock

from fibb_trading.trade.exchange import (
    _open_leg_filled_qty,
    _open_leg_remaining_qty,
    _order_fill_qty,
)


class TestOpenLegPositionDelta(unittest.TestCase):
    def test_remaining_zero_when_position_filled(self):
        trader = MagicMock()
        trader.get_position_amount.return_value = -0.03
        self.assertEqual(
            _open_leg_remaining_qty(trader, "BTCUSDT", "SHORT", 0.0, 0.03),
            0.0,
        )
        self.assertEqual(
            _open_leg_filled_qty(trader, "BTCUSDT", "SHORT", 0.0, 0.03),
            0.03,
        )

    def test_remaining_partial_fill(self):
        trader = MagicMock()
        trader.get_position_amount.return_value = -0.01
        self.assertAlmostEqual(
            _open_leg_remaining_qty(trader, "BTCUSDT", "SHORT", 0.0, 0.03),
            0.02,
        )

    def test_order_fill_qty_falls_back_to_query(self):
        trader = MagicMock()
        trader.get_order.return_value = {"executedQty": "0.03"}
        executed = _order_fill_qty(
            trader,
            "BTCUSDT",
            {"executedQty": "0", "orderId": 99},
        )
        self.assertEqual(executed, 0.03)
        trader.get_order.assert_called_once_with("BTCUSDT", 99)


if __name__ == "__main__":
    unittest.main()
