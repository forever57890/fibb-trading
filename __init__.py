"""
fibb_trading — FiBB 15m BTC 多層通道策略（Fibonacci × ATR 通道）。

- fibb_trading.core：通道計算、進場訊號、回測模擬
- fibb_trading.backtest：Binance K 線回測

執行範例（專案根目錄）::

    python -m backtest.fibb_backtest
    python -m fibb_trading.backtest.fibb_backtest
"""

__all__ = ["__doc__"]
