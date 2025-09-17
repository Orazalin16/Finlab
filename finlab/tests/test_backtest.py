import numpy as np
import pandas as pd

from quantfinlab.backtest import backtest_signals


def test_backtest_shapes():
    # synthetic upward price series
    price = pd.Series(100 * (1 + pd.Series(np.random.normal(0.0005, 0.01, 500))).cumprod())
    signal = pd.Series(1.0, index=price.index)  # always long
    bt = backtest_signals(price, signal, fee_bps=0.5, slippage_bps=1.0)
    assert len(bt.returns) == len(price)
    assert len(bt.equity_curve) == len(price)
