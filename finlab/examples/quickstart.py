from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from quantfinlab.data import get_price_data, to_close_series
from quantfinlab.features import log_returns
from quantfinlab.models.arima import ArimaForecaster
from quantfinlab.models.garch import GarchVolModel
from quantfinlab.models.lstm import LSTMConfig, train_lstm, forecast_one
from quantfinlab.strategies.momentum import momentum_long_only
from quantfinlab.strategies.mean_reversion import mean_reversion
from quantfinlab.backtest import backtest_signals
from quantfinlab.plotting import plot_equity_curve, plot_drawdown, plot_price_with_signals
from quantfinlab.metrics import cagr, sharpe_ratio, sortino_ratio, max_drawdown, hit_ratio, calmar_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    print(f"Downloading {args.ticker}...")
    data = get_price_data(args.ticker, start=args.start, end=args.end)
    price = to_close_series(data, args.ticker)
    ret = log_returns(price)

    print("Fitting ARIMA...")
    arima = ArimaForecaster(order=(1, 0, 1)).fit(ret)
    print("Next-step ARIMA forecast (log-return):", round(arima.forecast_one(), 6))

    print("Fitting GARCH...")
    garch = GarchVolModel().fit(ret)
    print("Forecast annualized vol:", round(garch.forecast_vol(), 4))

    print("Training LSTM (short)...")
    model, last_loss = train_lstm(ret, LSTMConfig(epochs=3, lookback=20, hidden_size=16))
    print("LSTM train loss:", round(last_loss, 6))
    print("LSTM next-step forecast:", round(forecast_one(model, ret, lookback=20), 6))

    print("Building strategies...")
    sig_mom = momentum_long_only(price, lookback=50, vol_target=0.15)
    sig_mr = mean_reversion(price, window=20, entry_z=1.0, exit_z=0.25)

    print("Backtesting...")
    bt_mom = backtest_signals(price, sig_mom, fee_bps=1.0, slippage_bps=2.0, allow_short=False)
    bt_mr = backtest_signals(price, sig_mr, fee_bps=1.0, slippage_bps=2.0, allow_short=True)

    def report(name, bt):
        s = bt.summary()
        print(f"\n{name} Summary")
        for k, v in s.items():
            print(f"  {k:>12}: {v:.4f}")

    report("Momentum (L-only)", bt_mom)
    report("Mean-Reversion", bt_mr)

    print("\nShowing plots... (close figure windows to exit)")
    plot_equity_curve(bt_mom.equity_curve, title="Momentum Equity")
    plot_drawdown(bt_mom.equity_curve, title="Momentum Drawdown")
    plot_price_with_signals(price, sig_mom, title="Price & Momentum Signal")

    plot_equity_curve(bt_mr.equity_curve, title="Mean-Reversion Equity")
    plot_drawdown(bt_mr.equity_curve, title="Mean-Reversion Drawdown")
    plot_price_with_signals(price, sig_mr, title="Price & MR Signal")


if __name__ == "__main__":
    main()
