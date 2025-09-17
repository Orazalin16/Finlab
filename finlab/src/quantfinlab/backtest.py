from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio, hit_ratio


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.Series
    costs: pd.Series

    def summary(self) -> dict:
        return {
            "CAGR": float(cagr(self.equity_curve)),
            "Sharpe": float(sharpe_ratio(self.returns)),
            "Sortino": float(sortino_ratio(self.returns)),
            "MaxDrawdown": float(max_drawdown(self.equity_curve)),
            "Calmar": float(calmar_ratio(self.returns, self.equity_curve)),
            "HitRatio": float(hit_ratio(self.returns)),
        }


def backtest_signals(
    price: pd.Series,
    signal: pd.Series,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    allow_short: bool = False,
    position_cap: float = 1.0,
) -> BacktestResult:
    """
    Vectorized backtest for a single asset and daily signals.

    Parameters
    ----------
    price : pd.Series
        Price series (Adj Close recommended).
    signal : pd.Series
        Desired position in [-1, 1] or [0, 1]. Will be clipped and forward-filled.
    fee_bps : float
        One-way fee in basis points on traded notional.
    slippage_bps : float
        Additional cost applied on position changes.
    allow_short : bool
        If False, negative signals are clipped to 0 (long-only).
    position_cap : float
        Cap absolute position size.

    Returns
    -------
    BacktestResult
    """
    price = price.dropna()
    ret = price.pct_change().fillna(0.0)

    sig = signal.reindex(price.index).ffill().fillna(0.0)
    if not allow_short:
        sig = sig.clip(lower=0.0, upper=position_cap)
    else:
        sig = sig.clip(lower=-position_cap, upper=position_cap)

    trades = sig.diff().abs().fillna(sig.abs())
    # Convert bps costs to decimal per trade
    total_cost_bps = fee_bps + slippage_bps
    cost = trades * (total_cost_bps / 1e4)

    strat_ret = sig.shift(1).fillna(0.0) * ret - cost
    equity = (1 + strat_ret).cumprod()

    return BacktestResult(returns=strat_ret.rename("strategy_return"),
                          equity_curve=equity.rename("equity"),
                          positions=sig.rename("position"),
                          costs=cost.rename("cost"))
