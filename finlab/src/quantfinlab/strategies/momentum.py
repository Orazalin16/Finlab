from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import sma, log_returns, rolling_vol


def momentum_long_only(price: pd.Series, lookback: int = 50, vol_target: float = 0.15) -> pd.Series:
    """
    Long-only momentum: hold 1 when price > SMA(lookback), else 0.
    Position is scaled to target annualized vol via ex-ante scaling.

    Parameters
    ----------
    price : pd.Series
    lookback : int
    vol_target : float
        Target annualized volatility (e.g., 0.15 = 15%).

    Returns
    -------
    pd.Series of position in [0, 1]
    """
    ma = sma(price, lookback)
    raw_signal = (price > ma).astype(float)

    # Volatility scaling using last 20-day log-return vol
    ret = log_returns(price)
    vol = rolling_vol(ret, window=20).reindex(price.index).fillna(method="bfill")
    # scale factor: vol_target / current_vol, cap at 1.0
    scale = (vol_target / (vol.replace(0.0, np.nan))).clip(upper=1.0)
    scale = scale.fillna(1.0)
    position = raw_signal * scale
    return position.rename(f"momL_{lookback}")
