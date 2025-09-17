from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(price: pd.Series, fillna: bool = True) -> pd.Series:
    r = np.log(price).diff()
    if fillna:
        r = r.fillna(0.0)
    return r.rename("log_ret")


def simple_returns(price: pd.Series) -> pd.Series:
    return price.pct_change().fillna(0.0).rename("ret")


def sma(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window).mean().rename(f"SMA_{window}")


def ema(price: pd.Series, window: int) -> pd.Series:
    return price.ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")


def rolling_vol(ret: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    vol = ret.rolling(window).std() * np.sqrt(trading_days)
    return vol.rename(f"VOL_{window}")


def zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std().replace(0, np.nan)
    z = (x - mu) / sd
    return z.rename(f"Z_{window}")


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    # Standard RSI implementation
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename(f"RSI_{window}")
