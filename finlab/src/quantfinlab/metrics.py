from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def cagr(equity_curve: pd.Series, trading_days: int = 252) -> float:
    equity_curve = _to_series(equity_curve).dropna()
    if equity_curve.empty:
        return np.nan
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / trading_days
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    r = _to_series(returns).dropna()
    return r.std() * np.sqrt(trading_days)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    r = _to_series(returns).dropna()
    excess = r - rf / trading_days
    denom = excess.std()
    return np.nan if denom == 0 or np.isnan(denom) else (excess.mean() * trading_days) / (denom * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    r = _to_series(returns).dropna()
    downside = r[r < 0]
    dd_std = downside.std()
    if dd_std == 0 or np.isnan(dd_std):
        return np.nan
    annual_return = (r.mean() - rf / trading_days) * trading_days
    return annual_return / (dd_std * np.sqrt(trading_days))


def max_drawdown(equity_curve: pd.Series) -> float:
    ec = _to_series(equity_curve).dropna()
    peak = ec.cummax()
    dd = (ec / peak - 1.0)
    return dd.min() if not dd.empty else np.nan


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, trading_days: int = 252) -> float:
    annual_ret = returns.mean() * trading_days
    mdd = abs(max_drawdown(equity_curve))
    return np.nan if mdd == 0 or np.isnan(mdd) else annual_ret / mdd


def hit_ratio(returns: pd.Series) -> float:
    r = _to_series(returns).dropna()
    if len(r) == 0:
        return np.nan
    return (r > 0).mean()
