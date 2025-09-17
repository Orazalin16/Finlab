from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


class GarchVolModel:
    """
    GARCH(1,1) on (log) returns. Returns annualized vol forecast (approx).

    Note: The model forecasts conditional daily variance; we scale to annualized vol.
    """
    def __init__(self):
        self._fit = None

    def fit(self, returns: pd.Series):
        r = pd.Series(returns).dropna()
        if len(r) < 50:
            raise ValueError("Need at least ~50 observations for GARCH(1,1).")
        am = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        self._fit = am.fit(disp="off")
        return self

    def forecast_vol(self, horizon: int = 1, trading_days: int = 252) -> float:
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        f = self._fit.forecast(horizon=horizon)
        # daily variance forecast -> daily std
        daily_var = f.variance.values[-1, -1] / (100**2)  # undo scaling
        daily_std = np.sqrt(daily_var)
        return float(daily_std * np.sqrt(trading_days))
