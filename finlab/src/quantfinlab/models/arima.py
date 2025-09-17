from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ArimaForecaster:
    def __init__(self, order=(1, 0, 1)):
        self.order = order
        self._model = None
        self._fit = None

    def fit(self, y: pd.Series):
        y = pd.Series(y).dropna()
        if len(y) < sum(self.order) + 3:
            raise ValueError("Not enough data to fit ARIMA.")
        self._model = ARIMA(y, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self._fit = self._model.fit()
        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        fc = self._fit.get_forecast(steps=steps)
        mean = fc.predicted_mean
        mean.index = range(len(mean))  # relative horizon
        return mean

    def forecast_one(self) -> float:
        return float(self.forecast(steps=1).iloc[0])
