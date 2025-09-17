import numpy as np
import pandas as pd

from quantfinlab.metrics import cagr, sharpe_ratio, sortino_ratio, max_drawdown, hit_ratio, calmar_ratio


def test_metrics_run_without_error():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 0.01, 252*2))
    eq = (1 + r).cumprod()
    assert not np.isnan(cagr(eq))
    assert not np.isnan(sharpe_ratio(r))
    assert not np.isnan(sortino_ratio(r))
    assert not np.isnan(max_drawdown(eq))
    assert not np.isnan(hit_ratio(r))
    assert not np.isnan(calmar_ratio(r, eq))
