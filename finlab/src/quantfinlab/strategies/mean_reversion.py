from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import zscore


def mean_reversion(price: pd.Series, window: int = 20, entry_z: float = 1.0, exit_z: float = 0.25) -> pd.Series:
    """
    Symmetric mean-reversion: go long when price is "too low" (z <= -entry),
    short when "too high" (z >= +entry), and flatten near mean (|z| <= exit).

    Returns a position in [-1, 1].
    """
    z = zscore(price, window=window)
    pos = pd.Series(0.0, index=price.index, name=f"mr_{window}")

    long_cond = z <= -entry_z
    short_cond = z >= entry_z
    flat_cond = z.abs() <= exit_z

    pos[long_cond] = 1.0
    pos[short_cond] = -1.0
    # Keep previous position until exit
    pos = pos.replace(to_replace=0.0, method="ffill").fillna(0.0)
    pos[flat_cond] = 0.0
    pos = pos.replace(to_replace=0.0, method="ffill").fillna(0.0)
    return pos
