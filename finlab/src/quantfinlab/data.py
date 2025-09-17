from __future__ import annotations

import os
import pathlib
from typing import Iterable, List, Optional
import pandas as pd
import yfinance as yf


def _ensure_cache_dir(path: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_price_data(
    tickers: Iterable[str] | str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_dir: str | os.PathLike = "data_cache",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV price data using yfinance.
    Returns a DataFrame with columns MultiIndex (Ticker, [Open, High, Low, Close, Adj Close, Volume]).

    Parameters
    ----------
    tickers : list[str] | str
        Tickers like ["AAPL", "MSFT"]. FX/crypto supported by Yahoo tickers (e.g. "EURUSD=X", "BTC-USD").
    start, end : str
        Date range (YYYY-MM-DD).
    interval : str
        "1d", "1h", etc.
    cache_dir : str
        Directory to cache CSVs.
    force_download : bool
        If True, ignore cache and pull fresh data.

    Notes
    -----
    Yahoo! Finance has occasional data gaps. This is educational. Verify before production use.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    cache_dir = _ensure_cache_dir(cache_dir)

    frames = []
    for t in tickers:
        cache_file = cache_dir / f"{t}_{interval}_{start}_{end or 'latest'}.csv"
        if cache_file.exists() and not force_download:
            df = pd.read_csv(cache_file, parse_dates=["Date"]).set_index("Date")
        else:
            df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for ticker {t}.")
            df.to_csv(cache_file, index=True)
        df.columns = pd.MultiIndex.from_product([[t], df.columns])
        frames.append(df)

    data = pd.concat(frames, axis=1).sort_index()
    # Forward-fill missing values for continuity (still log gaps accordingly).
    data = data.ffill()
    return data


def to_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Convenience: extract Close for a single ticker from the multiindex columns.
    """
    if isinstance(data.columns, pd.MultiIndex):
        return data[ticker]["Adj Close"].rename(ticker) if "Adj Close" in data[ticker].columns else data[ticker]["Close"].rename(ticker)
    # If single-index columns (already single ticker)
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    return data[col].rename(ticker)
