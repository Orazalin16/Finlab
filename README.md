# quant-finlab — Financial Data Analysis & Backtesting

A compact project showcasing **data ingestion**, **time‑series modeling** (ARIMA, GARCH, LSTM), and **strategy backtesting** (momentum & mean‑reversion) on real market data.

> Built with Python: `pandas`, `numpy`, `statsmodels`, `arch`, `torch`, `matplotlib`, `plotly`, and `yfinance`.

---

## Highlights

- **Data**: Download equities/crypto/FX with `yfinance`.
- **Models**:
  - **ARIMA** (directional forecasting),
  - **GARCH(1,1)** (volatility forecasting),
  - **LSTM** (sequence modeling, PyTorch).
- **Strategies**:
  - **Momentum** (SMA cross / long‑only variant with vol‑scaling),
  - **Mean Reversion** (z‑score around a rolling mean).
- **Backtesting**: Vectorized engine with fees, slippage, and rich risk metrics.
- **Plots**: Equity curve, drawdowns, and price + signals.
- **Examples**: One‑file quickstart that runs end‑to‑end.

---

## Installation

```bash
# 1) Clone your fork
git clone https://github.com/<your-username>/quant-finlab.git
cd quant-finlab

# 2) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

> Optional (for dev quality): `pip install black isort flake8 pytest`

---

## Quickstart

```bash
python examples/quickstart.py --ticker AAPL --start 2018-01-01 --end 2025-09-01
```

What it does:

1. Downloads daily prices for `--ticker`.
2. Trains **ARIMA**, **GARCH**, and a small **LSTM** on (log) returns.
3. Builds **momentum** and **mean‑reversion** signals with a volatility filter.
4. Runs a **backtest** with fees and slippage.
5. Prints **risk metrics** and shows plots (interactive Plotly + Matplotlib).

Example output (abridged):

```
CAGR: 12.3%
Sharpe: 1.35
Sortino: 2.12
Max Drawdown: -18.4%
Hit Ratio: 0.54
```

---

## Project Structure

```
quant-finlab/
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ pyproject.toml              # tooling config (black, isort, pytest)
├─ src/quantfinlab/
│  ├─ __init__.py
│  ├─ data.py                  # data download & caching
│  ├─ features.py              # returns, SMA/EMA/RSI, z-score, vol
│  ├─ plotting.py              # equity, drawdown, signal overlays
│  ├─ backtest.py              # vectorized backtester with costs
│  ├─ metrics.py               # Sharpe/Sortino/Max DD/CAGR/Hit
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ arima.py              # statsmodels ARIMA wrapper
│  │  ├─ garch.py              # arch GARCH(1,1) wrapper
│  │  └─ lstm.py               # PyTorch LSTM forecaster
│  └─ strategies/
│     ├─ __init__.py
│     ├─ momentum.py           # long-only momentum with vol-scaling
│     └─ mean_reversion.py     # z-score mean-reversion signals
├─ examples/
│  └─ quickstart.py            # end-to-end demo
└─ tests/
   ├─ test_metrics.py
   └─ test_backtest.py
```

---


## Notes & Disclaimers

- LSTM and ARIMA examples are intentionally light; hyperparameters are modest so the example runs quickly. Improve them for deeper analyses.
- `yfinance` sources Yahoo Finance data; availability/accuracy may vary.

---

