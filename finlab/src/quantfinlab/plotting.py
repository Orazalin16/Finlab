from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()


def plot_drawdown(equity: pd.Series, title: str = "Drawdown"):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig, ax = plt.subplots(figsize=(10, 2.5))
    dd.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def plot_price_with_signals(price: pd.Series, signal: pd.Series, title: str = "Price & Signals"):
    # Interactive Plotly chart
    df = pd.DataFrame({"price": price, "signal": signal.reindex(price.index).ffill().fillna(0.0)})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=(df["signal"] * df["price"]), name="Signal x Price", mode="lines", yaxis="y2"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Signal × Price", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
        height=500,
    )
    fig.show()
