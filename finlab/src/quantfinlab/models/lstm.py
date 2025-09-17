from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class _SeqDataset(Dataset):
    def __init__(self, series: pd.Series, lookback: int = 20):
        x = series.values.astype(np.float32)
        self.X, self.y = [], []
        for i in range(lookback, len(x)):
            self.X.append(x[i - lookback:i])
            self.y.append(x[i])
        self.X = np.stack(self.X) if self.X else np.zeros((0, lookback), dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).unsqueeze(-1), torch.tensor(self.y[idx])


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last step
        out = self.fc(out)
        return out.squeeze(-1)


@dataclass
class LSTMConfig:
    lookback: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 32


def train_lstm(series: pd.Series, cfg: LSTMConfig = LSTMConfig()) -> Tuple[LSTMForecaster, float]:
    ds = _SeqDataset(series.dropna(), lookback=cfg.lookback)
    if len(ds) < 50:
        raise ValueError("Not enough data to train LSTM.")
    model = LSTMForecaster(input_size=1, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model, float(loss.item())


def forecast_one(model: LSTMForecaster, recent_series: pd.Series, lookback: int = 20) -> float:
    x = torch.tensor(recent_series.values[-lookback:].astype(np.float32)).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        yhat = model(x).item()
    return float(yhat)
