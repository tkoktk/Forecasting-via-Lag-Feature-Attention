import numpy as np
import pandas as pd


def log_returns(close):
    close = pd.Series(close).astype(float)
    lr = np.log(close / close.shift(1))
    return lr


def forward_log_return(close, horizon):
    """Target: log(Close[t+H] / Close[t])"""
    close = pd.Series(close).astype(float)
    target = np.log(close.shift(-horizon) / close)

    return target


def parkinson_volatility(high, low, window):
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    hl_ratio = (high / low).replace(0, np.nan)
    squared_log = np.log(hl_ratio) ** 2

    park_vol = np.sqrt(
        squared_log.rolling(window, min_periods=window).mean() / (4.0 * np.log(2))
    )

    return park_vol
