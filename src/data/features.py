import numpy as np
import pandas as pd


def log_returns(close):
    close = pd.Series(close).astype(float)
    lr = np.log(close / close.shift(1))
    print(
        f"Log returns computed: {lr.notna().sum()} valid observations, {lr.isna().sum()} NaN"
    )
    return lr


def forward_log_return(close, horizon):
    """Target: log(Close[t+H] / Close[t])"""
    close = pd.Series(close).astype(float)
    target = np.log(close.shift(-horizon) / close)
    print(
        f"Forward log return (h={horizon}): {target.notna().sum()} valid targets, {target.isna().sum()} NaN (expected {horizon} at end)"
    )
    return target


def parkinson_volatility(high, low, window):
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    hl_ratio = (high / low).replace(0, np.nan)
    squared_log = np.log(hl_ratio) ** 2

    park_vol = np.sqrt(
        squared_log.rolling(window, min_periods=window).mean() / (4.0 * np.log(2))
    )

    print(
        f"Parkinson vol (window={window}): {park_vol.notna().sum()} valid observations"
    )
    return park_vol
